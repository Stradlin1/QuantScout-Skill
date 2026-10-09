"""All interaction is terminal based; reports remain machine/human readable."""
import argparse
import json
from pathlib import Path
import sys
from .onnx_reader import read_model, ModelError
from .rules import load_ruleset
from .report import analyze, markdown
from .resource_queries import register_commands, run_query
from .text_utils import json_text

DEFAULT_RULESET = Path(__file__).parent / 'resources' / 'rulesets' / 'x5-bayes-e'
STATES = ['VIOLATION', 'NEEDS_VERIFICATION', 'NOT_COVERED', 'NO_VIOLATION_FOUND']

def parser():
    p = argparse.ArgumentParser(description='RDK X5 ONNX Doctor V1.3：纯终端、只读 Shape 推断与多算子静态诊断')
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('analyze', help='分析模型，终端摘要 + JSON/Markdown 报告')
    a.add_argument('--model', required=True, type=Path)
    a.add_argument('--ruleset', type=Path, default=DEFAULT_RULESET)
    a.add_argument('--out', type=Path, required=True)
    rules = sub.add_parser('rules', help='规则集操作').add_subparsers(dest='rules_command', required=True)
    validate = rules.add_parser('validate')
    validate.add_argument('--ruleset', type=Path, default=DEFAULT_RULESET)
    listing = rules.add_parser('list', help='列出已注册算子规则与来源')
    listing.add_argument('--ruleset', type=Path, default=DEFAULT_RULESET)
    listing.add_argument('--operator')
    listing.add_argument('--json', action='store_true')
    nodes = sub.add_parser('nodes', help='列出/搜索/过滤分析结果中的节点')
    nodes.add_argument('--analysis', required=True, type=Path)
    nodes.add_argument('--status', choices=STATES)
    nodes.add_argument('--search', default='', help='搜索 ID、原名、算子或 Tensor，忽略大小写')
    nodes.add_argument('--json', action='store_true', help='JSON 输出，方便管道调用')
    for command, help_text in [('inspect','查看节点参数、Tensor 与逐规则证据'),('trace','查询节点到模型输出的依赖路径')]:
        c = sub.add_parser(command, help=help_text)
        c.add_argument('--analysis', required=True, type=Path)
        c.add_argument('--node', required=True, help='内部 ID 或唯一的原始名称')
        c.add_argument('--json', action='store_true')
    from .shape_queries import register_shape_commands
    register_shape_commands(sub)
    register_commands(sub)
    return p

def load_analysis(path):
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('schema_version') not in ('1.0', '1.1', '1.2', '1.3'):
            raise ValueError('不支持的 analysis schema_version')
        if not all(isinstance(data.get(key), list) for key in ('nodes','tensors','edges','diagnostics','traces')):
            raise ValueError('analysis 缺少节点、Tensor、边或诊断列表')
        node_ids = {n['id'] for n in data['nodes']}
        diagnostic_ids=[d['node_id'] for d in data['diagnostics']]
        if len(node_ids) != len(data['nodes']) or len(diagnostic_ids)!=len(set(diagnostic_ids)) or set(diagnostic_ids)!=node_ids:
            raise ValueError('analysis 节点 ID 不一致')
        return data
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f'无法读取分析报告 {path}：{exc}') from exc

def select_node(data, key):
    matches = [n for n in data['nodes'] if n['id'] == key]
    if not matches:
        matches = [n for n in data['nodes'] if n['original_name'] == key]
    if not matches:
        raise ValueError(f'未找到节点：{key}；使用 nodes --search 搜索')
    if len(matches) != 1:
        raise ValueError(f'原始名称重复：{key}；请使用内部 ID：' + ', '.join(n['id'] for n in matches))
    return matches[0]

def node_trace(data, node):
    # Queries can trace any parsed node, even a NOT_COVERED operator, without claiming it is a violation.
    from .graph_ir import GraphIR
    from .trace import trace_node
    cached = next((t for t in data['traces'] if t['node_id'] == node['id']), None)
    if cached:
        return cached
    ir = GraphIR(data['model'], data['nodes'], {t['name']:t for t in data['tensors']}, data['edges'], [])
    return trace_node(ir, node['id'])

def print_trace(trace):
    print(f"节点：{trace['node_id']}\n直接前驱：{', '.join(trace['predecessors']) or '无'}\n直接后继：{', '.join(trace['successors']) or '无'}")
    print('可达输出：' + (', '.join(display(x) for x in trace['reachable_outputs']) or '无'))
    for path in trace['representative_paths']:
        print(f"输出 {display(path['output_tensor'])}：" + ' → '.join(path['nodes']))
        for edge in path['edges']:
            print(f"  {edge['source']} → {edge['target']}，Tensor：{', '.join(display(x) for x in edge['tensors'])}")
    print(f"其他路径省略：{trace['paths_truncated']}；{trace['reason']}")

def queries(args):
    data = load_analysis(args.analysis)
    diagnostics = {d['node_id']:d for d in data['diagnostics']}
    if args.command == 'nodes':
        query = args.search.casefold()
        nodes = [n for n in data['nodes'] if (not args.status or diagnostics.get(n['id'],{}).get('status') == args.status)
                 and (not query or any(query in str(s).casefold() for s in [n['id'], n['original_name'], n['op_type'], *n['inputs'], *n['outputs']]))]
        rows = [{'id':n['id'], 'name':n['original_name'], 'operator':n['op_type'], 'status':diagnostics[n['id']]['status']} for n in nodes]
        if args.json:
            print(json_text(rows, ensure_ascii=False, indent=2))
        else:
            print('节点 ID\t算子\t状态\t原始名称')
            for row in rows:
                print(f"{row['id']}\t{display(row['operator'])}\t{row['status']}\t{display(row['name'])}")
            print(f'匹配 {len(rows)} 个节点；状态只表示本地规则检查结果。')
        return 0
    node = select_node(data, args.node)
    trace = node_trace(data, node)
    diagnostic = diagnostics[node['id']]
    if args.command == 'trace':
        if args.json:
            print(json_text(trace,ensure_ascii=False,indent=2))
        else:
            print_trace(trace)
        return 0
    tensors = {t['name']:t for t in data['tensors']}
    facts = {'node':node, 'tensors':[tensors[name] for name in dict.fromkeys(node['inputs'] + node['outputs']) if name],
             'diagnostic':diagnostic, 'trace':trace}
    if args.json:
        print(json_text(facts,ensure_ascii=False,indent=2))
        return 0
    print(f"节点：{node['id']} / {display(node['original_name'])}\n算子：{display(node['domain'] or '标准 ONNX')} / {display(node['op_type'])}\n状态：{diagnostic['status']}")
    print('参数与提取证据：' + json_text(node.get('conv',node.get('operator_facts',node['attributes'])),ensure_ascii=False,indent=2))
    print('Tensor：' + json_text(facts['tensors'],ensure_ascii=False,indent=2))
    for result in diagnostic['results']:
        print(f"[{result['status']}] {result['rule_id']}：{result['field']} = {display(result['actual'])}，允许 {result['expected']}\n  {result['reason']}\n  来源：{result['source_url']} / {result['source_section']}")
        if 'source' in result:
            print('  来源版本/字段证据：'+json_text({'source':result['source'],'evidence':result['evidence'],'reason_code':result['reason_code']},ensure_ascii=False))
        else:
            print('  历史报告无完整来源版本/字段证据；不猜补，需重跑 analyze 获取 V1.2 证据。')
    if not diagnostic['results']:
        print('本次规则未覆盖该算子/版本，不代表 BPU 兼容。')
    for issue in diagnostic['issues']:
        print('元信息问题：' + issue)
    for suggestion in diagnostic['suggestions']:
        print(f"建议（{suggestion['category']}）：{suggestion['text']}\n  需验证：{suggestion['validation_needed']}")
    print_trace(trace)
    print('尚未用实际工具链验证。')
    return 0

def display(value):
    # Escape terminal control sequences in untrusted model strings (including ANSI/OSC).
    return json_text(str(value),ensure_ascii=False)[1:-1]

def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command in ('shapes','shape'):
            from .shape_queries import run_shape_query
            return run_shape_query(args,load_analysis(args.analysis),display)
        if args.command in ('tensors','tensor','candidates','candidate'):
            return run_query(args, load_analysis(Path(args.analysis)), display=display, node_trace=node_trace)
        if args.command in ('nodes','inspect','trace'):
            return queries(args)
        if args.command == 'rules':
            rules, _ = load_ruleset(args.ruleset)
            if args.rules_command == 'list':
                if args.operator and args.operator not in rules.by_operator:
                    raise ValueError(f'未注册算子：{args.operator}')
                rows = [{**r.model_dump(exclude_none=True), 'operator': op.operator,
                         'operator_rule_version': op.ruleset_version,
                         'source_version': r.source_version or op.source_version}
                        for op in rules.by_operator.values() if not args.operator or op.operator == args.operator for r in op.rules]
                print(json_text(rows, ensure_ascii=False, indent=2))
                return 0
            print(f'规则校验通过：{rules.ruleset_id} {rules.ruleset_version}（{len(rules.all_rules)} 条）')
            for operator, op in rules.by_operator.items():
                print(f'{operator}: {len(op.rules)} 条；子版本 {op.ruleset_version}；来源 {display(op.source_version)}')
            return 0
        rules, manifest = load_ruleset(args.ruleset)
        ir = read_model(args.model)
        out = args.out.resolve()
        for name in ('analysis.json','report.md'):
            if (out / name).resolve() == args.model.resolve():
                raise ValueError('输出路径不能覆盖原始模型')
        data = analyze(ir, rules, manifest)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'analysis.json').write_text(json_text(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        (out / 'report.md').write_text(markdown(data),encoding='utf-8')
        print(f"分析完成：{out}\nConv 状态：{data['summary']['conv_status_counts']}")
        print('算子\t数量\t无违规\t违规\t待验证\t未覆盖')
        for row in data['summary']['operator_coverage']:
            states=row['status_counts']
            print(f"{display(row['operator'])}\t{row['node_count']}\t{states['NO_VIOLATION_FOUND']}\t{states['VIOLATION']}\t{states['NEEDS_VERIFICATION']}\t{states['NOT_COVERED']}")
        names = {n['id']:n['original_name'] for n in data['nodes']}
        findings=[d for d in data['diagnostics'] if d['status'] in ('VIOLATION','NEEDS_VERIFICATION')]
        for d in findings[:20]:
            if d['status'] in ('VIOLATION','NEEDS_VERIFICATION'):
                print(f"{d['status']} {d['node_id']} / {display(names[d['node_id']])}")
                for r in d['results']:
                    if r['status'] in ('FAIL','UNKNOWN'):
                        print(f"  {r['status']} {r['rule_id']}：{r['field']}={display(r['actual'])}，允许 {r['expected']}；{r['reason']}")
        if len(findings)>20:print(f'其余 {len(findings)-20} 个异常/待验证节点请用 nodes/inspect 或 JSON 查看。')
        for trace in data['traces']:
            print_trace(trace)
        resources = data['resource_analysis']['summary']
        print(f"Tensor 理论原始载荷：输出已知 {resources['model_outputs']['known_bytes_sum']} B；中间已知 {resources['intermediate_activations']['known_bytes_sum']} B；不是峰值内存/BPU/DDR 分配。")
        print(f"优化候选/信息不足观察：{data['optimization_candidates']['summary']['counts_by_classification']}；未改写模型。")
        for warning in data['unverified_assumptions']:
            print('注意：' + display(warning))
        print('尚未用实际工具链验证。详情：inspect --analysis <analysis.json> --node <节点 ID>')
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError, StopIteration) as exc:
        print(f'错误：{display(exc)}', file=sys.stderr)
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
