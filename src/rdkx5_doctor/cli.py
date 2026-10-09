"""All interaction is terminal based; reports remain machine/human readable."""
import argparse
import json
from pathlib import Path
import sys
from .onnx_reader import read_model, ModelError
from .rules import load_ruleset
from .report import analyze, markdown

DEFAULT_RULESET = Path(__file__).parent / 'resources' / 'rulesets' / 'x5-bayes-e'
STATES = ['VIOLATION', 'NEEDS_VERIFICATION', 'NOT_COVERED', 'NO_VIOLATION_FOUND']

def parser():
    p = argparse.ArgumentParser(description='RDK X5 ONNX Doctor：纯终端、只读 Conv2D 静态分析')
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('analyze', help='分析模型，终端摘要 + JSON/Markdown 报告')
    a.add_argument('--model', required=True, type=Path)
    a.add_argument('--ruleset', type=Path, default=DEFAULT_RULESET)
    a.add_argument('--out', type=Path, required=True)
    rules = sub.add_parser('rules', help='规则集操作').add_subparsers(dest='rules_command', required=True)
    validate = rules.add_parser('validate')
    validate.add_argument('--ruleset', type=Path, default=DEFAULT_RULESET)
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
    return p

def load_analysis(path):
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('schema_version') != '1.0':
            raise ValueError('不支持的 analysis schema_version')
        if not all(isinstance(data.get(key), list) for key in ('nodes','tensors','edges','diagnostics','traces')):
            raise ValueError('analysis 缺少节点、Tensor、边或诊断列表')
        node_ids = {n['id'] for n in data['nodes']}
        if len(node_ids) != len(data['nodes']) or any(d['node_id'] not in node_ids for d in data['diagnostics']):
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
            print(json.dumps(rows, ensure_ascii=False, indent=2))
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
            print(json.dumps(trace,ensure_ascii=False,indent=2))
        else:
            print_trace(trace)
        return 0
    tensors = {t['name']:t for t in data['tensors']}
    facts = {'node':node, 'tensors':[tensors[name] for name in dict.fromkeys(node['inputs'] + node['outputs']) if name],
             'diagnostic':diagnostic, 'trace':trace}
    if args.json:
        print(json.dumps(facts,ensure_ascii=False,indent=2))
        return 0
    print(f"节点：{node['id']} / {display(node['original_name'])}\n算子：{display(node['domain'] or '标准 ONNX')} / {display(node['op_type'])}\n状态：{diagnostic['status']}")
    print('参数：' + json.dumps(node.get('conv',node['attributes']),ensure_ascii=False,indent=2))
    print('Tensor：' + json.dumps(facts['tensors'],ensure_ascii=False,indent=2))
    for result in diagnostic['results']:
        print(f"[{result['status']}] {result['rule_id']}：{result['field']} = {display(result['actual'])}，允许 {result['expected']}\n  {result['reason']}\n  来源：{result['source_url']} / {result['source_section']}")
    if not diagnostic['results']:
        print('V1 未检查该算子，不代表 BPU 兼容。')
    for issue in diagnostic['issues']:
        print('元信息问题：' + issue)
    for suggestion in diagnostic['suggestions']:
        print(f"建议（{suggestion['category']}）：{suggestion['text']}\n  需验证：{suggestion['validation_needed']}")
    print_trace(trace)
    print('尚未用实际工具链验证。')
    return 0

def display(value):
    # Escape terminal control sequences in untrusted model strings (including ANSI/OSC).
    return json.dumps(str(value),ensure_ascii=False)[1:-1]

def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command in ('nodes','inspect','trace'):
            return queries(args)
        if args.command == 'rules':
            rules, _ = load_ruleset(args.ruleset)
            print(f'规则校验通过：{rules.ruleset_id} {rules.ruleset_version}（{len(rules.rules)} 条）')
            return 0
        rules, manifest = load_ruleset(args.ruleset)
        ir = read_model(args.model)
        out = args.out.resolve()
        for name in ('analysis.json','report.md'):
            if (out / name).resolve() == args.model.resolve():
                raise ValueError('输出路径不能覆盖原始模型')
        data = analyze(ir, rules, manifest)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'analysis.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        (out / 'report.md').write_text(markdown(data),encoding='utf-8')
        print(f"分析完成：{out}\nConv 状态：{data['summary']['conv_status_counts']}")
        names = {n['id']:n['original_name'] for n in data['nodes']}
        for d in data['diagnostics']:
            if d['status'] in ('VIOLATION','NEEDS_VERIFICATION'):
                print(f"{d['status']} {d['node_id']} / {display(names[d['node_id']])}")
                for r in d['results']:
                    if r['status'] in ('FAIL','UNKNOWN'):
                        print(f"  {r['status']} {r['rule_id']}：{r['field']}={display(r['actual'])}，允许 {r['expected']}；{r['reason']}")
        for trace in data['traces']:
            print_trace(trace)
        for warning in data['unverified_assumptions']:
            print('注意：' + display(warning))
        print('尚未用实际工具链验证。详情：inspect --analysis <analysis.json> --node <节点 ID>')
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        print(f'错误：{exc}', file=sys.stderr)
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
