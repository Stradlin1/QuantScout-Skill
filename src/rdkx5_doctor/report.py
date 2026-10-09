"""Machine facts and Chinese human report from the same evidence."""
from collections import Counter
import html
import json
from .conv_extractor import extract_conv
from .rules import check_node
from .trace import trace_node

LIMITATIONS = ['尚未用实际工具链验证：未经过 OpenExplorer/hb_mapper 实测。',
 'V1 仅检查已收录的标准 ONNX Conv2D 规则，其他算子未覆盖。',
 '静态检查不能确定真实 CPU/BPU 分配、量化误差、延迟、峰值 DDR 或任务精度。',
 '路径只表示数据依赖，不表示下游节点违规。',
 '模型 SHA256 仅覆盖 .onnx 文件；external data 内容不包含在该摘要中。']

def analyze(ir, ruleset, manifest):
    diagnostics = []
    for node in ir.nodes:
        extracted = extract_conv(node, ir.tensors) if node['op_type'] == 'Conv' else None
        if extracted:
            node['conv'] = extracted
        diagnostics.append(check_node(node, extracted, ruleset))
    traces = [trace_node(ir, d['node_id']) for d in diagnostics if d['status'] == 'VIOLATION']
    by_id = {n['id']: n for n in ir.nodes}
    by_diagnostic = {d['node_id']: d for d in diagnostics}
    for trace in traces:
        trace['original_name'] = by_id[trace['node_id']]['original_name']
        trace['evidence'] = [r for r in by_diagnostic[trace['node_id']]['results'] if r['status'] == 'FAIL']
    conv_ids = {n['id'] for n in ir.nodes if n['op_type'] == 'Conv'}
    return {'schema_version': '1.0', 'model': ir.model,
        'ruleset': {'id': ruleset.ruleset_id, 'version': ruleset.ruleset_version,
            'toolchain_version': ruleset.toolchain_version, 'toolchain_verified': False,
            'sources': [s.model_dump() for s in manifest.sources],
            'rules': [r.model_dump(exclude_none=True) for r in ruleset.rules],
            'exclusions': manifest.exclusions, 'review_status': manifest.review_status},
        'nodes': ir.nodes, 'tensors': list(ir.tensors.values()), 'edges': ir.edges,
        'diagnostics': diagnostics, 'traces': traces,
        'summary': {'all_status_counts': dict(Counter(d['status'] for d in diagnostics)),
            'conv_status_counts': dict(Counter(d['status'] for d in diagnostics if d['node_id'] in conv_ids)),
            'conv_node_count': len(conv_ids), 'conv2d_checked_count': sum(d.get('scope') == 'conv2d' for d in diagnostics)},
        'unverified_assumptions': ir.warnings, 'limitations': LIMITATIONS}

def safe(value):
    return html.escape(str(value)).replace('|', '&#124;').replace('\n', ' ')

def markdown(data):
    m, r = data['model'], data['ruleset']
    lines = ['# RDK X5 ONNX Doctor 分析报告', '', '**尚未用实际工具链验证：未经过 OpenExplorer/hb_mapper 实测。**',
        '', '## 1. 模型概况', '', f"- 文件：{safe(m['path'])}", f"- SHA256：`{m['sha256']}`（仅 ONNX 文件）",
        f"- 节点：{m['node_count']}；initializer：{m['initializer_count']}；结构校验：{m['validation']}",
        f"- Opset：{safe(m['opset_imports'])}", f"- 算子计数：{safe(m['operator_counts'])}", '',
        '| 边界 | Tensor | Shape | dtype |', '|---|---|---|---|']
    for kind in ('inputs', 'outputs'):
        for t in m[kind]:
            lines.append(f"| {kind} | {safe(t['name'])} | {safe(t['shape'])} | {safe(t['dtype'])} |")
    lines += ['', '## 2. 本次规则来源', '', f"规则包 `{r['id']}` / `{r['version']}`；工具链版本：`{r['toolchain_version']}`。", '']
    for source in r['sources']:
        lines.append(f"- [{safe(source['title'])}]({source['url']})；版本：{safe(source['version'])}；章节：{safe(source['section'])}")
    lines += ['', '已收录条款：', '', '| 规则 ID | 字段 | 判定方式 |', '|---|---|---|']
    for rule in r['rules']:
        lines.append(f"| {rule['id']} | {rule['field']} | {rule['checkability']} / {rule['check_type']} |")
    lines += ['', '未覆盖或需审查的条款：', ''] + ['- ' + safe(x) for x in r['exclusions']]
    lines += ['', '## 3. Conv2D 检查汇总', '', f"Conv 节点 {data['summary']['conv_node_count']}；确认 Conv2D 并执行检查 {data['summary']['conv2d_checked_count']}。", '', '| 状态 | 个数 |', '|---|---|']
    for status in ('VIOLATION','NEEDS_VERIFICATION','NO_VIOLATION_FOUND','NOT_COVERED'):
        lines.append(f"| {status} | {data['summary']['conv_status_counts'].get(status, 0)} |")
    lines += ['', '`NO_VIOLATION_FOUND` 仅表示已检查规则未发现违规；非 Conv 节点不按通过统计。', '', '## 4. 异常节点清单', '', '| 节点 ID / 名称 | 规则 | 实际值 | 允许值 | 理由 |', '|---|---|---|---|---|']
    nodes = {n['id']: n for n in data['nodes']}
    findings = 0
    for diagnostic in data['diagnostics']:
        for result in diagnostic['results']:
            if result['status'] in ('FAIL','UNKNOWN'):
                findings += 1
                nid = diagnostic['node_id']
                lines.append(f"| {nid} / {safe(nodes[nid]['original_name'])} | {result['rule_id']} ({result['status']}) | {safe(result['actual'])} | {safe(result['expected'])} | {safe(result['reason'])} |")
        for issue in diagnostic['issues']:
            lines.append(f"| {diagnostic['node_id']} | 元信息待验证 | — | — | {safe(issue)} |")
    if not findings:
        lines += ['', '未发现已检查规则违规。']
    lines += ['', '## 5. 节点风险溯源', '']
    for trace in data['traces']:
        lines += [f"### {trace['node_id']}", '', safe(trace['reason']), '', f"直接前驱：{safe(trace['predecessors'])}；直接后继：{safe(trace['successors'])}", f"可达输出：{safe(trace['reachable_outputs'])}；其他路径省略：{trace['paths_truncated']}", '']
        for path in trace['representative_paths']:
            lines += [f"- 输出 `{safe(path['output_tensor'])}`：" + ' → '.join(path['nodes'])]
            for edge in path['edges']:
                lines.append(f"  - {edge['source']} → {edge['target']}，Tensor：{safe(edge['tensors'])}")
    if not data['traces']:
        lines.append('无确定违规源节点，无需生成违规溯源。')
    lines += ['', '## 6. 优化建议', '']
    for diagnostic in data['diagnostics']:
        for suggestion in diagnostic['suggestions']:
            lines += [f"- **{suggestion['node_id']} / {suggestion['category']}**：{safe(suggestion['text'])} 证据：{safe(suggestion['evidence'])}；规则：{safe(suggestion['rule_id'])}；验证：{safe(suggestion['validation_needed'])}"]
    lines += ['', '## 7. 尚不能下结论的事项', ''] + ['- ' + safe(x) for x in data['limitations'] + data['unverified_assumptions']]
    lines += ['', '## 8. 文件索引', '', '- [机器事实 analysis.json](analysis.json)', '- 终端节点列表：`python -m rdkx5_doctor nodes --analysis analysis.json`',
        '- 参数与证据：`python -m rdkx5_doctor inspect --analysis analysis.json --node <节点 ID>`',
        '- 依赖路径：`python -m rdkx5_doctor trace --analysis analysis.json --node <节点 ID>`', '']
    return '\n'.join(lines)
