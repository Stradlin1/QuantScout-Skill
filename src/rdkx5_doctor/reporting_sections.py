"""Bounded Markdown policy over complete saved node evidence."""
from collections import Counter, defaultdict
from pathlib import Path
from .text_utils import printable
import html

STATES=('NO_VIOLATION_FOUND','VIOLATION','NEEDS_VERIFICATION','NOT_COVERED')

def safe(value):
    return (html.escape(printable(value)).replace('|','&#124;').replace('\n',' ')
            .replace('`','&#96;').replace('[','&#91;').replace(']','&#93;'))

def unknown_groups(data):
    groups={}
    for diagnostic in data['diagnostics']:
        if diagnostic['status']!='NEEDS_VERIFICATION':continue
        for result in diagnostic['results']:
            if result['status']!='UNKNOWN' or not result.get('blocking',True):continue
            key=(diagnostic.get('operator','unknown'),result.get('reason_code','LEGACY_UNKNOWN'))
            group=groups.setdefault(key,{'operator':key[0],'reason_code':key[1],'node_ids':[], 'reason':result['reason']})
            if diagnostic['node_id'] not in group['node_ids']:group['node_ids'].append(diagnostic['node_id'])
        if diagnostic['issues']:
            key=(diagnostic.get('operator','unknown'),'ONNX_OR_METADATA_ISSUE')
            group=groups.setdefault(key,{'operator':key[0],'reason_code':key[1],'node_ids':[],'reason':diagnostic['issues'][0]})
            group['node_ids'].append(diagnostic['node_id'])
    return list(groups.values())

def markdown(data, violation_limit=50, representative_limit=10):
    model,rules,summary=data['model'],data['ruleset'],data['summary']
    nodes={n['id']:n for n in data['nodes']}
    records={t['name']:t for t in data['resource_analysis']['tensor_records']}
    lines=['# RDK X5 ONNX Doctor V1.2 分析报告','',
           '**静态规则检查不是工具链 CPU/BPU 分配，也不意味着模型不能运行。未经过 OpenExplorer/hb_mapper 实测。**','',
           '## 0. 执行范围与核心结论','',
           f"模型 `{safe(Path(model['path']).name)}`；SHA256 `{model['sha256']}`（仅 ONNX protobuf）。",
           f"checker={safe(model['validation'])}；opset={safe(model['opset_imports'])}；节点 {model['node_count']}；Tensor {len(data['tensors'])}；输入 {len(model['inputs'])} / 输出 {len(model['outputs'])}。",
           f"规则入口覆盖 {summary['covered_node_count']}；未覆盖 {summary['uncovered_node_count']}；确定违规 {summary['all_status_counts'].get('VIOLATION',0)}；待验证 {summary['all_status_counts'].get('NEEDS_VERIFICATION',0)}。",
           f"Registry {safe(rules['version'])}；子版本 {safe(rules['operator_rule_versions'])}；工具链 {safe(rules['toolchain_version'])}，verified=false。",
           '覆盖表示静态规则范围，不是 BPU 运行比例。','',
           '## 1. 算子覆盖与结论矩阵','',
           '| 算子 | 数量 | 已检查无违规 | 违反硬约束 | 待验证 | 未覆盖 | 规则 / 来源版本 |',
           '|---|---|---|---|---|---|---|']
    for row in summary['operator_coverage']:
        counts=row['status_counts']
        lines.append('| '+ ' | '.join([safe(row['operator']),str(row['node_count']),*[str(counts[s]) for s in STATES],safe(row['rule_version'])+' / '+safe(row['source_version'])])+' |')
    lines+=['','四状态按节点守恒；NO_VIOLATION_FOUND 只说明已执行条件未发现违规。','',
            '## 2. 确定违反 BPU 约束的节点','']
    violations=[d for d in data['diagnostics'] if d['status']=='VIOLATION']
    for diagnostic in violations[:violation_limit]:
        nid=diagnostic['node_id'];node=nodes[nid]
        lines += [f"### {nid} / {safe(node['original_name'])} / {safe(node['op_type'])}",'',
                  f"输入 {safe(node['inputs'])}；输出 {safe(node['outputs'])}。"]
        for r in diagnostic['results']:
            if r['status']!='FAIL':continue
            source=r['source']
            lines.append(f"- `{r['rule_id']}`：实际 {safe(r['actual'])}；允许 {safe(r['expected'])}；{safe(r['reason'])}。证据 {safe(r['evidence'])}；[{safe(source['document'])}]({source['url']}) / {safe(source['version'])} / {safe(source['section'])}。")
        trace=next((t for t in data['traces'] if t['node_id']==nid),None)
        if trace:lines.append(f"可达输出 {safe(trace['reachable_outputs'])}；前驱 {safe(trace['predecessors'])} / 后继 {safe(trace['successors'])}。完整 Tensor 代表路径在 JSON/trace；依赖不表示下游同样违规。")
        lines+=['回源位置：训练模型定义/forward 或导出脚本；本项目不修改 ONNX。','']
    if not violations:lines.append('已收录且实际执行的 X5 BPU 约束未发现确定违规；尚有未覆盖/待验证项。')
    elif len(violations)>violation_limit:lines.append(f'其余 {len(violations)-violation_limit} 个违规节点请用 CLI/JSON 查看；完整事实未截断。')
    lines+=['','## 3. 需要更多元信息或工具链确认','',
            '| 算子 / 原因码 | 节点数 | 代表节点 | 缺失条件 |','|---|---|---|---|']
    groups=unknown_groups(data)
    for group in groups:
        ids=group['node_ids'];omitted=len(ids)-representative_limit
        sample=', '.join(ids[:representative_limit])+(f'；其余 {omitted} 个用 CLI/JSON 查看' if omitted>0 else '')
        lines.append(f"| {safe(group['operator'])} / {safe(group['reason_code'])} | {len(ids)} | {sample} | {safe(group['reason'])} |")
    if not groups:lines.append('| — | 0 | — | 无阻碍性未知项 |')
    lines+=['','同一节点可在多个原因组出现，不能相加当作待验证节点总数。非阻碍性融合提示不计入关键 UNKNOWN。','',
            '## 4. 重要 Tensor 资源 / Tensor Resource Analysis','',
            '**理论原始载荷，不是实际 BPU/DDR/SRAM 或峰值内存。Hypothetical INT8 raw-payload scenario 仅是假设元素数×1 B。**','',
            '| 边界 Tensor | Shape / dtype | 原始 B | 假设 INT8 B |','|---|---|---|---|']
    for t in model['inputs']+model['outputs']:
        r=records[t['name']]
        lines.append(f"| {safe(t['name'])} | {safe(r['shape'])} / {safe(r['dtype'])} | {safe(r['raw_bytes'])} | {safe(r['hypothetical_int8_bytes'])} |")
    lines+=['','### 前 10 个已知大小的中间 Tensor','',
            '| Tensor | Shape / dtype | 原始 B | MiB | producer | consumers | 假设 INT8 B |','|---|---|---|---|---|---|---|']
    resource_summary=data['resource_analysis']['summary']
    for name in resource_summary['largest_intermediates']:
        r=records[name]
        lines.append('| '+' | '.join(map(safe,[name,str(r['shape'])+' / '+str(r['dtype']),r['raw_bytes'],r['mib'],r['producer_node_id'],r['consumer_count'],r['hypothetical_int8_bytes']]))+' |')
    unknown=[r for r in records.values() if r['raw_bytes'] is None]
    reasons=Counter('符号维度' if any(type(x)is str for x in r.get('shape') or []) else r['size_status'] for r in unknown)
    lines += ['',f"资源未知 {len(unknown)} 个；按原因 {safe(dict(reasons))}；仅已知集合的 Top10，不保证全图最大。",
              f"中间已知载荷之和 {resource_summary['intermediate_activations']['known_bytes_sum']} B（{resource_summary['intermediate_activations']['completeness']}）；多消费者中间 Tensor {resource_summary['multi_consumer_intermediate_count']} 个。",
              '1 MiB=1,048,576 B；1 MB=1,000,000 B。载荷之和不是峰值，fanout 不代表额外分配。','',
              '## 5. Graph Optimization Candidates','']
    candidates=data['optimization_candidates']['candidates'];cs=data['optimization_candidates']['summary']
    lines+=[f"候选/观察 {cs['candidate_count']}；按模式 {safe(cs['counts_by_pattern'])}；按分类 {safe(cs['counts_by_classification'])}。",
            'SEMANTICALLY_REDUNDANT 为局部语义冗余；REVIEW_REQUIRED 需接口/分支/融合审查；INSUFFICIENT_INFORMATION 未证明冗余。没有改写模型。']
    for c in candidates[:representative_limit]:
        lines.append(f"- {c['candidate_id']} / {safe(c['pattern'])} / {safe(c['classification'])}：节点 {safe(c['node_ids'])}，原名 {safe(c['node_names'])}；条件/阻碍 {safe(c['blockers'])}；完整 Tensor 证据、重叠与验证步骤见 candidate/JSON。")
    if len(candidates)>representative_limit:lines.append(f'其余 {len(candidates)-representative_limit} 个候选/观察请用 candidates/candidate 查询。')
    if not candidates:lines.append('未发现已实现五种模式的候选或信息不足观察。')
    lines+=['','## 6. 基于证据的回源建议','']
    actionable=[d for d in data['diagnostics'] if d['status'] in ('VIOLATION','NEEDS_VERIFICATION')]
    if not actionable:lines.append('没有节点级证据要求改变结构；不制造必须优化的建议。')
    else:
        shown=set()
        for diagnostic in actionable:
            for r in diagnostic['results']:
                if r['status'] not in ('FAIL','UNKNOWN') or not r.get('blocking',True):continue
                key=(diagnostic['operator'],r['reason_code'],r['status'])
                if key in shown:continue
                shown.add(key)
                lines.append(f"- {diagnostic['node_id']} / {r['rule_id']} / {r['status']}：{safe(r['reason'])}。"+
                             ('导出层先核对维度/轴/常量形状；若修改模型定义/forward 的计算结构，需要评估重训或微调。' if r['status']=='FAIL' else '先补字段或验证转换条件；缺乏结构修改依据。')+
                             '在训练/导出工程实施，重新导出 → checker → analyze → 接口/数值及任务指标复验；不修补原 ONNX。')
    lines+=['','## 7. 实际使用规则来源、未覆盖与限制','']
    sources={}
    for diagnostic in data['diagnostics']:
        for r in diagnostic['results']:
            s=r['source'];sources[(s['url'],s['version'])]=s
    for s in sources.values():lines.append(f"- [{safe(s['document'])}]({s['url']})；文档版本 {safe(s['version'])}；抓取 {safe(s['retrieved_at'])}；仅 X5 ONNX BPU 栏。")
    uncovered=[row['operator'] for row in summary['operator_coverage'] if row['status_counts']['NOT_COVERED']]
    lines += [f'未覆盖算子/范围：{safe(uncovered)}。']
    lines += ['- '+safe(x) for x in rules['exclusions']+data['limitations']+data['unverified_assumptions']]
    lines+=['','## 8. 查询方法和复现元数据','',
            '- 完整证据：[analysis.json](analysis.json)；机器路径和模型边界保存在 JSON。',
            '- `python -m rdkx5_doctor rules validate`；`rules list --operator Mul`。',
            '- `python -m rdkx5_doctor nodes --analysis analysis.json --status VIOLATION`（或 NEEDS_VERIFICATION）。',
            '- `python -m rdkx5_doctor inspect --analysis analysis.json --node <node_id>`。',
            '- `python -m rdkx5_doctor trace --analysis analysis.json --node <node_id>`。',
            '- `python -m rdkx5_doctor tensors --analysis analysis.json --kind output`；`tensor --analysis analysis.json --name <tensor>`。',
            '- `python -m rdkx5_doctor candidates --analysis analysis.json`；`candidate --analysis analysis.json --id <candidate_id>`。',
            '- 在训练/导出工程生成新模型后：`analyze --model <new_model.onnx> --out reports/new-run`。', '']
    return '\n'.join(lines)
