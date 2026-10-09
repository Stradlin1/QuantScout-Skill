"""Machine facts and Chinese human report from the same evidence."""
from collections import Counter
from .rules import check_operator
from .trace import trace_node
from .tensor_resource import analyze_tensor_resources
from .optimization_candidates import analyze_optimization_candidates

LIMITATIONS = ['尚未用实际工具链验证：未经过 OpenExplorer/hb_mapper 实测。',
 '只检查实际注册的标准 ONNX 算子和已审查版本；其它算子不代表兼容。',
 '静态检查不能确定真实 CPU/BPU 分配、量化误差、延迟、峰值 DDR 或任务精度。',
 '路径只表示数据依赖，不表示下游节点违规。',
 '模型 SHA256 仅覆盖 .onnx 文件；external data 内容不包含在该摘要中。']

def analyze(ir, ruleset, manifest):
    diagnostics = []
    node_index = {n['id']:n for n in ir.nodes}
    for node in ir.nodes:
        diagnostics.append(check_operator(ir,node,ruleset,node_index))
    traces = [trace_node(ir, d['node_id']) for d in diagnostics if d['status'] == 'VIOLATION']
    by_id = {n['id']: n for n in ir.nodes}
    by_diagnostic = {d['node_id']: d for d in diagnostics}
    for trace in traces:
        trace['original_name'] = by_id[trace['node_id']]['original_name']
        trace['evidence'] = [r for r in by_diagnostic[trace['node_id']]['results'] if r['status'] == 'FAIL']
    conv_ids = {n['id'] for n in ir.nodes if n['op_type'] == 'Conv'}
    coverage = []
    for operator in dict.fromkeys(n['op_type'] for n in ir.nodes):
        members = [d for n,d in zip(ir.nodes,diagnostics) if n['op_type']==operator]
        op = ruleset.by_operator.get(operator)
        coverage.append({'operator':operator,'node_count':len(members),
             'status_counts':{s:sum(d['status']==s for d in members) for s in ('NO_VIOLATION_FOUND','VIOLATION','NEEDS_VERIFICATION','NOT_COVERED')},
             'rule_version':op.ruleset_version if op else None,'source_version':op.source_version if op else None})
    return {'schema_version': '1.2', 'model': ir.model,
        'ruleset': {'id': ruleset.ruleset_id, 'version': ruleset.ruleset_version,
            'toolchain_version': ruleset.toolchain_version, 'toolchain_verified': False,
            'sources': [s.model_dump() for s in manifest.sources],
            'rules': [r.model_dump(exclude_none=True) for r in ruleset.all_rules],
            'operator_rule_versions': {name:op.ruleset_version for name,op in ruleset.by_operator.items()},
            'operator_opsets': {name:{'min':op.opset_min,'max':op.opset_max} for name,op in ruleset.by_operator.items()},
            'exclusions': manifest.exclusions, 'review_status': manifest.review_status},
        'nodes': ir.nodes, 'tensors': list(ir.tensors.values()), 'edges': ir.edges,
        'diagnostics': diagnostics, 'traces': traces,
        'summary': {'all_status_counts': dict(Counter(d['status'] for d in diagnostics)),
            'conv_status_counts': dict(Counter(d['status'] for d in diagnostics if d['node_id'] in conv_ids)),
            'conv_node_count': len(conv_ids), 'conv2d_checked_count': sum(d.get('scope') == 'conv2d' for d in diagnostics),
            'operator_coverage': coverage,
            'covered_node_count':sum(d['status']!='NOT_COVERED' for d in diagnostics),
            'uncovered_node_count':sum(d['status']=='NOT_COVERED' for d in diagnostics)},
        'unverified_assumptions': ir.warnings, 'limitations': LIMITATIONS,
        'resource_analysis': analyze_tensor_resources(ir),
        'optimization_candidates': analyze_optimization_candidates(ir)}


# Keep the public report.markdown entry point while separating rendering policy.
from .reporting_sections import markdown
