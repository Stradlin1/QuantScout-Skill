"""Validate and count existing Schema 1.3 evidence; no model analysis."""
from collections import Counter
import json
from pathlib import Path

STATES = ('NO_VIOLATION_FOUND', 'VIOLATION', 'NEEDS_VERIFICATION', 'NOT_COVERED')


def parse_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'JSON 重复字段：{key}')
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'非法 JSON 数值：{value}')))


def read_json(path):
    try:
        return parse_json(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError(f'无法读取 JSON {path}：{exc}') from exc


def require(condition, message):
    if not condition:
        raise ValueError(f'事实摘要数据错误：{message}')


def counts_match(raw, actual, keys):
    # Schema 1.3 Counter omits zero categories; absent categories are only zero
    # after recounting the complete diagnostic list.
    return (isinstance(raw, dict) and set(raw) <= set(keys)
            and all(type(v) is int and v >= 0 for v in raw.values())
            and all(raw.get(k, 0) == actual.get(k, 0) for k in keys))


def extract_facts(data):
    require(isinstance(data, dict) and data.get('schema_version') == '1.3',
            '仅验证过 Schema 1.3；旧版或未知 Schema 无法确认兼容性')
    try:
        nodes, diagnostics, tensors = data['nodes'], data['diagnostics'], data['tensors']
        require(all(isinstance(x, list) for x in (nodes, diagnostics, tensors)), '节点/诊断/Tensor 列表缺失')
        by_id = {n['id']: n for n in nodes}
        require(all(isinstance(n['id'], str) and n['id'] and isinstance(n['op_type'], str) and n['op_type'] for n in nodes), '节点标识或 op_type 无效')
        require(len(by_id) == len(nodes), '节点 ID 重复')
        require(all(isinstance(t['name'], str) and t['name'] for t in tensors), 'Tensor 名称无效')
        require(len({t['name'] for t in tensors}) == len(tensors), 'Tensor 名称重复')
        require(len(diagnostics) == len(nodes) and {d['node_id'] for d in diagnostics} == set(by_id), '节点与诊断不是一一对应')
        require(type(data['model']['node_count']) is int and data['model']['node_count'] == len(nodes), '模型节点总数不一致')
        statuses, fail_ids, groups = Counter(), set(), {}
        for d in diagnostics:
            node = by_id[d['node_id']]
            require(d['status'] in STATES and d['operator'] == node['op_type'], '诊断状态或算子不一致')
            require(isinstance(d['results'], list), '规则记录缺失')
            statuses[d['status']] += 1
            seen_rules = set()
            for r in d['results']:
                require(isinstance(r['rule_id'], str) and bool(r['rule_id']), '规则 ID 无效')
                require(r.get('reason_code') is None or isinstance(r['reason_code'], str), '原因码无效')
                require(r['status'] in ('PASS', 'FAIL', 'UNKNOWN', 'NOT_APPLICABLE'), '规则状态无效')
                require(r['rule_id'] not in seen_rules, '同节点规则记录重复')
                seen_rules.add(r['rule_id'])
                require(r['node_id'] == node['id'] and r['operator'] == node['op_type'], '规则节点或算子不一致')
                if r['status'] not in ('FAIL', 'UNKNOWN'):
                    continue
                if r['status'] == 'FAIL':
                    fail_ids.add(node['id'])
                key = (r['status'], node['op_type'], r['rule_id'], r.get('reason_code'))
                groups.setdefault(key, []).append(dict(node_id=node['id'], original_name=node.get('original_name'),
                    field=r.get('field'), actual=r.get('actual'), expected=r.get('expected'), source=r.get('source')))
            require((node['id'] in fail_ids) == (d['status'] == 'VIOLATION'), 'FAIL 与最终 VIOLATION 不一致')
            # Preserve final state for nodes with no UNKNOWN rule (e.g. shape issues).
            if d['status'] in ('NEEDS_VERIFICATION', 'NOT_COVERED') and not any(r['status'] == 'UNKNOWN' for r in d['results']):
                key = (d['status'], node['op_type'], None, None)
                groups.setdefault(key, []).append(dict(node_id=node['id'], original_name=node.get('original_name')))
        summary = data['summary']
        require(counts_match(summary['all_status_counts'], statuses, STATES), '四种状态计数不守恒或与诊断不一致')
        operators = Counter(n['op_type'] for n in nodes)
        if 'operator_counts' in data['model']:
            qualified = Counter((n['domain'] + '::' if n['domain'] else '') + n['op_type'] for n in nodes)
            require(counts_match(data['model']['operator_counts'], qualified, qualified), '模型算子计数不一致')
        coverage = summary['operator_coverage']
        require(isinstance(coverage, list) and len(coverage) == len(operators) and {r['operator'] for r in coverage} == set(operators), '算子覆盖列表不一致')
        for row in coverage:
            op = row['operator']
            require(type(row['node_count']) is int and row['node_count'] == operators[op], '算子节点数不一致')
            require(counts_match(row['status_counts'], Counter(d['status'] for d in diagnostics if d['operator'] == op), STATES), '算子状态统计不一致')
        shape = data.get('shape_analysis', {}).get('summary')
        require(data.get('ruleset') is None or isinstance(data['ruleset'], dict), '规则集元信息无效')
        if shape is not None:
            require(isinstance(shape, dict), 'Shape summary 无效')
            for key, value in shape.items():
                if key.endswith('_count'):
                    require(type(value) is int and value >= 0, f'Shape {key} 无效')
            for prefix in ('before', 'after'):
                k, u = f'{prefix}_known_tensor_count', f'{prefix}_unknown_tensor_count'
                if k in shape and u in shape:
                    require(shape[k] + shape[u] == len(tensors), 'Shape Tensor 计数不守恒')
            if 'conflict_count' in shape and 'conflicts' in data['shape_analysis']:
                require(shape['conflict_count'] == len(data['shape_analysis']['conflicts']), 'Shape 冲突数不一致')
        return dict(model=data['model'], tensor_count=len(tensors), status_counts={s: statuses[s] for s in STATES},
                    operators=dict(sorted(operators.items())), violation_node_count=len(fail_ids),
                    fail_record_count=sum(len(v) for k, v in groups.items() if k[0] == 'FAIL'),
                    groups=groups, shape=shape, ruleset=data.get('ruleset'))
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f'事实摘要字段缺失或损坏：{exc}') from exc


def profile_evidence(data, record, profile):
    from .toolchain_profile import preflight
    expected = preflight(data['model'], profile)
    require(record == expected, 'preflight 与当前 analysis/Profile 不一致，无法确认适用性')
    return record


MODES = ('overview', 'anomalies', 'io')
MAX_LIMIT = 10
MAX_VALUE_CHARS = 256


def bounded(value):
    """Bound display data, retaining explicit truncation metadata, never totals."""
    from .text_utils import json_text, printable
    if isinstance(value, str):
        value = printable(value)
        if len(value) <= MAX_VALUE_CHARS:
            return value
        return {'display_fragment': value[:MAX_VALUE_CHARS], 'truncated': True,
                'original_character_count': len(value)}
    if isinstance(value, (list, dict)):
        text = json_text(value, ensure_ascii=False, allow_nan=False)
        if len(text) > MAX_VALUE_CHARS:
            return {'display_fragment': text[:MAX_VALUE_CHARS], 'truncated': True,
                    'original_character_count': len(text)}
    return value


def fact_text(value):
    from .text_utils import json_text
    if value is None:
        return '未提供'
    if isinstance(value, dict) and value.get('truncated') is True:
        return f'{value["display_fragment"]}…（截断，原显示文本共 {value["original_character_count"]} 字符）'
    if isinstance(value, str):
        return value
    return json_text(value, ensure_ascii=False, allow_nan=False)


def graph_scope(data):
    """Inspect existing attribute markers only; never open or traverse ONNX."""
    markers = [n.get('attributes') for n in data['nodes']]
    present = any(isinstance(attrs, dict) and any(isinstance(v, dict) and v.get('subgraph') == 'not_covered'
                  for v in attrs.values()) for attrs in markers)
    known = all(isinstance(attrs, dict) for attrs in markers)
    return dict(counted_graph='main_graph_only', nested_subgraphs_expanded=False,
                nested_subgraphs_present=True if present else False if known else None)


def _validate_metadata(data):
    """Validate metadata types for summary use; do not infer or repair fields."""
    model = data['model']
    for key in ('path', 'validation', 'sha256'):
        require(model.get(key) is None or isinstance(model[key], str), f'model.{key} 类型无效')
    imports = model.get('opset_imports')
    require(imports is None or isinstance(imports, list), 'OpSet 元信息无效')
    for entry in imports or []:
        require(isinstance(entry, dict) and isinstance(entry.get('domain'), str)
                and type(entry.get('version')) is int and entry['version'] > 0, 'OpSet 条目无效')
    names = {t['name'] for t in data['tensors']}
    for side in ('inputs', 'outputs'):
        records = model.get(side)
        require(records is None or isinstance(records, list), f'graph {side} 类型无效')
        seen = set()
        for r in records or []:
            require(isinstance(r, dict) and isinstance(r.get('name'), str) and r['name'] in names
                    and r['name'] not in seen, f'graph {side} 名称缺失、重复或不属于 Tensor 列表')
            seen.add(r['name'])
            require(r.get('dtype') is None or isinstance(r['dtype'], str), 'I/O dtype 无效')
            shape = r.get('shape')
            require(shape is None or isinstance(shape, list), 'I/O Shape 无效')
            for dim in shape or []:
                require(dim is None or isinstance(dim, str) or type(dim) is int and dim >= 0, 'I/O 维度类型无效')
    for row in data['summary']['operator_coverage']:
        if 'coverage_counts' in row:
            selected = [d for d in data['diagnostics'] if d['operator'] == row['operator']]
            categories = ('AUTO_CHECKED', 'PARTIAL_OR_CONDITIONAL', 'NOT_COVERED')
            require(all(d.get('coverage_type') in categories for d in selected), 'coverage_type 无效')
            require(counts_match(row['coverage_counts'], Counter(d['coverage_type'] for d in selected), categories), '算子覆盖计数不一致')


def build_fact_package(analysis, *, mode='overview', limit=2, preflight=None, profile=None):
    """Create bounded facts with stable IDs, paths and a raw analysis SHA binding."""
    import hashlib
    require(mode in MODES, '不支持的摘要 mode')
    require(type(limit) is int and 1 <= limit <= MAX_LIMIT, f'limit 必须为 1～{MAX_LIMIT}')
    path = Path(analysis)
    # Parse the exact same byte snapshot that is hashed.
    raw = path.read_bytes()
    data = parse_json(raw.decode('utf-8'))
    facts = extract_facts(data)
    _validate_metadata(data)
    record = load_summary_profile(data, path, preflight, profile)
    claims = []
    def claim(id_, kind, source, value, text):
        claims.append(dict(id=id_, kind=kind, source_path=source, value=value, canonical_text=text))
    scope = graph_scope(data)
    scope_text = '仅统计已解析主图节点；嵌套子图未逐节点展开、不计入诊断统计'
    if scope['nested_subgraphs_present'] is True:
        scope_text += '；已有元信息记录了嵌套子图'
    elif scope['nested_subgraphs_present'] is None:
        scope_text += '；子图是否存在无法确认'
    claim('LIMIT.SCOPE', 'scope', 'analysis.json:nodes/*/attributes', scope, scope_text)
    if mode != 'io':
        claim('LIMIT.EXECUTION', 'boundary', 'tool_execution_boundary',
              dict(docker_quantization_executed=False, compiler_executed=False, runtime_placement_verified=False),
              '本次静态检查未执行实际量化、编译或板端测试；未确认 CPU/BPU 分配；静态状态不等于完整兼容')
    model = data['model']
    name = bounded(Path(model['path']).name) if model.get('path') else None
    info = dict(name=name, checker_status=bounded(model.get('validation')), opset_imports=bounded(model.get('opset_imports')),
                main_graph_node_count=model['node_count'], tensor_count=facts['tensor_count'],
                graph_input_count=len(model['inputs']) if isinstance(model.get('inputs'), list) else None,
                graph_output_count=len(model['outputs']) if isinstance(model.get('outputs'), list) else None)
    claim('MODEL.NAME', 'metadata', 'analysis.json:model/path', name, f'模型名称：{fact_text(name)}')
    if mode == 'overview':
        claim('MODEL.OPSET', 'metadata', 'analysis.json:model/opset_imports', info['opset_imports'], f'OpSet 导入：{fact_text(info["opset_imports"])}')
        claim('MODEL.MAIN_GRAPH_NODES', 'count', 'analysis.json:model/node_count', model['node_count'], f'已解析主图节点 {model["node_count"]} 个')
        claim('DIAG.STATUS_COUNTS', 'counts', 'analysis.json:summary/all_status_counts', facts['status_counts'],
              '、'.join(f'{k} {v}' for k, v in facts['status_counts'].items()) + f'；四项合计 {model["node_count"]} 个主图节点')
    diagnostic_info = None
    if mode in ('overview', 'anomalies'):
        claim('DIAG.VIOLATION_NODES', 'count', 'analysis.json:summary/all_status_counts/VIOLATION', facts['violation_node_count'], f'唯一违规主图节点 {facts["violation_node_count"]} 个')
        claim('DIAG.FAIL_RECORDS', 'count', 'analysis.json:diagnostics/*/results/status=FAIL', facts['fail_record_count'], f'FAIL 规则记录共 {facts["fail_record_count"]} 条')
        if facts['fail_record_count'] == 0:
            claim('DIAG.NO_FAIL', 'boundary', 'analysis.json:diagnostics/*/results', 0, '当前已执行规则未发现确定 FAIL')
        failures, unknowns = [], []
        failure_nodes = set()
        unknown_count = sum(len(v) for k, v in facts['groups'].items() if k[0] == 'UNKNOWN')
        for di, d in enumerate(data['diagnostics']):
            for ri, r in enumerate(d['results']):
                category = r['status']
                # Overview uses distinct representative nodes, anomalies retains rule records.
                if category == 'FAIL' and len(failures) < min(limit, 2 if mode == 'overview' else MAX_LIMIT) and (mode != 'overview' or d['node_id'] not in failure_nodes):
                    sample = {k: bounded(r.get(k)) for k in ('node_id', 'operator', 'rule_id', 'field', 'actual', 'expected', 'reason_code', 'source')}
                    sample['original_name'] = bounded(data['nodes'][di].get('original_name')) if data['nodes'][di]['id'] == d['node_id'] else bounded(next(n for n in data['nodes'] if n['id'] == d['node_id']).get('original_name'))
                    failures.append(sample)
                    failure_nodes.add(d['node_id'])
                    claim(f'FAIL.{len(failures)-1}', 'rule_record', f'analysis.json:diagnostics/{di}/results/{ri}', sample,
                          f'代表 FAIL：节点 {fact_text(sample["node_id"])}（原名 {fact_text(sample["original_name"])}），算子 {fact_text(sample["operator"])}，规则 {fact_text(sample["rule_id"])}，字段 {fact_text(sample["field"])}，实际值 {fact_text(sample["actual"])}，允许值 {fact_text(sample["expected"])}')
                if mode == 'anomalies' and category == 'UNKNOWN' and len(unknowns) < limit:
                    sample = {k: bounded(r.get(k)) for k in ('node_id', 'operator', 'rule_id', 'reason_code', 'actual', 'expected')}
                    unknowns.append(sample)
                    claim(f'UNKNOWN.{len(unknowns)-1}', 'rule_record', f'analysis.json:diagnostics/{di}/results/{ri}', sample,
                          f'UNKNOWN 记录：节点 {fact_text(sample["node_id"])}，规则 {fact_text(sample["rule_id"])}，原因码 {fact_text(sample["reason_code"])}；此记录不等于额外违规节点')
        omitted = facts['fail_record_count'] - len(failures)
        claim('DIAG.OMITTED_FAILURES', 'count', 'analysis.json:diagnostics/*/results/status=FAIL', omitted, f'另外 {omitted} 条 FAIL 记录未展示')
        diagnostic_info = dict(status_counts=facts['status_counts'], violation_node_count=facts['violation_node_count'],
            fail_rule_record_count=facts['fail_record_count'], representative_failures=failures, omitted_failure_record_count=omitted)
        if mode == 'anomalies':
            diagnostic_info.update(unknown_rule_record_count=unknown_count, representative_unknowns=unknowns, omitted_unknown_record_count=unknown_count-len(unknowns))
            claim('DIAG.UNKNOWN_RECORDS', 'count', 'analysis.json:diagnostics/*/results/status=UNKNOWN', unknown_count,
                  f'UNKNOWN 规则记录共 {unknown_count} 条；未知记录不作为已确定 FAIL')
            claim('DIAG.OMITTED_UNKNOWNS', 'count', 'analysis.json:diagnostics/*/results/status=UNKNOWN', unknown_count-len(unknowns), f'另有 {unknown_count-len(unknowns)} 条 UNKNOWN 记录未展示')
            for state in ('NEEDS_VERIFICATION', 'NOT_COVERED'):
                count = facts['status_counts'][state]
                claim(f'DIAG.{state}', 'count', f'analysis.json:summary/all_status_counts/{state}', count, f'{state} 状态主图节点 {count} 个；该状态不表示 PASS 或已确定 FAIL')
                selected = [(i, d) for i, d in enumerate(data['diagnostics']) if d['status'] == state][:limit]
                samples = []
                for j, (i, d) in enumerate(selected):
                    sample = dict(node_id=bounded(d['node_id']), operator=bounded(d['operator']))
                    samples.append(sample)
                    claim(f'{state}.{j}', 'node', f'analysis.json:diagnostics/{i}', sample, f'{state} 代表节点 {fact_text(sample["node_id"])}，算子 {fact_text(sample["operator"])}')
                diagnostic_info[state.lower()] = dict(total=count, samples=samples, omitted_count=count-len(samples))
    io = None
    if mode == 'io':
        io = {}
        for side in ('inputs', 'outputs'):
            values = model.get(side)
            total = len(values) if isinstance(values, list) else None
            selected = values[:limit] if isinstance(values, list) else []
            samples = []
            id_side = side.upper()
            claim(f'IO.{id_side}_COUNT', 'count', f'analysis.json:model/{side}', total, f'ONNX graph {side} 总数：{fact_text(total)}')
            for index, r in enumerate(selected):
                sample = {k: bounded(r.get(k)) for k in ('name', 'shape', 'dtype')}
                samples.append(sample)
                claim(f'IO.{id_side}.{index}', 'tensor_metadata', f'analysis.json:model/{side}/{index}', sample,
                      f'graph {side}：名称 {fact_text(sample["name"])}，Shape {fact_text(sample["shape"])}，dtype {fact_text(sample["dtype"])}')
            omitted = total-len(selected) if total is not None else None
            io[side] = dict(total_count=total, displayed_count=len(selected), records=samples, omitted_count=omitted)
            claim(f'IO.{id_side}_DISPLAY', 'count', f'analysis.json:model/{side}', dict(total=total, shown=len(selected), omitted=omitted),
                  f'graph {side} 共 {fact_text(total)} 项，展示 {len(selected)} 项，省略 {fact_text(omitted)} 项')
    preflight_info = dict(record_present=record is not None, status=record['status'] if record else None, toolchain_compiler_checked=False)
    if mode != 'io':
        ptext = f'当前 Profile：{record["status"]}；仅比较当前用户配置的版本条件，实际编译支持未验证' if record else '未提供 Profile 验证结果；本次总结未执行 Profile 检查'
        claim('PREFLIGHT.STATUS', 'profile', 'preflight.json' if record else 'not_provided', preflight_info, ptext)
    result = dict(summary_facts_schema_version='1.0', source_analysis_schema='1.3', analysis_sha256=hashlib.sha256(raw).hexdigest(),
                  mode=mode, sample_limit=limit, string_limit=MAX_VALUE_CHARS, scope=scope, model=info,
                  preflight=preflight_info, allowed_claims=claims)
    if record:
        result['profile_binding'] = dict(preflight_sha256=hashlib.sha256(Path(preflight).read_bytes()).hexdigest(),
                                       profile_sha256=hashlib.sha256(Path(profile).read_bytes()).hexdigest())
    if diagnostic_info is not None:
        result['diagnostics'] = diagnostic_info
    if io is not None:
        result['io'] = io
    return result


def load_summary_profile(data, analysis, preflight=None, profile=None):
    require((preflight is None) == (profile is None), '--preflight 与 --profile 必须同时提供')
    if preflight is None:
        return None
    from .toolchain_profile import load_profile
    require(Path(preflight).resolve().parent == Path(analysis).resolve().parent, 'preflight 必须位于当前报告目录')
    return profile_evidence(data, read_json(preflight), load_profile(profile))


def write_new(path, content):
    """Exclusive creation; all validation and encoding precede creation."""
    import os
    path = Path(path)
    encoded = content.encode('utf-8')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o644)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(encoded)
    except BaseException:
        path.unlink(missing_ok=True)  # Only the file just exclusively created here.
        raise
    return path


def export_fact_package(analysis, *, out=None, **kwargs):
    from .text_utils import json_text
    package = build_fact_package(analysis, **kwargs)
    text = json_text(package, ensure_ascii=False, indent=2, allow_nan=False)
    if out is not None:
        target = Path(out)
        require(target.suffix == '.json' and target.resolve().parent == Path(analysis).resolve().parent,
                '事实包输出必须为当前报告目录内的新 .json 文件')
        write_new(target, text + '\n')
    return text
