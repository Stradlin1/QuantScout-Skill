"""Validate and count existing Schema 1.3 evidence; no model analysis."""
from collections import Counter
import json
from pathlib import Path

STATES = ('NO_VIOLATION_FOUND', 'VIOLATION', 'NEEDS_VERIFICATION', 'NOT_COVERED')


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'JSON 重复字段：{key}')
            result[key] = value
        return result
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'非法 JSON 数值：{value}')))
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
