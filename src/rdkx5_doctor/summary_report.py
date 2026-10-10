"""Separate, factual Markdown renderer; never reads report.md or suggestions."""
from pathlib import Path
from .summary_facts import extract_facts, profile_evidence, read_json, require
from .text_utils import json_text, printable


def cell(value):
    if value is None:
        return '未提供'
    text = json_text(value, ensure_ascii=False, allow_nan=False) if isinstance(value, (dict, list)) else printable(value)
    # Escape Markdown/HTML, including arbitrary model names and source strings.
    return ''.join(f'&#{ord(c)};' if c in '\\`*_{}[]()#+!|<>&~' or 0xD800 <= ord(c) <= 0xDFFF else c for c in text)


def markdown(facts, profile=None, limit=3, official=None):
    require(type(limit) is int and limit > 0, '代表记录 limit 必须大于零')
    model = facts['model']
    def value(key):
        return cell(model.get(key))
    def count_io(key):
        raw = model.get(key)
        return str(len(raw)) if isinstance(raw, list) else '未提供'
    lines = ['# ONNX 检查事实摘要', '', '## 1. 模型基本信息', '',
             f'- 模型：{cell(Path(model["path"]).name) if isinstance(model.get("path"), str) else "未提供"}',
             f'- 格式检查：{value("validation")}', f'- OpSet：{value("opset_imports")}',
             f'- 节点：{model["node_count"]}；Tensor：{facts["tensor_count"]}；输入：{count_io("inputs")}；输出：{count_io("outputs")}',
             f'- SHA256：{value("sha256")}（仅 ONNX 文件，不含 external data 内容）',
             '', '## 2. 当前 Profile 检查状态', '']
    if profile is None:
        lines.append('未提供 preflight.json；本次事实总结未执行 Profile 检查。')
    else:
        lines += [f'状态：{cell(profile["status"])}；Profile：{cell(profile["profile_id"])}。',
                  f'标准域实际 OpSet：{cell(profile["actual_standard_opset"])}；当前配置要求：{cell(profile["required_standard_opset"])}。',
                  'MATCH 仅表示当前用户工具链配置的版本条件匹配；MISMATCH 表示条件不匹配；UNKNOWN 表示无法确认。实际编译支持未验证。']
    lines += ['', '## 3. 静态规则检查统计', '', '| 最终诊断状态 | 节点数量 |', '| --- | ---: |']
    lines += [f'| {state} | {count} |' for state, count in facts['status_counts'].items()]
    lines += ['', f'合计：{model["node_count"]} 个节点。', '', '| 算子（精确 op_type） | 节点数量 |', '| --- | ---: |']
    lines += [f'| {cell(op)} | {count} |' for op, count in facts['operators'].items()]
    lines += ['', f'规则集：{cell(facts["ruleset"] and {k: facts["ruleset"].get(k) for k in ("id", "version", "toolchain_version")})}',
              '', '## 4. 已确定的静态约束冲突', '',
              f'唯一违规节点：{facts["violation_node_count"]}；FAIL 规则记录：{facts["fail_record_count"]}。']
    def groups(selected):
        rendered = []
        for (status, op, rule, reason), records in facts['groups'].items():
            if status not in selected:
                continue
            unique = len({r['node_id'] for r in records})
            rendered += ['', f'- {cell(status)} / {cell(op)} / 规则 {cell(rule)} / 原因码 {cell(reason)}：{unique} 个唯一节点，{len(records)} 条记录。']
            for r in records[:limit]:
                rendered.append(f'  - 节点 ID：{cell(r["node_id"])}；原名：{cell(r["original_name"])}。')
                if 'field' in r:
                    rendered.append(f'    字段：{cell(r["field"])}；实际值：{cell(r["actual"])}；允许值：{cell(r["expected"])}；来源：{cell(r["source"])}。')
            rendered.append(f'  - 省略 {len(records) - min(limit, len(records))} 条记录；完整证据保留于 analysis.json。')
        return rendered
    lines += groups({'FAIL'}) if facts['fail_record_count'] else ['', '本次已执行的规则未发现确定 FAIL。']
    lines += ['', '## 5. 待验证与未覆盖情况', '',
              'UNKNOWN 是规则记录状态；NOT_COVERED 是节点状态，均不表示 PASS。同一节点可在多个分组中出现，分组数量不能相加作为唯一节点总数。UNKNOWN 记录也可能属于已经 VIOLATION 的节点。']
    uncertain = groups({'UNKNOWN', 'NEEDS_VERIFICATION', 'NOT_COVERED'})
    lines += uncertain or ['', '本次诊断记录无待验证或未覆盖分组。']
    lines += ['', '## 6. Shape 检查摘要', '']
    labels = {'before_known_tensor_count': '推断前载荷已知 Tensor', 'before_unknown_tensor_count': '推断前载荷未知 Tensor',
              'after_known_tensor_count': '推断后载荷已知 Tensor', 'after_unknown_tensor_count': '推断后载荷未知 Tensor',
              'newly_proved_tensor_count': '恢复为载荷已知的 Tensor', 'newly_proved_axis_count': '新证明维度轴', 'conflict_count': '冲突'}
    shape = facts['shape']
    if shape is None:
        lines.append('未提供 Shape 统计。')
    else:
        lines += ['| 已有统计 | 数量 |', '| --- | ---: |']
        lines += [f'| {label} | {cell(shape.get(key))} |' for key, label in labels.items()]
        lines += [f'已有未知项原因码统计：{cell(shape.get("reason_counts"))}。',
                  '载荷已知/未知计数复用原始 Shape summary，受 Shape 和 dtype 共同影响。']
    lines += ['', '## 7. 检查范围说明', '',
              '本摘要仅整理当前 analysis.json 与明确提供且核对适用的 Profile 记录，不读取 report.md 的分析段落。',
              '本次总结未执行官方检索。',
              '未执行实际 Docker 量化、模型编译或板端测试。静态检查不确定实际 CPU/BPU 分配、精度、FPS 或延迟。',
              'NO_VIOLATION_FOUND 仅表示已执行的已收录规则未发现确定冲突，不代表模型完全兼容 RDK X5。', '']
    if official is None:
        lines += ['未提供当前运行的官方查询记录。', '']
    else:
        lines += ['已有官方查询记录（调用者确认属于当前运行）：']
        for r in official['records']:
            lines.append(f'- 查询键：{cell(r["query_key"])}；查询状态：{cell(r["lookup_status"])}；来源：{cell(r["source"])}；交叉来源：{cell(r.get("cross_sources"))}。')
        if not official['records']:
            lines.append('查询记录为空；无法确认已执行官方查询。')
        lines.append('')
    return '\n'.join(lines)


def write_summary(analysis, *, preflight=None, profile=None, official_lookup=None, limit=3):
    analysis = Path(analysis)
    data = read_json(analysis)
    facts = extract_facts(data)
    require((preflight is None) == (profile is None), '--preflight 与 --profile 必须同时提供')
    record = None
    if preflight is not None:
        from .toolchain_profile import load_profile
        require(Path(preflight).resolve().parent == analysis.resolve().parent, 'preflight 必须位于当前报告目录')
        record = profile_evidence(data, read_json(preflight), load_profile(profile))
    official = None
    if official_lookup is not None:
        from .official_knowledge import KnowledgeLookup
        require(Path(official_lookup).resolve().parent == analysis.resolve().parent, '官方查询记录必须位于当前报告目录')
        official = KnowledgeLookup.model_validate(read_json(official_lookup)).model_dump()
        for r in official['records']:
            key = r['query_key']
            matches = [n for n in data['nodes'] if n['op_type'] == key['operator'] and n['domain'] == key['domain']]
            imports = [i['version'] for i in data['model'].get('opset_imports', [])
                       if i['domain'] == key['domain'] or i['domain'] in ('', 'ai.onnx') and key['domain'] in ('', 'ai.onnx')]
            require(matches and key['imported_opset'] == (imports[0] if len(imports) == 1 else None), '官方查询键不适用于当前模型')
    content = markdown(facts, record, limit, official)
    target = analysis.parent / 'summary.md'
    # Exclusive creation protects existing artifacts and symlinks, including models.
    with target.open('x', encoding='utf-8') as stream:
        stream.write(content)
    return target
