"""Publish constrained Agent composition, not unconstrained Chinese prose.

Fact placeholders are replaced with validated canonical facts. The Agent chooses
facts, order, headings and safe connective phrasing. Unsupported prose requires
review and is refused; this is not a general semantic hallucination detector.
"""
import hashlib
import json
from pathlib import Path
import re
from .summary_facts import build_fact_package, read_json, require, write_new
from .summary_report import cell

TOKEN = re.compile(r'\{\{([A-Z][A-Z0-9_.]*)\}\}')
HEADINGS = {
    'overview': {'模型概况', '静态检查', '已确定冲突', '检查范围', '模型与静态检查'},
    'anomalies': {'已确定冲突', '待验证记录', '未覆盖范围', '检查范围', '异常记录'},
    'io': {'输入输出概况', '输入', '输出', '检查范围', '输入输出记录'},
}
CONNECTIVES = ('本次记录显示：', '本次记录：', '记录如下：', '其中，', '同时，', '另外，')


def required_claims(pack):
    mode = pack['mode']
    required = {'LIMIT.SCOPE'}
    if mode == 'io':
        required |= {'IO.INPUTS_COUNT', 'IO.OUTPUTS_COUNT', 'IO.INPUTS_DISPLAY', 'IO.OUTPUTS_DISPLAY'}
        required |= {c['id'] for c in pack['allowed_claims'] if c['id'].startswith(('IO.INPUTS.', 'IO.OUTPUTS.'))}
    else:
        required |= {'DIAG.VIOLATION_NODES', 'DIAG.FAIL_RECORDS', 'LIMIT.EXECUTION', 'PREFLIGHT.STATUS'}
        if mode == 'overview':
            required |= {'MODEL.NAME', 'MODEL.OPSET', 'MODEL.MAIN_GRAPH_NODES', 'DIAG.STATUS_COUNTS'}
        else:
            required |= {'DIAG.NEEDS_VERIFICATION', 'DIAG.NOT_COVERED', 'DIAG.UNKNOWN_RECORDS', 'DIAG.OMITTED_UNKNOWNS'}
        if pack['diagnostics']['fail_rule_record_count']:
            required.add('FAIL.0')
        else:
            required.add('DIAG.NO_FAIL')
        required.add('DIAG.OMITTED_FAILURES')
    return required


def render_draft(pack, draft):
    require(isinstance(draft, dict) and set(draft) == {'draft_schema_version', 'mode', 'source_analysis_sha256', 'sections'}, '草稿结构无效')
    require(draft['draft_schema_version'] == '1.0' and draft['mode'] == pack['mode'], '草稿 Schema 或 mode 不一致')
    require(draft['source_analysis_sha256'] == pack['analysis_sha256'], '草稿来源 SHA 不一致')
    sections = draft['sections']
    mode = pack['mode']
    require(isinstance(sections, list) and (3 if mode == 'overview' else 1) <= len(sections) <= (3 if mode == 'io' else 4), '草稿小节数量超出模式限制')
    claims = {c['id']: c for c in pack['allowed_claims']}
    seen, headings = set(), set()
    rendered = ['# ONNX 检查事实摘要', '']
    for section in sections:
        require(isinstance(section, dict) and set(section) == {'heading', 'fact_ids', 'text'}, '草稿小节字段无效')
        heading, ids, text = section['heading'], section['fact_ids'], section['text']
        require(isinstance(heading, str) and heading in HEADINGS[mode] and heading not in headings, '小节标题无效或重复')
        headings.add(heading)
        require(isinstance(ids, list) and ids and all(isinstance(i, str) and i in claims for i in ids), '草稿引用不存在的 fact ID')
        require(len(ids) == len(set(ids)) and not seen.intersection(ids), '草稿事实引用重复')
        require(isinstance(text, str) and 0 < len(text) <= 4096, '草稿文本长度无效')
        references = TOKEN.findall(text)
        require(len(references) == len(ids) and set(references) == set(ids), '文本事实引用与 fact_ids 不一致；禁止手写或篡改数值/状态/节点 ID')
        residual = TOKEN.sub('', text)
        for phrase in CONNECTIVES:
            residual = residual.replace(phrase, '')
        require(not residual.strip(' \n，。、；：'), 'review_required：自由扩写无法机器核验，禁止分析、建议或无依据结论；仅允许事实占位符与契约连接语')
        seen.update(ids)
        # Escape the substituted evidence, rather than trusting names as Markdown.
        safe = TOKEN.sub(lambda match: cell(claims[match[1]]['canonical_text']), text)
        rendered += [f'## {heading}', '', safe, '']
    require(required_claims(pack) <= seen, '草稿缺少该模式必需事实或省略数量')
    return '\n'.join(rendered)


def publish_summary(analysis, facts, draft, out, *, preflight=None, profile=None):
    analysis, facts, draft, out = map(Path, (analysis, facts, draft, out))
    parent = analysis.resolve().parent
    require(out.suffix == '.md' and out.resolve().parent == parent, '输出必须为当前报告目录内的新 Markdown 文件')
    if out.exists() or out.is_symlink():
        raise FileExistsError('目标文件已存在，未覆盖')
    require(facts.resolve().parent == parent and draft.resolve().parent == parent, '事实包及草稿必须位于当前报告目录')
    supplied = read_json(facts)
    require(isinstance(supplied, dict) and supplied.get('summary_facts_schema_version') == '1.0', '事实包 Schema 无效')
    expected = build_fact_package(analysis, mode=supplied.get('mode'), limit=supplied.get('sample_limit'), preflight=preflight, profile=profile)
    normalize = lambda value: json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(',', ':'))
    require(normalize(supplied) == normalize(expected), '事实包与当前 analysis 重算结果不一致（SHA、数值、类型或证据被篡改）')
    require(draft.stat().st_size <= 65536, '草稿超过大小限制')
    content = render_draft(expected, read_json(draft))
    require(hashlib.sha256(analysis.read_bytes()).hexdigest() == expected['analysis_sha256'], '发布过程中 analysis 已变化')
    return write_new(out, content)
