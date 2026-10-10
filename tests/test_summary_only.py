import copy
import hashlib
import html
import json
from pathlib import Path

import pytest
from rdkx5_doctor.cli import main, DEFAULT_RULESET
from rdkx5_doctor.onnx_reader import read_model
from rdkx5_doctor.report import analyze
from rdkx5_doctor.rules import load_ruleset
from rdkx5_doctor.summary_facts import extract_facts, read_json, profile_evidence
from rdkx5_doctor.summary_report import markdown, write_summary
from rdkx5_doctor.toolchain_profile import load_profile, preflight

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / '.agents/skills/rdk-x5-onnx-doctor/references/toolchain_profile_opset11.yaml'


@pytest.fixture
def data(make_model):
    rules, manifest = load_ruleset(DEFAULT_RULESET)
    return analyze(read_model(make_model(kernel=(32, 33), branch=True)), rules, manifest)


def test_multiple_fail_unknown_and_no_advice(data):
    facts = extract_facts(data)
    fail = [r for d in data['diagnostics'] for r in d['results'] if r['status'] == 'FAIL']
    assert len(fail) >= 2
    assert facts['violation_node_count'] == 1
    assert facts['fail_record_count'] == len(fail)
    assert sum(facts['status_counts'].values()) == len(data['nodes'])
    assert facts['status_counts']['NOT_COVERED'] > 0
    text = markdown(facts)
    assert '未执行 Profile' in text
    assert all(word not in text for word in ('建议', '优化方案', '原因推测', '优先处理', '下一步'))
    assert data['nodes'][0]['id'] in html.unescape(text)


def test_no_fail_does_not_pass_everything(make_model):
    rules, manifest = load_ruleset(DEFAULT_RULESET)
    d = analyze(read_model(make_model(branch=True)), rules, manifest)
    facts = extract_facts(d)
    assert facts['violation_node_count'] == 0
    assert facts['status_counts']['NOT_COVERED']
    assert '本次已执行的规则未发现确定 FAIL' in markdown(facts)
    assert '不代表模型完全兼容' in markdown(facts)


@pytest.mark.parametrize('opsets,status', [([{'domain': '', 'version': 11}], 'MATCH'), ([{'domain': '', 'version': 14}], 'MISMATCH'), ([], 'UNKNOWN')])
def test_profile_states(data, opsets, status):
    data['model']['opset_imports'] = opsets
    profile = load_profile(PROFILE)
    record = preflight(data['model'], profile)
    assert profile_evidence(data, record, profile)['status'] == status
    assert f'状态：{status}' in markdown(extract_facts(data), record)
    record['status'] = 'UNKNOWN' if status != 'UNKNOWN' else 'MATCH'
    with pytest.raises(ValueError, match='不一致'):
        profile_evidence(data, record, profile)


@pytest.mark.parametrize('mutation', [
    lambda d: d.update(schema_version='1.2'),
    lambda d: d['summary'].pop('all_status_counts'),
    lambda d: d['summary']['all_status_counts'].update(VIOLATION=99),
    lambda d: d['model'].update(node_count=99),
    lambda d: d['diagnostics'].append(d['diagnostics'][0]),
    lambda d: d['nodes'].append(d['nodes'][0]),
    lambda d: d['diagnostics'][0]['results'].append(d['diagnostics'][0]['results'][0]),
    lambda d: d['summary']['operator_coverage'][0].update(node_count=99),
    lambda d: d['shape_analysis']['summary'].update(conflict_count=99),
    lambda d: d['diagnostics'][0].update(status='NO_VIOLATION_FOUND'),
    lambda d: d['diagnostics'][0].update(operator='Shape'),
])
def test_reject_inconsistent(data, mutation):
    mutation(data)
    with pytest.raises(ValueError):
        extract_facts(data)


def test_large_group_limit_and_dedup(data):
    seed = copy.deepcopy(data['diagnostics'][0])
    node = copy.deepcopy(data['nodes'][0])
    count = 100
    data['nodes'], data['diagnostics'] = [], []
    for i in range(count):
        n, d = copy.deepcopy(node), copy.deepcopy(seed)
        n['id'] = d['node_id'] = f'main/node_{i:06}'
        for r in d['results']:
            r['node_id'] = n['id']
        data['nodes'].append(n)
        data['diagnostics'].append(d)
    data['model']['node_count'] = count
    data['model']['operator_counts'] = {'Conv': count}
    data['summary']['all_status_counts'] = {'VIOLATION': count}
    data['summary']['operator_coverage'] = [dict(operator='Conv', node_count=count, status_counts={'VIOLATION': count})]
    facts = extract_facts(data)
    assert facts['violation_node_count'] == count
    text = markdown(facts, limit=2)
    assert '省略 98 条记录' in text
    assert 'main/node_000099' not in html.unescape(text)
    assert facts['fail_record_count'] == count * sum(r['status'] == 'FAIL' for r in seed['results'])


def test_cli_model_then_summary_preserves_artifacts(make_model, tmp_path):
    model = make_model(kernel=(32, 33))
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    out = tmp_path / 'run'
    assert main(['analyze', '--model', str(model), '--out', str(out)]) == 0
    before = {p: p.read_bytes() for p in out.iterdir()}
    assert main(['summary', '--analysis', str(out / 'analysis.json')]) == 0
    assert (out / 'summary.md').is_file()
    assert all(p.read_bytes() == raw for p, raw in before.items())
    assert hashlib.sha256(model.read_bytes()).hexdigest() == digest
    original = (out / 'summary.md').read_bytes()
    assert main(['summary', '--analysis', str(out / 'analysis.json')]) == 2
    assert (out / 'summary.md').read_bytes() == original


@pytest.mark.parametrize('raw', ['{broken', '{"a":1,"a":2}', '{"a":NaN}', '[]'])
def test_bad_json_refuses_output(tmp_path, raw):
    path = tmp_path / 'analysis.json'
    path.write_text(raw)
    assert main(['summary', '--analysis', str(path)]) == 2
    assert not (tmp_path / 'summary.md').exists()


def test_missing_json(tmp_path):
    assert main(['summary', '--analysis', str(tmp_path / 'absent.json')]) == 2
    assert not (tmp_path / 'summary.md').exists()


def test_preflight_file_validation(data, tmp_path):
    path = tmp_path / 'analysis.json'
    path.write_text(json.dumps(data))
    record = tmp_path / 'preflight.json'
    record.write_text(json.dumps(preflight(data['model'], load_profile(PROFILE))))
    before = {p: p.read_bytes() for p in (path, record)}
    write_summary(path, preflight=record, profile=PROFILE)
    assert '状态：MATCH' in (tmp_path / 'summary.md').read_text()
    assert all(p.read_bytes() == raw for p, raw in before.items())


def test_optional_missing_and_escaping(data):
    data.pop('shape_analysis')
    data['model'].pop('sha256')
    data['nodes'][0]['original_name'] = '<script>\x1b[31m|\n#Injected'
    for r in data['diagnostics'][0]['results']:
        r.pop('reason_code', None)
    text = markdown(extract_facts(data))
    assert '未提供 Shape 统计' in text
    assert 'SHA256：未提供' in text
    assert '<script>' not in text and '\x1b' not in text and '\n#Injected' not in text


def test_official_only_status_sources(data, tmp_path):
    from test_skill_contract import record
    path = tmp_path / 'analysis.json'
    path.write_text(json.dumps(data))
    lookup = tmp_path / 'official_lookup.json'
    r = record()
    r['query_key'] = dict(operator='Relu', domain='', imported_opset=11)
    r['extracted_conditions'] = ['建议修改结构']
    lookup.write_text(json.dumps(dict(schema_version='1.0', records=[r])))
    write_summary(path, official_lookup=lookup)
    text = (tmp_path / 'summary.md').read_text()
    assert 'FETCHED' in text and '建议' not in text
    assert 'supported_op_list' in html.unescape(text)


def test_many_unknown_records_do_not_duplicate_final_node(data):
    diagnostic = next(d for d in data['diagnostics'] if any(r['status'] == 'UNKNOWN' for r in d['results']))
    unknown = [r for r in diagnostic['results'] if r['status'] == 'UNKNOWN']
    assert unknown
    extra = copy.deepcopy(unknown[0])
    extra['rule_id'] += '_additional_existing_record'
    diagnostic['results'].append(extra)
    facts = extract_facts(data)
    assert facts['violation_node_count'] == 1
    assert sum(facts['status_counts'].values()) == len(data['nodes'])
    assert sum(len(v) for k, v in facts['groups'].items() if k[0] == 'UNKNOWN') == sum(r['status'] == 'UNKNOWN' for d in data['diagnostics'] for r in d['results'])


@pytest.mark.parametrize('limit', [0, -1])
def test_bad_limit_does_not_create(data, tmp_path, limit):
    path = tmp_path / 'analysis.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        write_summary(path, limit=limit)
    assert not (tmp_path / 'summary.md').exists()


def test_symlink_output_cannot_touch_model(data, tmp_path):
    path = tmp_path / 'analysis.json'
    path.write_text(json.dumps(data))
    source = tmp_path / 'original.onnx'
    source.write_bytes(b'original model')
    (tmp_path / 'summary.md').symlink_to(source)
    with pytest.raises(FileExistsError):
        write_summary(path)
    assert source.read_bytes() == b'original model'


def test_cross_run_sidecar_rejected(data, tmp_path):
    path = tmp_path / 'analysis.json'
    path.write_text(json.dumps(data))
    other = tmp_path / 'old'
    other.mkdir()
    sidecar = other / 'preflight.json'
    sidecar.write_text(json.dumps(preflight(data['model'], load_profile(PROFILE))))
    with pytest.raises(ValueError, match='当前报告目录'):
        write_summary(path, preflight=sidecar, profile=PROFILE)
    assert not (tmp_path / 'summary.md').exists()


def test_summary_skill_has_independent_end_condition():
    skill = ROOT / '.agents/skills/rdk-x5-onnx-doctor'
    text = (skill / 'SKILL.md').read_text()
    contract = (skill / 'references/summary_only_contract.md').read_text()
    assert 'summary_only' in text and 'summary_only_contract.md' in text
    assert '聊天回复只提供已验证事实和文件路径' in contract
    assert '不继承 answer_contract.md' in contract


def test_unknown_without_fail_keeps_final_state(make_model):
    rules, manifest = load_ruleset(DEFAULT_RULESET)
    d = analyze(read_model(make_model(unknown_weight=True)), rules, manifest)
    facts = extract_facts(d)
    assert facts['fail_record_count'] == 0
    assert facts['status_counts']['NEEDS_VERIFICATION'] == 1
    assert 'UNKNOWN' in markdown(facts)
    assert facts['status_counts']['NO_VIOLATION_FOUND'] == 0
