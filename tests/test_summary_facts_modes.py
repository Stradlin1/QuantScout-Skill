"""Offline facts/mode tests. These are not natural-language Agent E2E."""
import copy
import json
from pathlib import Path

import onnx
import pytest
from onnx import helper as h, TensorProto as T
from rdkx5_doctor.cli import main, DEFAULT_RULESET
from rdkx5_doctor.onnx_reader import read_model
from rdkx5_doctor.report import analyze
from rdkx5_doctor.rules import load_ruleset
from rdkx5_doctor.summary_facts import build_fact_package, export_fact_package


@pytest.fixture
def analysis_path(make_model, tmp_path):
    rules, manifest = load_ruleset(DEFAULT_RULESET)
    data = analyze(read_model(make_model(kernel=(32, 33), branch=True)), rules, manifest)
    path = tmp_path / 'analysis.json'
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False))
    return path


def alter(path, fn):
    data = json.loads(path.read_text())
    fn(data)
    path.write_text(json.dumps(data))


@pytest.mark.parametrize('mode', ['overview', 'anomalies', 'io'])
def test_modes_stable_and_json_only(analysis_path, mode, capsys):
    first = export_fact_package(analysis_path, mode=mode)
    assert first == export_fact_package(analysis_path, mode=mode)
    pack = json.loads(first)
    assert pack['scope']['counted_graph'] == 'main_graph_only'
    assert pack['scope']['nested_subgraphs_expanded'] is False
    assert pack['preflight']['status'] is None
    assert len({c['id'] for c in pack['allowed_claims']}) == len(pack['allowed_claims'])
    if mode != 'io':
        diag = pack['diagnostics']
        assert sum(diag['status_counts'].values()) == pack['model']['main_graph_node_count']
        assert diag['violation_node_count'] == 1
        assert diag['fail_rule_record_count'] >= 2
        if mode == 'overview':
            assert len(diag['representative_failures']) == 1
            assert 'operator_counts' not in first and 'shape_analysis' not in first
        else:
            assert diag['status_counts']['NOT_COVERED'] > 0
            assert diag['unknown_rule_record_count'] > 0
    else:
        assert 'diagnostics' not in pack
        assert pack['io']['inputs']['records'][0]['shape'] == [1, 2, 40, 40]
    assert main(['summary-facts', '--analysis', str(analysis_path), '--mode', mode, '--json']) == 0
    assert json.loads(capsys.readouterr().out) == pack


@pytest.mark.parametrize('limit', [0, -1, 11])
def test_bad_limits_no_output(analysis_path, limit):
    target = analysis_path.parent / 'facts.json'
    assert main(['summary-facts', '--analysis', str(analysis_path), '--limit', str(limit), '--out', str(target)]) == 2
    assert not target.exists()


@pytest.mark.parametrize('schema', ['1.0', '1.1', '1.2', '9.0'])
def test_old_schema(analysis_path, schema):
    alter(analysis_path, lambda d: d.update(schema_version=schema))
    with pytest.raises(ValueError, match='Schema 1.3'):
        build_fact_package(analysis_path)


@pytest.mark.parametrize('raw', ['{"a":1,"a":2}', '{"x":NaN}', '{"x":Infinity}', '[1]', '{broken'])
def test_bad_json_no_output(analysis_path, raw):
    analysis_path.write_text(raw)
    out = analysis_path.parent / 'facts.json'
    assert main(['summary-facts', '--analysis', str(analysis_path), '--out', str(out)]) == 2
    assert not out.exists()


@pytest.mark.parametrize('mutate', [
    lambda d: d['model'].update(node_count=True),
    lambda d: d['model'].update(inputs='unknown'),
    lambda d: d['model'].update(opset_imports=[{'domain':'','version':True}]),
    lambda d: d['model']['inputs'][0].update(shape=[True]),
    lambda d: d['summary']['operator_coverage'][0]['coverage_counts'].update(AUTO_CHECKED=99),
    lambda d: d['diagnostics'].append(d['diagnostics'][0]),
])
def test_bad_metadata_and_counts(analysis_path, mutate):
    alter(analysis_path, mutate)
    with pytest.raises(ValueError):
        build_fact_package(analysis_path)


def test_io_order_symbolic_and_omissions(analysis_path):
    def modify(d):
        d['model']['inputs'][0]['shape'] = ['batch', None, 3]
        d['model']['inputs'].append(copy.deepcopy(d['model']['outputs'][0]))
    alter(analysis_path, modify)
    pack = build_fact_package(analysis_path, mode='io', limit=1)
    assert pack['io']['inputs']['records'][0]['shape'] == ['batch', None, 3]
    assert pack['io']['inputs']['total_count'] == 2
    assert pack['io']['inputs']['omitted_count'] == 1
    assert pack['io']['outputs']['omitted_count'] == 1


def test_unknown_in_violation_not_new_node(analysis_path):
    def modify(d):
        extra = copy.deepcopy(next(r for diag in d['diagnostics'] for r in diag['results'] if r['status'] == 'UNKNOWN'))
        extra.update(node_id=d['diagnostics'][0]['node_id'], operator='Conv')
        d['diagnostics'][0]['results'].append(extra)
    alter(analysis_path, modify)
    diag = build_fact_package(analysis_path, mode='anomalies')['diagnostics']
    assert diag['violation_node_count'] == 1
    assert diag['unknown_rule_record_count'] >= 2


def test_string_bounds(analysis_path):
    alter(analysis_path, lambda d: d['model'].update(path='x'*2000 + '.onnx'))
    pack = build_fact_package(analysis_path)
    assert pack['model']['name']['truncated'] is True
    assert len(pack['model']['name']['display_fragment']) == 256
    assert '截断' in next(c['canonical_text'] for c in pack['allowed_claims'] if c['id'] == 'MODEL.NAME')


def test_facts_output_protection_and_no_onnx(analysis_path, monkeypatch):
    import rdkx5_doctor.cli as cli
    def forbidden(*args, **kwargs):
        pytest.fail('facts must not analyze or open ONNX')
    monkeypatch.setattr(cli, 'analyze', forbidden)
    monkeypatch.setattr(cli, 'read_model', forbidden)
    out = analysis_path.parent / 'facts.json'
    assert main(['summary-facts', '--analysis', str(analysis_path), '--out', str(out)]) == 0
    before = out.read_bytes()
    assert main(['summary-facts', '--analysis', str(analysis_path), '--out', str(out)]) == 2
    assert out.read_bytes() == before
    with pytest.raises(ValueError):
        export_fact_package(analysis_path, out=analysis_path.parent / 'wrong.onnx')


def make_if_model(tmp_path):
    branches = []
    for name in ('yes', 'no'):
        value = h.make_tensor(name, T.FLOAT, [1], [1.0])
        branches.append(h.make_graph([h.make_node('Constant', [], [name], value=value)], name, [], [h.make_tensor_value_info(name, T.FLOAT, [1])]))
    graph = h.make_graph([h.make_node('If', ['cond'], ['y'], then_branch=branches[0], else_branch=branches[1])], 'subgraph',
                         [h.make_tensor_value_info('cond', T.BOOL, [])], [h.make_tensor_value_info('y', T.FLOAT, [1])])
    model = h.make_model(graph, opset_imports=[h.make_opsetid('', 11)])
    onnx.checker.check_model(model)
    path = tmp_path / 'if.onnx'
    onnx.save(model, path)
    return path


def test_subgraph_main_only(tmp_path):
    path = make_if_model(tmp_path)
    rules, manifest = load_ruleset(DEFAULT_RULESET)
    data = analyze(read_model(path), rules, manifest)
    target = tmp_path / 'analysis.json'
    target.write_text(json.dumps(data))
    pack = build_fact_package(target)
    assert data['model']['node_count'] == 1
    assert pack['scope']['nested_subgraphs_present'] is True
    assert sum(pack['diagnostics']['status_counts'].values()) == 1
    assert '未逐节点展开' in pack['allowed_claims'][0]['canonical_text']


def test_large_failure_counts_remain_complete(analysis_path):
    data = json.loads(analysis_path.read_text())
    node, diag = data['nodes'][0], data['diagnostics'][0]
    count = 300
    data['nodes'], data['diagnostics'] = [], []
    for i in range(count):
        n, d = copy.deepcopy(node), copy.deepcopy(diag)
        n['id'] = d['node_id'] = f'main/node_{i:06}'
        for r in d['results']:
            r['node_id'] = n['id']
        data['nodes'].append(n); data['diagnostics'].append(d)
    data['model']['node_count'] = count
    data['model']['operator_counts'] = {'Conv': count}
    data['summary']['all_status_counts'] = {'VIOLATION': count}
    data['summary']['operator_coverage'] = [dict(operator='Conv',node_count=count,status_counts={'VIOLATION':count})]
    analysis_path.write_text(json.dumps(data))
    pack = build_fact_package(analysis_path, limit=2)
    diag = pack['diagnostics']
    assert diag['violation_node_count'] == count
    assert len(diag['representative_failures']) == 2
    assert diag['omitted_failure_record_count'] == diag['fail_rule_record_count']-2
    assert len(json.dumps(pack)) < 15000


def test_missing_optional_io_stays_unknown(analysis_path):
    alter(analysis_path, lambda d: d['model'].pop('inputs'))
    pack = build_fact_package(analysis_path, mode='io')
    assert pack['io']['inputs']['total_count'] is None
    assert pack['io']['inputs']['omitted_count'] is None
    assert '未提供' in next(c['canonical_text'] for c in pack['allowed_claims'] if c['id']=='IO.INPUTS_COUNT')


def test_no_fail_not_covered_is_not_compatibility(make_model, tmp_path):
    rules, manifest = load_ruleset(DEFAULT_RULESET)
    data = analyze(read_model(make_model(branch=True)), rules, manifest)
    path = tmp_path/'clean.json';path.write_text(json.dumps(data))
    pack = build_fact_package(path)
    assert pack['diagnostics']['fail_rule_record_count'] == 0
    assert pack['diagnostics']['status_counts']['NOT_COVERED'] > 0
    assert any(c['id']=='DIAG.NO_FAIL' for c in pack['allowed_claims'])


def test_invalid_mode_and_out_parent(analysis_path, tmp_path):
    with pytest.raises(SystemExit):
        main(['summary-facts','--analysis',str(analysis_path),'--mode','unknown'])
    with pytest.raises(ValueError):
        export_fact_package(analysis_path,out=tmp_path.parent/'outside.json')
    assert not (tmp_path.parent/'outside.json').exists()


def test_preflight_binding_and_mismatch(analysis_path):
    from rdkx5_doctor.toolchain_profile import load_profile, preflight
    profile = Path(__file__).resolve().parents[1]/'.agents/skills/rdk-x5-onnx-doctor/references/toolchain_profile_opset11.yaml'
    sidecar = analysis_path.parent/'preflight.json'
    record = preflight(json.loads(analysis_path.read_text())['model'],load_profile(profile))
    sidecar.write_text(json.dumps(record))
    pack = build_fact_package(analysis_path,preflight=sidecar,profile=profile)
    assert pack['preflight']['status']=='MATCH'
    assert pack['profile_binding']['preflight_sha256']
    record['status']='UNKNOWN';sidecar.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='不一致'):
        build_fact_package(analysis_path,preflight=sidecar,profile=profile)
