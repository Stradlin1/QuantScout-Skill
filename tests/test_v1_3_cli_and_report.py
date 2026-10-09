import json
from pathlib import Path
import pytest
from rdkx5_doctor.cli import main,DEFAULT_RULESET
from rdkx5_doctor.report import analyze,markdown
from rdkx5_doctor.onnx_reader import read_model
from rdkx5_doctor.rules import load_ruleset
from test_static_shape_propagation import chain

@pytest.mark.parametrize('schema',['1.0','1.1','1.2'])
def test_legacy_shape_error_and_old_queries(schema,tmp_path,capsys):
    p=Path(__file__).parents[1]/'examples/demo-report/analysis.json'
    data=json.loads(p.read_text());data['schema_version']=schema
    q=tmp_path/'legacy.json';q.write_text(json.dumps(data))
    assert main(['nodes','--analysis',str(q),'--json'])==0;capsys.readouterr()
    assert main(['shapes','--analysis',str(q),'--summary'])==2
    assert '重新 analyze' in capsys.readouterr().err

def test_cli_proof_summary_and_escaping(tmp_path,capsys,monkeypatch):
    ir=chain(tmp_path);rules,manifest=load_ruleset(DEFAULT_RULESET);data=analyze(ir,rules,manifest)
    p=tmp_path/'analysis.json';p.write_text(json.dumps(data))
    # Queries must never reopen ONNX.
    monkeypatch.setattr('rdkx5_doctor.cli.read_model',lambda _:pytest.fail('query reopened model'))
    assert main(['shapes','--analysis',str(p),'--summary','--json'])==0
    assert json.loads(capsys.readouterr().out)['after_unknown_tensor_count']==0
    assert main(['shape','--analysis',str(p),'--tensor','y','--json'])==0
    result=json.loads(capsys.readouterr().out);assert result['fact']['final_shape']==[1,32,8,8] and result['proofs']
    assert main(['shapes','--analysis',str(p),'--limit','-1'])==2;capsys.readouterr()
    md=markdown(data);assert 'Shape 推断与未知来源' in md and '无需修改' not in md
    assert data['schema_version']=='1.3'
    assert len(data['shape_analysis']['facts'])==len(data['tensors'])
    assert '专项' in md
    for row in data['summary']['operator_coverage']:assert sum(row['coverage_counts'].values())==row['node_count']

def test_original_known_resource_and_ids_preserved(tmp_path):
    from copy import deepcopy
    from rdkx5_doctor.tensor_resource import analyze_tensor_resources
    ir=chain(tmp_path);before=analyze_tensor_resources(ir);ids=deepcopy(ir.nodes);edges=deepcopy(ir.edges)
    rules,manifest=load_ruleset(DEFAULT_RULESET);data=analyze(ir,rules,manifest)
    a={r['name']:r['raw_bytes'] for r in data['resource_analysis']['tensor_records']}
    for r in before['tensor_records']:
        if r['raw_bytes'] is not None:assert r['raw_bytes']==a[r['name']]
    assert [(n['id'],n['inputs'],n['outputs']) for n in data['nodes']]==[(n['id'],n['inputs'],n['outputs']) for n in ids]
    assert data['edges']==edges and not data['shape_analysis']['conflicts']
