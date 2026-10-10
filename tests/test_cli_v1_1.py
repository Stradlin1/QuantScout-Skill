import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from rdkx5_doctor.cli import main, load_analysis
from rdkx5_doctor.onnx_reader import read_model
from rdkx5_doctor.report import analyze
from rdkx5_doctor.rules import load_ruleset
from rdkx5_doctor.cli import DEFAULT_RULESET

@pytest.fixture
def report(make_model,tmp_path,capsys):
    model=make_model(kernel=(32,1),branch=True)
    out=tmp_path/'reports'
    before=model.read_bytes()
    assert main(['analyze','--model',str(model),'--out',str(out)])==0
    assert before==model.read_bytes()
    capsys.readouterr()
    return out/'analysis.json',model,out


def test_extended_schema_unchanged_v1_diagnostics(report):
    path,model,out=report
    data=load_analysis(path)
    baseline=json.loads((Path(__file__).parents[1]/'examples/legacy-demo-report/analysis.json').read_text())
    assert data['schema_version']=='1.3'
    assert set(baseline)<=set(data)
    for result,old in zip(data['diagnostics'][0]['results'],baseline['diagnostics'][0]['results']):
        assert {key:result[key] for key in old}==old
    assert not any('graph.html'==p.name for p in out.iterdir())
    md=(out/'report.md').read_text()
    assert 'Tensor Resource Analysis' in md and 'Graph Optimization Candidates' in md
    for c in data['optimization_candidates']['candidates']:
        assert c['candidate_id'] in md
    assert 'INSUFFICIENT_INFORMATION' in md
    assert data['ruleset']['version']=='0.4.0'
    assert load_ruleset(DEFAULT_RULESET)[0].by_operator['Conv'].ruleset_version==baseline['ruleset']['version']


def test_all_queries_without_model_loading(report,capsys):
    path,model,_=report
    args=['--analysis',str(path)]
    model.unlink()
    with patch('rdkx5_doctor.cli.read_model',side_effect=AssertionError('Must not load ONNX for saved queries')):
        assert main(['tensors',*args,'--kind','output','--json'])==0
        outputs=json.loads(capsys.readouterr().out)
        assert outputs[0]['name']=='joined' and outputs[0]['raw_bytes']==5760
        assert main(['tensor',*args,'--name','c','--json'])==0
        assert json.loads(capsys.readouterr().out)['consumer_count']==2
        assert main(['candidates',*args,'--pattern','IDENTITY','--json'])==0
        cs=json.loads(capsys.readouterr().out)
        assert len(cs)==1 and cs[0]['pattern']=='IDENTITY'
        assert main(['candidate',*args,'--id',cs[0]['candidate_id'],'--json'])==0
        c=json.loads(capsys.readouterr().out)
        assert c['reachable_outputs']==['sum','joined'] and c['trace']['node_id']==cs[0]['node_ids'][-1]
        assert main(['tensor',*args,'--name','missing'])==2
        assert 'Tensor' in capsys.readouterr().err
        assert main(['candidate',*args,'--id','missing'])==2
        assert '候选' in capsys.readouterr().err


def test_sorting_unknowns_limit_and_text(report,capsys):
    path,_,_=report
    data=json.loads(path.read_text())
    data['resource_analysis']['tensor_records'][0].update(raw_bytes=None,mib=None,size_status='UNKNOWN_SHAPE',unknown_reason='symbolic')
    path.write_text(json.dumps(data))
    args=['--analysis',str(path)]
    assert main(['tensors',*args,'--sort','bytes','--limit','0','--json'])==0
    rows=json.loads(capsys.readouterr().out)
    assert rows[-1]['raw_bytes'] is None
    assert main(['tensors',*args,'--sort','name','--limit','2','--json'])==0
    rows=json.loads(capsys.readouterr().out)
    assert len(rows)==2 and [r['name'] for r in rows]==sorted(r['name'] for r in data['resource_analysis']['tensor_records'])[:2]
    assert main(['tensors',*args,'--limit','1'])==0
    output=capsys.readouterr().out
    assert 'Unknown-size Tensors: 1' in output and 'not actual' in output
    assert main(['tensors',*args,'--unknown','--json'])==0
    assert len(json.loads(capsys.readouterr().out))==1
    with pytest.raises(SystemExit) as e:
        main(['tensors',*args,'--limit','-1'])
    assert e.value.code==2


@pytest.mark.parametrize('command,extra',[('nodes',[]),('inspect',['--node','main/node_000000']),('trace',['--node','main/node_000000'])])
def test_existing_queries_accept_old_v1(report,command,extra,capsys):
    path,_,_=report
    data=json.loads(path.read_text())
    data['schema_version']='1.0'
    del data['resource_analysis'];del data['optimization_candidates']
    path.write_text(json.dumps(data))
    assert main([command,'--analysis',str(path),*extra,'--json'])==0
    assert json.loads(capsys.readouterr().out)


@pytest.mark.parametrize('command,extra',[('tensors',[]),('tensor',['--name','c']),('candidates',[]),('candidate',['--id','OPT-0001'])])
def test_new_queries_reject_old_v1(report,command,extra,capsys):
    path,_,_=report
    data=json.loads(path.read_text());data['schema_version']='1.0'
    del data['resource_analysis'];del data['optimization_candidates']
    path.write_text(json.dumps(data))
    assert main([command,'--analysis',str(path),*extra])==2
    assert 'rerun analyze' in capsys.readouterr().err


def test_reproducible_facts_and_no_candidates(make_model):
    model=make_model()
    rules,manifest=load_ruleset(DEFAULT_RULESET)
    a=analyze(read_model(model),rules,manifest)
    b=analyze(read_model(model),rules,manifest)
    assert a==b
    assert a['optimization_candidates']['candidates']==[]
    json.dumps(a,allow_nan=False)


def test_terminal_escape_resource_and_candidate_views(report,capsys):
    path,_,_=report
    data=json.loads(path.read_text())
    name='\x1b[31mname\x1b]0;title\x07\x9b31m'
    data['resource_analysis']['tensor_records'][0]['name']=name
    data['optimization_candidates']['candidates'][0]['node_names']=[name]
    path.write_text(json.dumps(data))
    for command in ('tensors','candidates'):
        assert main([command,'--analysis',str(path),'--json'])==0
        parsed=json.loads(capsys.readouterr().out)
        assert (any(r['name']==name for r in parsed) if command=='tensors' else parsed[0]['node_names']==[name])
        assert main([command,'--analysis',str(path)])==0
        output=capsys.readouterr().out
        assert '\x1b' not in output and '\x07' not in output and '\x9b' not in output


def test_malformed_new_sections(report,capsys):
    path,_,_=report
    data=json.loads(path.read_text())
    data['resource_analysis']={'schema_version':'1.0','tensor_records':'bad','summary':{}}
    path.write_text(json.dumps(data))
    assert main(['tensors','--analysis',str(path)])==2
    assert '结构无效' in capsys.readouterr().err
