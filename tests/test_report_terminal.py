import hashlib
import json
import subprocess
import sys
from rdkx5_doctor.cli import main,DEFAULT_RULESET

def test_reports_terminal_and_source_immutable(make_model,tmp_path,capsys):
    model=make_model(kernel=(32,1),branch=True,name='name\x1b[31m')
    before=hashlib.sha256(model.read_bytes()).hexdigest()
    out=tmp_path/'report'
    assert main(['analyze','--model',str(model),'--out',str(out)])==0
    assert before==hashlib.sha256(model.read_bytes()).hexdigest()
    data=json.loads((out/'analysis.json').read_text())
    assert data['model']['sha256']==before
    assert data['diagnostics'][0]['status']=='VIOLATION'
    assert all(d['status']=='NOT_COVERED' for d in data['diagnostics'][1:3])
    assert all(d['status']=='NO_VIOLATION_FOUND' for d in data['diagnostics'][3:])
    assert len(data['traces'][0]['reachable_outputs'])==2
    report=(out/'report.md').read_text()
    for section in range(1,9):
        assert f'## {section}.' in report
    assert '未经过 OpenExplorer/hb_mapper 实测' in report
    assert set(p.name for p in out.iterdir())=={'analysis.json','report.md'}
    stdout=capsys.readouterr().out
    assert 'X5-CONV2D-KERNEL-H' in stdout and 'sum' in stdout and 'joined' in stdout
    assert '\x1b' not in stdout

def test_cli_errors(tmp_path,capsys):
    assert main(['rules','validate','--ruleset',str(DEFAULT_RULESET)])==0
    assert main(['analyze','--model',str(tmp_path/'missing.onnx'),'--out',str(tmp_path/'out')])==2
    p=subprocess.run([sys.executable,'-m','rdkx5_doctor','--help'],capture_output=True,text=True)
    assert p.returncode==0 and all(x in p.stdout for x in ('analyze','nodes','inspect','trace'))
    assert main(['nodes','--analysis',str(tmp_path/'missing.json')])==2

def test_output_symlink_cannot_modify_model(make_model,tmp_path):
    model=make_model()
    out=tmp_path/'aliased';out.mkdir()
    (out/'report.md').symlink_to(model)
    before=model.read_bytes()
    assert main(['analyze','--model',str(model),'--out',str(out)])==2
    assert model.read_bytes()==before

def test_search_filter_inspect_trace(make_model,tmp_path,capsys):
    model=make_model(kernel=(32,1),branch=True)
    out=tmp_path/'report'
    assert main(['analyze','--model',str(model),'--out',str(out)])==0
    capsys.readouterr()
    args=['--analysis',str(out/'analysis.json')]
    assert main(['nodes',*args,'--status','VIOLATION','--json'])==0
    rows=json.loads(capsys.readouterr().out)
    assert len(rows)==1 and rows[0]['id']=='main/node_000000'
    assert main(['nodes',*args,'--search','joined','--json'])==0
    assert json.loads(capsys.readouterr().out)[0]['operator']=='Concat'
    assert main(['inspect',*args,'--node','conv','--json'])==0
    facts=json.loads(capsys.readouterr().out)
    assert facts['node']['conv']['fields']['kernel_h']==32
    assert facts['diagnostic']['results'][0]['status']=='FAIL'
    assert main(['trace',*args,'--node','main/node_000000','--json'])==0
    assert json.loads(capsys.readouterr().out)['reachable_outputs']==['sum','joined']
    # Arbitrary NOT_COVERED nodes can be traced from report topology too.
    assert main(['trace',*args,'--node','main/node_000001','--json'])==0
    assert json.loads(capsys.readouterr().out)['reachable_outputs']==['sum','joined']
    assert main(['inspect',*args,'--node','main/node_000003'])==0
    assert 'X5-ADD-' in capsys.readouterr().out
    assert main(['nodes',*args,'--search','does-not-exist'])==0
    assert '匹配 0' in capsys.readouterr().out
    assert main(['inspect',*args,'--node','same'])==2
    assert '原始名称重复' in capsys.readouterr().err
    assert main(['trace',*args,'--node','missing'])==2
    assert '未找到节点' in capsys.readouterr().err

def test_malformed_analysis(tmp_path,capsys):
    path=tmp_path/'bad.json'
    for contents in ('broken','{}','[]','{"schema_version":"1.0"}'):
        path.write_text(contents)
        assert main(['nodes','--analysis',str(path)])==2
        assert '无法读取分析报告' in capsys.readouterr().err
