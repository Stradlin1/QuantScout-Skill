import json
from pathlib import Path
import pytest
from rdkx5_doctor.report import analyze
from rdkx5_doctor.rules import load_ruleset, multiop_status, Rule
from rdkx5_doctor.onnx_reader import read_model
from rdkx5_doctor.cli import DEFAULT_RULESET, main

@pytest.mark.parametrize('schema',['1.0','1.1'])
def test_real_historical_queries(schema,capsys):
    root=Path(__file__).parents[1]
    path=root/'examples'/('demo-report' if schema=='1.0' else 'v1_1_demo-report')/'analysis.json'
    assert json.loads(path.read_text())['schema_version']==schema
    for command,extra in [('nodes',[]),('inspect',['--node','main/node_000000']),('trace',['--node','main/node_000000'])]:
        assert main([command,'--analysis',str(path),*extra,'--json'])==0
        assert json.loads(capsys.readouterr().out)
    if schema=='1.1':
        for command in ('tensors','candidates'):
            assert main([command,'--analysis',str(path),'--json'])==0
            assert json.loads(capsys.readouterr().out)
    else:
        assert main(['tensors','--analysis',str(path),'--json'])==2

def test_coverage_and_per_rule_evidence(make_model):
    registry,manifest=load_ruleset(DEFAULT_RULESET)
    data=analyze(read_model(make_model(branch=True)),registry,manifest)
    assert data['schema_version']=='1.3'
    summary=data['summary']
    assert summary['covered_node_count']+summary['uncovered_node_count']==data['model']['node_count']
    assert sum(summary['all_status_counts'].values())==data['model']['node_count']
    assert sum(row['node_count'] for row in summary['operator_coverage'])==data['model']['node_count']
    for row in summary['operator_coverage']:assert sum(row['status_counts'].values())==row['node_count']
    assert len(data['diagnostics'])==len({d['node_id'] for d in data['diagnostics']})==len(data['nodes'])
    for diagnostic in data['diagnostics']:
        for result in diagnostic['results']:
            assert result['node_id']==diagnostic['node_id']
            assert result['observed']['value']==result['actual']
            assert result['evidence'] and result['reason_code']
            assert all(result['source'][k] for k in ('document','version','section','url','column','retrieved_at'))

def test_status_precedence():
    r=Rule(id='r',field='input_rank',check_type='range',min=1,max=10,checkability='auto_check',source_url='https://example.org',source_section='s',message='m')
    def result(status):return {'rule_id':'r','status':status,'checkability':'auto_check'}
    assert multiop_status([result('FAIL'),result('UNKNOWN')],[r])=='VIOLATION'
    assert multiop_status([result('UNKNOWN')],[r])=='NEEDS_VERIFICATION'
    assert multiop_status([result('PASS')],[r])=='NO_VIOLATION_FOUND'
    assert multiop_status([result('NOT_APPLICABLE')],[r])=='NOT_COVERED'
    assert multiop_status([],[])=='NOT_COVERED'
