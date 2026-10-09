from pathlib import Path
import json
import pytest
from rdkx5_doctor.toolchain_profile import load_profile, preflight
from rdkx5_doctor.cli import main
PROFILE=Path(__file__).resolve().parents[1]/'.agents/skills/rdk-x5-onnx-doctor/references/toolchain_profile_opset11.yaml'

@pytest.mark.parametrize('version,status',[(11,'MATCH'),(8,'MISMATCH'),(10,'MISMATCH'),(12,'MISMATCH'),(14,'MISMATCH'),(20,'MISMATCH')])
def test_current_profile(version,status):
    result=preflight({'opset_imports':[{'domain':'','version':version}], 'validation':'checker_passed'},load_profile(PROFILE))
    assert result['status']==status and result['actual_standard_opset']==version
    assert result['profile_origin']=='user_provided_project_constraint'
    assert result['compiler_checked'] is False and result['checker_status']=='checker_passed'

@pytest.mark.parametrize('entries',[None,[],[{'domain':'custom','version':11}],
    [{'domain':'','version':True}],[{'domain':'','version':'11'}],[{'domain':'','version':0}],
    [{'domain':'','version':11},{'domain':'ai.onnx','version':11}],
    [{'domain':'','version':11},{'domain':'','version':12}]])
def test_ambiguous_or_missing(entries):
    result=preflight({'opset_imports':entries},load_profile(PROFILE))
    assert result['status']=='UNKNOWN' and result['actual_standard_opset'] is None

def test_custom_domain_is_separate():
    result=preflight({'opset_imports':[{'domain':'ai.onnx','version':11},{'domain':'custom','version':99}]},load_profile(PROFILE))
    assert result['status']=='MATCH' and result['requires_separate_review']
    assert result['custom_domains']==[{'domain':'custom','version':99}]

def test_user_override_not_model_guess():
    profile=load_profile(PROFILE).model_copy(update={'profile_id':'explicit-other','onnx_default_domain_required_opset':12})
    assert preflight({'opset_imports':[{'domain':'','version':12}]},profile)['status']=='MATCH'

@pytest.mark.parametrize('schema',['1.0','1.1','1.2','1.3'])
def test_old_analysis_profile_cli(schema,tmp_path,capsys):
    p=tmp_path/'old.json';p.write_text(json.dumps(dict(schema_version=schema,nodes=[],tensors=[],edges=[],diagnostics=[],traces=[],model={})))
    assert main(['preflight','--analysis',str(p),'--profile',str(PROFILE),'--json'])==0
    assert json.loads(capsys.readouterr().out)['status']=='UNKNOWN'

def test_invalid_profile_rejected(tmp_path):
    p=tmp_path/'bad.yaml';p.write_text(PROFILE.read_text()+'\ncompiler_checked: true\n')
    with pytest.raises(ValueError):load_profile(p)

@pytest.mark.parametrize('model',[None,[], 'missing metadata'])
def test_malformed_legacy_metadata_stays_unknown(model):
    result=preflight(model,load_profile(PROFILE))
    assert result['status']=='UNKNOWN' and result['checker_status']=='unknown'
