import copy
import shutil
import yaml
import pytest
from rdkx5_doctor.rules import load_ruleset, Rule
from rdkx5_doctor.cli import DEFAULT_RULESET, main

URL = 'https://developer.d-robotics.cc/x5_sdk_doc/toolchain_development/intermediate/supported_op_list.html'

@pytest.fixture
def registered(tmp_path):
    conv = yaml.safe_load((DEFAULT_RULESET/'conv2d.yaml').read_text())
    manifest = yaml.safe_load((DEFAULT_RULESET/'manifest.yaml').read_text())
    manifest['ruleset_id'] = 'test-registry'; manifest['ruleset_version'] = '0.2.0'
    op = dict(ruleset_id='test-sigmoid', ruleset_version='0.2.0', model_domain='', operator='Sigmoid',
              scope='standard_onnx', source_document='X5 芯片用户手册', source_version='1.1.2',
              source_url=URL, toolchain_version='unverified', opset_min=6, opset_max=23,
              rules=[dict(id='X5-SIGMOID-INPUT-RANK', operator='Sigmoid', check_type='range', field='input_rank',
                          min=1, max=10, checkability='auto_check', source_url=URL,
                          source_document='X5 芯片用户手册', source_version='1.1.2',
                          source_column='X5 BPU 支持约束', source_section='X5 ONNX / Sigmoid / BPU',
                          reason_code='BPU_RANK_OUT_OF_RANGE', message='rank outside source interval')])
    manifest['operators'] = {}
    for data, file in [(conv,'conv2d.yaml'),(op,'sigmoid.yaml')]:
        manifest['operators'][data['operator']] = dict(file=file, ruleset_id=data['ruleset_id'],
             version=data['ruleset_version'], source_version=data['source_version'],
             opset_min=data.get('opset_min'), opset_max=data.get('opset_max'))
        (tmp_path/file).write_text(yaml.safe_dump(data,allow_unicode=True))
    (tmp_path/'manifest.yaml').write_text(yaml.safe_dump(manifest,sort_keys=False,allow_unicode=True))
    return tmp_path

def test_registration_and_legacy_api(registered):
    registry, _ = load_ruleset(registered)
    assert list(registry.by_operator)==['Conv','Sigmoid']
    assert len(registry.rules)==19 and len(registry.all_rules)==20
    assert len(registry.rules_by_id)==20
    assert registry.get_rules('Sigmoid','',11).operator=='Sigmoid'
    for domain,opset in [('custom',11),('',5),('',24),('',None)]:
        assert registry.get_rules('Sigmoid',domain,opset) is None
    assert registry.get_rules('MatMul','',11) is None

@pytest.mark.parametrize('case', ['missing','duplicate_id','duplicate_file','version','opset','field','predicate','extra','absolute','traversal','domain','source','unknown_op','unsafe_yaml'])
def test_reject_invalid_registry(registered, case):
    manifest=yaml.safe_load((registered/'manifest.yaml').read_text())
    op=yaml.safe_load((registered/'sigmoid.yaml').read_text())
    if case=='missing': (registered/'sigmoid.yaml').unlink()
    elif case=='duplicate_id': op['rules'][0]['id']='X5-CONV2D-KERNEL-H'
    elif case=='duplicate_file': manifest['operators']['Sigmoid']['file']='conv2d.yaml'
    elif case=='version': manifest['operators']['Sigmoid']['version']='wrong'
    elif case=='opset': manifest['operators']['Sigmoid']['opset_min']=13
    elif case=='field': op['rules'][0]['field']='kernel_h'
    elif case=='predicate': op['rules'][0].update(check_type='predicate',predicate='eval');op['rules'][0].pop('min');op['rules'][0].pop('max')
    elif case=='extra': op['expression']='lambda x: True'
    elif case=='absolute': manifest['operators']['Sigmoid']['file']='/tmp/sigmoid.yaml'
    elif case=='traversal': manifest['operators']['Sigmoid']['file']='../sigmoid.yaml'
    elif case=='domain': op['model_domain']='custom'
    elif case=='source': op['rules'][0]['source_column']='CPU 支持约束'
    elif case=='unknown_op': manifest['operators']['Relu']=manifest['operators'].pop('Sigmoid')
    elif case=='unsafe_yaml': (registered/'sigmoid.yaml').write_text('!!python/object/apply:os.system [echo forbidden]')
    if case not in ('missing','unsafe_yaml'):
        (registered/'sigmoid.yaml').write_text(yaml.safe_dump(op,allow_unicode=True))
    (registered/'manifest.yaml').write_text(yaml.safe_dump(manifest,sort_keys=False,allow_unicode=True))
    with pytest.raises(ValueError,match='规则集校验失败'):load_ruleset(registered)

def test_reject_escaped_symlink(registered,tmp_path):
    outside=tmp_path.parent/'outside.yaml';outside.write_text((registered/'sigmoid.yaml').read_text())
    (registered/'sigmoid.yaml').unlink();(registered/'sigmoid.yaml').symlink_to(outside)
    with pytest.raises(ValueError,match='软链越界'):load_ruleset(registered)

def test_legacy_manifest_without_registry(tmp_path):
    manifest=yaml.safe_load((DEFAULT_RULESET/'manifest.yaml').read_text())
    conv=yaml.safe_load((DEFAULT_RULESET/'conv2d.yaml').read_text())
    manifest.pop('operators',None)
    manifest['ruleset_id']=conv['ruleset_id'];manifest['ruleset_version']=conv['ruleset_version']
    for file,data in [('manifest.yaml',manifest),('conv2d.yaml',conv)]:
        (tmp_path/file).write_text(yaml.safe_dump(data,allow_unicode=True))
    registry,_=load_ruleset(tmp_path)
    assert len(registry.rules)==19 and list(registry.by_operator)==['Conv']

def test_registry_cli(registered,capsys):
    assert main(['rules','validate','--ruleset',str(registered)])==0
    assert '20 条' in capsys.readouterr().out
    assert main(['rules','list','--ruleset',str(registered),'--operator','Sigmoid','--json'])==0
    assert 'X5-SIGMOID-INPUT-RANK' in capsys.readouterr().out

def test_original_ten_preserved_and_basic_packs_registered():
    registry,_=load_ruleset(DEFAULT_RULESET)
    old={'Conv','Sigmoid','Concat','Slice','Add','Mul','Gemm','MatMul','Softmax','Resize'}
    assert old.issubset(registry.by_operator)
    assert set(registry.by_operator)-old=={'Reshape','Split','MaxPool','AveragePool'}
    assert sum(len(registry.by_operator[x].rules) for x in old)==49
    assert len(registry.all_rules)==66
    assert len(registry.rules)==19
    import subprocess
    from pathlib import Path
    root=Path(__file__).parents[1]
    original=subprocess.check_output(['git','show','HEAD:src/rdkx5_doctor/resources/rulesets/x5-bayes-e/conv2d.yaml'],cwd=root,text=True)
    assert (DEFAULT_RULESET/'conv2d.yaml').read_text()==original

def test_duplicate_yaml_keys(registered):
    text=(registered/'sigmoid.yaml').read_text()
    (registered/'sigmoid.yaml').write_text(text+'operator: Mul\n')
    with pytest.raises(ValueError,match='重复'):load_ruleset(registered)

def test_no_x3_or_cpu_constraints():
    registry,_=load_ruleset(DEFAULT_RULESET)
    for operator,op in registry.by_operator.items():
        if operator=='Conv':continue
        assert all(r.source_column=='X5 BPU 支持约束' and r.source_version=='1.1.2' for r in op.rules)
        assert not any(r.max==2048 or r.field=='dtype' for r in op.rules)


def test_reject_known_predicate_on_wrong_operator(registered):
    op=yaml.safe_load((registered/'sigmoid.yaml').read_text())
    rule=op['rules'][0]
    rule.update(check_type='predicate', predicate='x5_elementwise_broadcast_mergeable')
    rule.pop('min'); rule.pop('max')
    (registered/'sigmoid.yaml').write_text(yaml.safe_dump(op,allow_unicode=True))
    with pytest.raises(ValueError,match='谓词不属于对应算子/字段'):
        load_ruleset(registered)
