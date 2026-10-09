import yaml
import pytest
from rdkx5_doctor.rules import check_rule, check_node, load_ruleset, Rule
from rdkx5_doctor.onnx_reader import read_model
from rdkx5_doctor.conv_extractor import extract_conv
from rdkx5_doctor.cli import DEFAULT_RULESET

def diagnostic(path,rules):
    ir=read_model(path)
    return check_node(ir.nodes[0],extract_conv(ir.nodes[0],ir.tensors),rules)

def test_normal_pass(make_model,rules):
    d=diagnostic(make_model(),rules)
    assert d['status']=='NO_VIOLATION_FOUND'
    assert not any(r['status']=='FAIL' for r in d['results'])

@pytest.mark.parametrize('offset,status',[(0,'PASS'),(1,'FAIL')])
def test_official_kernel_boundary(make_model,rules,offset,status):
    rule=next(r for r in rules.rules if r.field=='kernel_h' and r.check_type=='range')
    d=diagnostic(make_model(kernel=(rule.max+offset,1)),rules)
    result=next(r for r in d['results'] if r['rule_id']==rule.id)
    assert result['status']==status and result['actual']==rule.max+offset

@pytest.mark.parametrize('offset,status',[(0,'PASS'),(1,'FAIL')])
def test_volume_boundary(make_model,rules,offset,status):
    rule=next(r for r in rules.rules if r.field=='kernel_elements_per_group')
    # Valid metadata-only input weight keeps fixtures tiny at the channel boundary.
    d=diagnostic(make_model(kernel=(1,1),cin=rule.max+offset,cout=1,shape=[1,rule.max+offset,1,1]),rules)
    assert next(r for r in d['results'] if r['rule_id']==rule.id)['status']==status

@pytest.mark.parametrize('field,attr,minimum,maximum',[('strides_h','strides',1,256),('dilation_h','dilations',1,16),('pads_top','pads',0,256)])
def test_numeric_boundaries(rules,field,attr,minimum,maximum):
    rule=next(r for r in rules.rules if r.field==field and r.check_type=='range')
    for value,status in [(minimum,'PASS'),(maximum,'PASS'),(minimum-1,'FAIL'),(maximum+1,'FAIL')]:
        assert check_rule(rule,{field:value})['status']==status

def test_dilated_checks_and_float_not_quantized(make_model,rules):
    d=diagnostic(make_model(shape=[1,2,41,40],attrs={'dilations':[2,2],'strides':[2,1]}),rules)
    results={r['rule_id']:r for r in d['results']}
    assert results['X5-CONV2D-DILATED-STRIDE-H']['status']=='FAIL'
    assert results['X5-CONV2D-DILATED-DIVISIBLE-H']['status']=='FAIL'
    assert results['X5-CONV2D-DILATED-OUTPUT']['status']=='UNKNOWN'
    assert results['X5-CONV2D-DILATED-OUTPUT']['actual'] is None

def test_unknown_and_non_applicable(make_model,rules):
    d=diagnostic(make_model(unknown_weight=True),rules)
    assert d['status']=='NEEDS_VERIFICATION'
    assert any(r['status']=='UNKNOWN' for r in d['results'])
    d=diagnostic(make_model(dims=3,kernel=(3,3,3)),rules)
    assert d['status']=='NOT_COVERED'
    assert all(r['status']=='NOT_APPLICABLE' for r in d['results'])

def test_nonstandard_domain_no_false_fail(make_model,rules):
    ir=read_model(make_model(kernel=(32,1)))
    node=ir.nodes[0];node['domain']='custom'
    d=check_node(node,extract_conv(node,ir.tensors),rules)
    assert d['status']=='NEEDS_VERIFICATION'
    assert not any(r['status']=='FAIL' for r in d['results'])

def test_schema_rejects_bad_rules(tmp_path):
    with pytest.raises(ValueError,match='规则集校验失败'):
        load_ruleset(tmp_path)
    for name in ('manifest.yaml','conv2d.yaml'):
        (tmp_path/name).write_text((DEFAULT_RULESET/name).read_text())
    data=yaml.safe_load((tmp_path/'conv2d.yaml').read_text())
    data['rules'][0]['field']='__import__("os")'
    (tmp_path/'conv2d.yaml').write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError,match='未知派生字段'):
        load_ruleset(tmp_path)
    data['rules'][0]['field']='kernel_h'
    data['rules'][0]['max']=0
    (tmp_path/'conv2d.yaml').write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError,match='range'):
        load_ruleset(tmp_path)

@pytest.mark.parametrize('op,actual,params,expected',[('equals',1,{'value':1},'PASS'),('one_of',2,{'values':[1,2]},'PASS'),('max_value','symbol',{'max':8},'UNKNOWN'),('equals',True,{'value':1},'FAIL')])
def test_generic_operations(op,actual,params,expected):
    r=Rule(id='test',check_type=op,field='kernel_h',checkability='auto_check',source_url='https://example.org',source_section='test',message='bad',**params)
    assert check_rule(r,{'kernel_h':actual})['status']==expected

def test_unknown_condition_and_conditional(rules):
    r=next(r for r in rules.rules if r.id=='X5-CONV2D-DILATED-OUTPUT')
    assert check_rule(r,{'has_dilation':None})['status']=='UNKNOWN'
    assert check_rule(r,{'has_dilation':False})['status']=='NOT_APPLICABLE'
    assert check_rule(r,{'has_dilation':True,'quantized_output_dtype':'int8'})['status']=='UNKNOWN'

def test_no_shortcut_guess(make_model,rules):
    d=diagnostic(make_model(branch=True,attrs={'strides':[3,3]}),rules)
    assert not any(r['status']=='FAIL' for r in d['results'])

def test_unknown_rank_is_not_assumed(rules):
    node={'id':'main/node_000000','op_type':'Conv','domain':'','inputs':['x','w'],'outputs':['y'],'attributes':{}}
    tensors={'x':{'shape':[1,2,8,8],'dtype':'float32'},'w':{'shape':None}}
    d=check_node(node,extract_conv(node,tensors),rules)
    assert d['status']=='NEEDS_VERIFICATION'
    assert all(r['status']=='UNKNOWN' for r in d['results'])

def test_reject_unsafe_or_duplicate_schema(tmp_path):
    for name in ('manifest.yaml','conv2d.yaml'):
        (tmp_path/name).write_text((DEFAULT_RULESET/name).read_text())
    base=yaml.safe_load((tmp_path/'conv2d.yaml').read_text())
    for mutate in ('duplicate','bad_condition','executable_url'):
        import copy
        data=copy.deepcopy(base)
        if mutate=='duplicate':
            data['rules'].append(data['rules'][0])
        elif mutate=='bad_condition':
            data['rules'][0]['when']={'field':'kernel_h','op':'greater_than','value':'string'}
        else:
            data['rules'][0]['source_url']='javascript:alert(1)'
        (tmp_path/'conv2d.yaml').write_text(yaml.safe_dump(data))
        with pytest.raises(ValueError,match='规则集校验失败'):
            load_ruleset(tmp_path)
