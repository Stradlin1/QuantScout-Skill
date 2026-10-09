import pytest
from rdkx5_doctor.graph_ir import GraphIR
from rdkx5_doctor.operator_facts import extract_operator, fixed_constant_status
from rdkx5_doctor.rules import load_ruleset, check_operator
from rdkx5_doctor.cli import DEFAULT_RULESET

def fake_ir(op, shapes, output=None, attrs=None, fixed=None, opset=11):
    names=[f'x{i}' for i in range(len(shapes))]
    node=dict(id='main/node_000000',original_name='test',op_type=op,domain='',inputs=names,outputs=['y'],attributes=attrs or {})
    tensors={name:dict(name=name,shape=shape,dtype='float32',producer=None,consumers=[node['id']],is_graph_input=True,is_initializer=False) for name,shape in zip(names,shapes)}
    for name in fixed or []:tensors[name].update(is_graph_input=False,is_initializer=True)
    tensors['y']=dict(name='y',shape=output if output is not None else shapes[0],dtype='float32',producer=node['id'],consumers=[],is_graph_output=True)
    return GraphIR({'opset_imports':[{'domain':'','version':opset}]},[node],tensors,[],[])

def diag(ir):return check_operator(ir,ir.nodes[0],load_ruleset(DEFAULT_RULESET)[0])

@pytest.mark.parametrize('r,expected',[(0,'FAIL'),(1,'PASS'),(4,'PASS'),(10,'PASS'),(11,'FAIL')])
def test_sigmoid_rank_boundaries(r,expected):
    ir=fake_ir('Sigmoid',[[2]*r])
    d=diag(ir)
    assert [x['status'] for x in d['results']]==[expected,expected]
    assert d['results'][0]['source']['version']=='1.1.2'
    assert d['results'][0]['evidence']['tensor']=='x0'
    assert not any('int16' in r['field'] for r in d['results'])

def test_sigmoid_unknown_and_domain():
    ir=fake_ir('Sigmoid',[None]);assert diag(ir)['status']=='NEEDS_VERIFICATION'
    ir=fake_ir('Sigmoid',[[1,2]]);ir.nodes[0]['domain']='custom';assert diag(ir)['status']=='NOT_COVERED'
    ir=fake_ir('Sigmoid',[[1]],opset=5);assert diag(ir)['status']=='NOT_COVERED'

@pytest.mark.parametrize('axis,status',[(0,'VIOLATION'),(1,'NO_VIOLATION_FOUND'),(-4,'VIOLATION'),(-1,'NO_VIOLATION_FOUND'),(4,'NEEDS_VERIFICATION'),(None,'NEEDS_VERIFICATION')])
def test_concat_axes(axis,status):
    ir=fake_ir('Concat',[[1,2,3,4],[1,2,3,4]],attrs={} if axis is None else {'axis':axis})
    for t in ir.tensors.values():t['layout']='NCHW'
    assert diag(ir)['status']==status

def test_concat_layout_and_rank_conflict():
    assert diag(fake_ir('Concat',[[1,2],[1,2]],attrs={'axis':0}))['status']=='NEEDS_VERIFICATION'
    ir=fake_ir('Concat',[[1,'c',3,4],[1,'c',3,4]],attrs={'axis':1})
    for t in ir.tensors.values():t['layout']='NCHW'
    assert diag(ir)['status']=='NO_VIOLATION_FOUND'
    assert diag(fake_ir('Concat',[[1,2],[1,2,3]],attrs={'axis':1}))['status']=='NEEDS_VERIFICATION'

def test_rank_four_without_layout_is_unknown():
    ir=fake_ir('Concat',[[1,2,3,4],[1,2,3,4]],attrs={'axis':0})
    assert diag(ir)['status']=='NEEDS_VERIFICATION'

def test_constant_source_policy():
    t={'is_initializer':True,'is_graph_input':True}
    assert fixed_constant_status(t,{})=='UNKNOWN'
    assert fixed_constant_status({}, {})=='UNKNOWN'
    assert fixed_constant_status({'is_graph_input':True},{})=='FEATURE'
    assert fixed_constant_status({'producer':'c'},{'c':{'op_type':'Constant','domain':''}})=='FIXED'
    assert fixed_constant_status({'producer':'c'},{'c':{'op_type':'Constant','domain':'custom'}})=='FEATURE'
