import itertools
import numpy as np
import pytest
from rdkx5_doctor.rule_predicates import broadcast_facts, x5_elementwise_broadcast_mergeable, at_most_one_fixed_constant_input
from rdkx5_doctor.operator_facts import extract_operator
from test_operator_facts import fake_ir,diag

@pytest.mark.parametrize('a,b,expected',[([2,5,4,5,3],[2,5,1,5,3],True),
    ([2,5,4,5,2],[1,1,1,5,2],True),([2,5,4,5,2],[2,1,4,1,2],False),
    ([2,1,4,1,2],[1,5,1,5,1],False),([1,2,3,4],[1,2,1,4],True),
    ([2,3],[3],True),(['n',2,3,4,5],['n',2,1,4,5],None),([0,2,3,4,5],[1,2,3,4,5],None)])
def test_official_adjacent_merge_patterns(a,b,expected):
    assert x5_elementwise_broadcast_mergeable({'input_shapes':[a,b]}) is expected

def test_broadcast_independent_numpy_oracle():
    shapes=[[],[1],[2],[3],[1,2],[2,1],[2,3],[1,2,1],[2,1,3]]
    for a,b in itertools.product(shapes,repeat=2):
        compatible,out,_=broadcast_facts([a,b])
        try:
            oracle=np.broadcast_shapes(tuple(a),tuple(b))
        except ValueError:
            assert compatible is False
        else:
            assert compatible is True and out==list(oracle)
    assert broadcast_facts([['a',2],['b',2]])[0] is None
    assert broadcast_facts([[0],[1]])[:2]==(True,[0])

@pytest.mark.parametrize('statuses,expected', [(['FIXED','FIXED'],False),(['FIXED','FEATURE'],True),
    (['UNKNOWN','FIXED'],None),(['UNKNOWN','FEATURE'],True),(['UNKNOWN','UNKNOWN'],None)])
def test_constant_predicate(statuses,expected):
    assert at_most_one_fixed_constant_input({'fixed_constant_statuses':statuses}) is expected

@pytest.mark.parametrize('op',['Add','Mul'])
@pytest.mark.parametrize('a,b,status', [([1,4,3,2],[1,4,3,2],'NO_VIOLATION_FOUND'),
    ([1,4,3,2],[4,1,1],'NO_VIOLATION_FOUND'),([1,2,1,4],[1,1,3,4],'NO_VIOLATION_FOUND'),
    ([2,5,4,5,2],[2,1,4,1,2],'VIOLATION'),([2]*10,[2]*10,'NO_VIOLATION_FOUND'),
    ([2]*11,[2]*11,'VIOLATION'),([],[],'VIOLATION'),([None,2],[1,2],'NO_VIOLATION_FOUND'),
    (['a',2],['b',2],'NEEDS_VERIFICATION'),([2,3],[4,3],'NEEDS_VERIFICATION')])
def test_elementwise_status(op,a,b,status):
    ir=fake_ir(op,[a,b]);assert diag(ir)['status']==status
    if a==[2,3] and b==[4,3]:assert not any(r['status']=='FAIL' for r in diag(ir)['results'])

@pytest.mark.parametrize('op',['Add','Mul'])
def test_constants_and_override(op):
    ir=fake_ir(op,[[2],[2]],fixed=['x0','x1']);assert diag(ir)['status']=='VIOLATION'
    ir.tensors['x0']['is_graph_input']=True;assert diag(ir)['status']=='NEEDS_VERIFICATION'
    ir=fake_ir(op,[[2],[2]],fixed=['x1']);assert diag(ir)['status']=='NO_VIOLATION_FOUND'
    ir.nodes[0]['domain']='custom';assert diag(ir)['status']=='NOT_COVERED'

def test_silu_tensor_connectivity():
    ir=fake_ir('Mul',[[1,2,3,4],[1,2,3,4]])
    sigmoid=dict(id='s',original_name='unrelated-name',op_type='Sigmoid',domain='',inputs=['x0'],outputs=['x1'],attributes={})
    ir.nodes.insert(0,sigmoid);ir.tensors['x1'].update(producer='s',is_graph_input=False)
    fact=extract_operator(ir,ir.nodes[1]);assert fact['pattern_evidence']['shared_tensor']=='x0'
    sigmoid['inputs']=['different'];assert 'pattern_evidence' not in extract_operator(ir,ir.nodes[1])

def test_add_stage():
    for a,b,status in [([1,2,3,4],[1,2,3,4],'NO_VIOLATION_FOUND'),
                       ([2,5,4,5,2],[2,1,4,1,2],'VIOLATION'),
                       (['a',2],['b',2],'NEEDS_VERIFICATION')]:
        assert diag(fake_ir('Add',[a,b]))['status']==status
