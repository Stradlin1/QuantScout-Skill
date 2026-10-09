import pytest
from rdkx5_doctor.graph_ir import GraphIR
from rdkx5_doctor.rules import check_operator

def check(rules,shapes,domain='',opset=11):
    n=dict(id='main/node_000000',original_name='matmul',op_type='MatMul',domain=domain,inputs=['a','b'],outputs=['y'],attributes={})
    tensors={name:dict(name=name,shape=shape,dtype='float32',producer=None,consumers=[],is_graph_input=True,is_initializer=False) for name,shape in zip(['a','b'],shapes)}
    tensors['y']=dict(name='y',shape=None,dtype='float32',producer=n['id'],consumers=[])
    ir=GraphIR({'opset_imports':[{'domain':'','version':opset}]},[n],tensors,[],[])
    return check_operator(ir,n,rules),n

@pytest.mark.parametrize('a,b,expected',[([2,3,4,5],[2,1,5,6],'NO_VIOLATION_FOUND'),([1,1,4,5],[2,3,5,6],'NO_VIOLATION_FOUND'),([2,3,4,5,6],[2,1,1,6,7],'NO_VIOLATION_FOUND'),([2,1,4,5],[1,3,5,6],'VIOLATION'),([2,1,4,5],[2,3,5,6],'VIOLATION'),([2,3,4,5,6],[2,1,4,6,7],'VIOLATION'),([3,4],[2,4,5],'VIOLATION'),([1,8192],[8192,1],'NO_VIOLATION_FOUND'),([1,8193],[8193,1],'VIOLATION'),([1,0],[0,1],'VIOLATION'),([4096,2,3],[4096,3,4],'NO_VIOLATION_FOUND'),([4097,2,3],[4097,3,4],'VIOLATION'),(['x',2,3],['y',3,4],'NEEDS_VERIFICATION'),(None,[3,4],'NEEDS_VERIFICATION'),([3],[3],'NEEDS_VERIFICATION')])
def test_matmul_boundaries(rules,a,b,expected):
    d,n=check(rules,[a,b]);assert d['status']==expected
    assert all(r['source']['column']=='X5 BPU 支持约束' and r['node_id']==n['id'] for r in d['results'])

def test_onnx_invalid_k_not_bpu_violation(rules):
    d,n=check(rules,[[2,3],[4,5]])
    assert d['status']=='NEEDS_VERIFICATION' and d['issues'] and not any(r['status']=='FAIL' for r in d['results'])

@pytest.mark.parametrize('domain,opset',[('custom',11),('',8),('',24)])
def test_domain_and_opset(rules,domain,opset):assert check(rules,[[2,3],[3,4]],domain,opset)[0]['status']=='NOT_COVERED'
