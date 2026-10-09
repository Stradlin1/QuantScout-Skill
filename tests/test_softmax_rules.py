import pytest
from rdkx5_doctor.graph_ir import GraphIR
from rdkx5_doctor.rules import check_operator

def check(rules,shape,axis=None,opset=11,domain=''):
    attrs={} if axis is None else {'axis':axis}
    n=dict(id='main/node_000000',original_name='softmax',op_type='Softmax',domain=domain,inputs=['x'],outputs=['y'],attributes=attrs)
    ts={x:dict(name=x,shape=shape,dtype='float32',producer=None,consumers=[]) for x in ('x','y')}
    ir=GraphIR({'opset_imports':[{'domain':'','version':opset}]},[n],ts,[],[])
    return check_operator(ir,n,rules),n.get('operator_facts')

@pytest.mark.parametrize('shape,axis,status,path',[([1,2,3,4],3,'NO_VIOLATION_FOUND','ELIGIBLE_WITH_COMPILER_CONFIGURATION'),([1,2,3,4],-1,'NO_VIOLATION_FOUND','ELIGIBLE_WITH_COMPILER_CONFIGURATION'),([1,2,3,4],1,'VIOLATION','NOT_ELIGIBLE_UNDER_DOCUMENTED_ONNX_PATH'),([1,2,3],2,'VIOLATION','NOT_ELIGIBLE_UNDER_DOCUMENTED_ONNX_PATH'),(None,3,'NEEDS_VERIFICATION','UNKNOWN')])
def test_softmax_path(rules,shape,axis,status,path):
    d,f=check(rules,shape,axis);assert d['status']==status and f['static_bpu_path']==path
    assert f['actual_runtime_placement']=='UNKNOWN' and not f['compiler_configuration_verified']
    assert d['results'][-1]['status']=='UNKNOWN' and not d['results'][-1]['blocking']

def test_opset_defaults_and_semantics(rules):
    a,fa=check(rules,[1,2,3,4],opset=11);b,fb=check(rules,[1,2,3,4],opset=13)
    assert fa['axis_raw']==1 and fb['axis_raw']==-1 and fa['onnx_semantics']!=fb['onnx_semantics']
    assert b['status']=='NEEDS_VERIFICATION'  # new semantics != toolchain proof

def test_custom_and_invalid_axis(rules):
    assert check(rules,[1,2,3,4],3,domain='pytorch')[0]['status']=='NOT_COVERED'
    d,f=check(rules,[1,2,3,4],4);assert d['issues'] and any(r['status']=='UNKNOWN' for r in d['results'])
