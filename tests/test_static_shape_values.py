import pytest
from onnx import helper as h, TensorProto as T
from rdkx5_doctor.graph_ir import GraphIR
from rdkx5_doctor.static_shape_values import TinyValueFact,tiny,evaluate,dependency_cone
from rdkx5_doctor.small_constants import decode_integer_tensor

def node(op,inputs=('a','b'),attrs=None,domain=''):
    return dict(id='main/node_000001',op_type=op,inputs=list(inputs),outputs=['y'],attributes=attrs or {},domain=domain)

def fact(n,shape,values,dt='int64'):return tiny(n,dt,shape,values)

@pytest.mark.parametrize('op,a,b,want',[('Div',7,2,3),('Div',-7,2,-3),('Div',7,-2,-3),('Div',-7,-2,3),('Add',3,4,7),('Sub',3,4,-1),('Mul',3,4,12)])
def test_integer_arithmetic(op,a,b,want):
    r=evaluate(node(op),GraphIR({},[],{},[],[]),{'a':fact('a',[],[a]),'b':fact('b',[1],[b])},11)
    assert r.values==(want,) and r.value_shape==(1,)

@pytest.mark.parametrize('op,a,b,reason',[('Div',1,0,'DIVISION_BY_ZERO'),('Mul',2**62,4,'INTEGER_OVERFLOW_OR_CAST_LOSS')])
def test_fail_closed(op,a,b,reason):
    r=evaluate(node(op),GraphIR({},[],{},[],[]),{'a':fact('a',[],[a]),'b':fact('b',[],[b])},11)
    assert r.status=='UNKNOWN' and r.reason_code==reason

def test_shape_partial_and_gather():
    ir=GraphIR({},[],{'x':{'shape':[1,'batch',64,64]}},[],[])
    r=evaluate(node('Shape',['x']),ir,{},11)
    assert r.values==(1,None,64,64) and r.value_shape==(4,) and r.status=='PARTIAL'
    values={'a':r,'b':fact('b',[],[-1])}
    assert evaluate(node('Gather'),ir,values,11).values==(64,)
    values['b']=fact('b',[],[5]);assert evaluate(node('Gather'),ir,values,11).status=='UNKNOWN'

def test_budgets_and_schema():
    assert fact('a',[65],list(range(65))).reason_code=='CONSTANT_BUDGET_EXCEEDED'
    assert evaluate(node('Add',domain='custom'),GraphIR({},[],{},[],[]),{},11).reason_code=='UNSUPPORTED_OPERATOR_VERSION'
    r=fact('a',[],[1]);
    for i in range(65):r=tiny('a','int64',[],[1],node('Shape'),[r])
    assert r.reason_code=='SHAPE_PROPAGATION_BUDGET_EXCEEDED'

def test_external_and_truncated_integer():
    t=h.make_tensor('a',T.INT64,[1],[1]);t.data_location=T.EXTERNAL
    assert decode_integer_tensor(t)['status']=='UNKNOWN'
    t.data_location=T.DEFAULT;t.ClearField('int64_data');t.raw_data=b'123'
    assert decode_integer_tensor(t)['status']=='UNKNOWN'

def test_lossless_cast_and_partial_broadcast():
    ir=GraphIR({},[],{},[],[]);v={'a':fact('a',[2],[1,None]),'b':fact('b',[],[2])}
    assert evaluate(node('Add'),ir,v,11).values==(3,None)
    assert evaluate(node('Cast',['a'],{'to':T.INT32}),ir,v,11).status=='UNKNOWN'
    assert evaluate(node('Cast',['a'],{'to':T.INT64}),ir,v,11).status=='PARTIAL'
    v['a']=fact('a',[1],[2**40]);assert evaluate(node('Cast',['a'],{'to':T.INT32}),ir,v,11).status=='UNKNOWN'

@pytest.mark.parametrize('op,attrs,shape,vals,want_shape,want_values',[('Concat',{'axis':0},[2],[1,2],(4,),(1,2,1,2)),('Unsqueeze',{'axes':[0]},[2],[1,2],(1,2),(1,2)),('Squeeze',{'axes':[0]},[1,2],[1,2],(2,),(1,2)),('Cast',{'to':T.INT32},[2],[1,2],(2,),(1,2))])
def test_other_integer_transfers(op,attrs,shape,vals,want_shape,want_values):
    ir=GraphIR({},[],{},[],[]);v={'a':fact('a',shape,vals),'b':fact('b',shape,vals)}
    r=evaluate(node(op,['a','b'] if op=='Concat' else ['a'],attrs),ir,v,11)
    assert r.value_shape==want_shape and r.values==want_values

def test_tiny_slice_and_invalid_broadcast():
    ir=GraphIR({},[],{},[],[])
    vs={'a':fact('a',[4],[1,2,3,4]),'start':fact('start',[1],[1]),'end':fact('end',[1],[3])}
    assert evaluate(node('Slice',['a','start','end']),ir,vs,11).values==(2,3)
    vs={'a':fact('a',[2],[1,2]),'b':fact('b',[3],[1,2,3])}
    assert evaluate(node('Add'),ir,vs,11).reason_code=='ONNX_SEMANTIC_CONFLICT'

@pytest.mark.parametrize('index,shape,expected',[([0,2],[2],(10,30)),([-1,-3],[2],(30,10)),([1],[],(20,))])
def test_gather_index_rank(index,shape,expected):
    values={'a':fact('a',[3],[10,20,30]),'b':fact('b',shape,index)}
    r=evaluate(node('Gather'),GraphIR({},[],{},[],[]),values,11)
    assert r.values==expected and r.value_shape==tuple(shape)

def test_shape_value_is_not_tensor_metadata():
    ir=GraphIR({},[],{'x':{'shape':[1,3,640,640]}},[],[])
    r=evaluate(node('Shape',['x']),ir,{},11)
    assert r.value_shape==(4,) and r.values==(1,3,640,640) and r.status=='PROVEN'
