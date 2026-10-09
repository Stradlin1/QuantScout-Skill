import json
import pytest
from rdkx5_doctor.tensor_resource import record, analyze_tensor_resources, WIDTHS
from rdkx5_doctor.graph_ir import GraphIR
from rdkx5_doctor.onnx_reader import read_model

def tensor(name='features',shape=None,**kw):
    return dict(name=name,shape=[1,64,320,320] if shape is None else shape,dtype='float32',producer='main/node_000000',consumers=[],**kw)

def test_exact_oracle_and_hypothetical():
    r=record(tensor())
    assert (r['element_count'],r['raw_bytes'],r['mib'],r['hypothetical_int8_bytes'])==(6553600,26214400,25.0,6553600)
    assert r['hypothetical_int8_bytes']/1048576==6.25

@pytest.mark.parametrize('dt,width',list(WIDTHS.items()))
def test_dtype_widths(dt,width):
    t=tensor(shape=[2,3]);t['dtype']=dt
    r=record(t)
    assert r['raw_bytes']==6*width
    assert r['hypothetical_int8_bytes']==(None if dt=='bool' else 6)

@pytest.mark.parametrize('shape,elements',[([],1),([1,0,320],0),([1,64],64),([2**63,2**63],2**126)])
def test_scalars_zero_and_unbounded_integer(shape,elements):
    r=record(tensor(shape=shape))
    assert r['element_count']==elements and r['raw_bytes']==elements*4

@pytest.mark.parametrize('shape',[['batch',64],[None,64],None,[-1,64],[True,64],[0,None]])
def test_unknown_shape(shape):
    t=tensor();t['shape']=shape
    r=record(t)
    assert r['raw_bytes'] is None and r['size_status']=='UNKNOWN_SHAPE' and r['unknown_reason']

@pytest.mark.parametrize('dt',['string','int4','uint4','float4e2m1','int2',None,'complex64','sequence'])
def test_unsupported_width(dt):
    t=tensor();t['dtype']=dt
    r=record(t)
    assert r['raw_bytes'] is None and r['size_status']=='UNKNOWN_OR_UNSUPPORTED_DTYPE'
    assert r['hypothetical_int8_bytes'] is None

def test_categories_fanout_partial_and_overlap():
    items=[tensor('weight',shape=[4],is_initializer=True,is_graph_input=True,is_graph_output=True),
           tensor('output',shape=[2],is_graph_output=True),tensor('feature',shape=[4]),
           tensor('dynamic',shape=['N',4]),tensor('const',shape=[2],kind='constant')]
    items[2]['consumers']=['a','b','a'];items[1]['consumers']=['c']
    items[4]['producer']='constant_node'
    ir=GraphIR({},[],{t['name']:t for t in items},[],[])
    r=analyze_tensor_resources(ir)
    by={t['name']:t for t in r['tensor_records']}
    assert by['weight']['resource_category']=='initializer' and by['weight']['classification_note']
    assert by['output']['resource_category']=='model_output' and by['output']['consumer_count']==1
    assert by['feature']['consumer_count']==2
    s=r['summary']
    assert s['initializers']['known_bytes_sum']==16
    assert s['model_outputs']['known_bytes_sum']==8
    assert s['intermediate_activations']=={'tensor_count':2,'known_bytes_sum':16,'unknown_tensor_count':1,'completeness':'PARTIAL'}
    assert s['multi_consumer_intermediate_count']==1
    assert s['largest_intermediates']==['feature']
    assert s['constants']['known_bytes_sum']==8
    json.dumps(r,allow_nan=False)

def test_input_unknown_category_and_sparse():
    t=tensor();t['producer']=None;t['is_graph_input']=True
    assert record(t)['resource_category']=='model_input'
    t['is_graph_input']=False
    assert record(t)['resource_category']=='other_unknown'
    t['storage_kind']='sparse'
    assert record(t)['raw_bytes'] is None and record(t)['hypothetical_int8_bytes'] is None

def test_real_ir_size_and_dedup(make_model):
    ir=read_model(make_model(branch=True))
    r=analyze_tensor_resources(ir)
    assert len(r['tensor_records'])==len(ir.tensors)
    assert r['summary']['largest_outputs']==['joined','sum']
    assert r['summary']['initializers']['known_bytes_sum']==144

def test_huge_display_float_cannot_invalidate_exact_bytes():
    r=record(tensor(shape=[2**63]*20))
    assert r['raw_bytes']==2**1262 and r['mib'] is None
    json.dumps(r,allow_nan=False)


def test_sparse_initializer_is_not_dense_payload(tmp_path):
    import onnx
    from onnx import helper as h,TensorProto as T
    values=h.make_tensor('sparse_weight',T.FLOAT,[1],[1.0])
    indices=h.make_tensor('indices',T.INT64,[1],[1])
    sparse=h.make_sparse_tensor(values,indices,[2,2])
    graph=h.make_graph([h.make_node('Identity',['sparse_weight'],['out'])],'sparse',[],[h.make_tensor_value_info('out',T.FLOAT,[2,2])])
    graph.sparse_initializer.append(sparse)
    model=h.make_model(graph,opset_imports=[h.make_opsetid('',15)])
    onnx.checker.check_model(model)
    path=tmp_path/'sparse.onnx';onnx.save(model,path)
    result=analyze_tensor_resources(read_model(path))
    records={t['name']:t for t in result['tensor_records']}
    assert records['sparse_weight']['resource_category']=='initializer'
    assert records['sparse_weight']['raw_bytes'] is None
    assert result['summary']['initializers']['completeness']=='PARTIAL'
