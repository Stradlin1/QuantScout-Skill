import copy
import onnx
from onnx import helper as h,TensorProto as T
from rdkx5_doctor.onnx_reader import read_model
from rdkx5_doctor.static_shape_propagation import refine_shapes,reshape_shape
from rdkx5_doctor.static_shape_values import slice_indices
import numpy as np
import pytest

def chain(tmp_path,shape=[1,64,8,8]):
    nodes=[h.make_node('Shape',['x'],['s']),h.make_node('Gather',['s','index'],['c'],axis=0),
           h.make_node('Div',['c','two'],['half']),h.make_node('Unsqueeze',['half'],['ends'],axes=[0]),
           h.make_node('Slice',['x','starts','ends','axes'],['y']),h.make_node('Relu',['y'],['z'])]
    init=[h.make_tensor('index',T.INT64,[],[1]),h.make_tensor('two',T.INT64,[],[2]),h.make_tensor('starts',T.INT64,[1],[0]),h.make_tensor('axes',T.INT64,[1],[1])]
    model=h.make_model(h.make_graph(nodes,'shape_test',[h.make_tensor_value_info('x',T.FLOAT,shape)],[h.make_tensor_value_info('z',T.FLOAT,[None]*4)],init),opset_imports=[h.make_opsetid('',11)])
    p=tmp_path/'chain.onnx';onnx.save(model,p);return read_model(p)

def test_computed_slice_shape_and_proof(tmp_path):
    ir=chain(tmp_path);assert type(ir.tensors['y']['shape'][1])is str
    r=refine_shapes(ir)
    assert ir.tensors['y']['shape']==[1,32,8,8] and ir.tensors['z']['shape']==[1,32,8,8]
    proof=next(p for p in r['proofs'] if p['tensor_name']=='y')
    assert proof['source_tensor_names']==['x','starts','ends','axes'] and proof['parameter_facts']
    assert r['summary']['conflict_count']==0
    assert refine_shapes(copy.deepcopy(chain(tmp_path)))==r

def test_conflict_and_symbol_preserved(tmp_path):
    ir=chain(tmp_path);ir.tensors['y']['shape'][1]=33
    r=refine_shapes(ir);assert ir.tensors['y']['shape'][1]==33 and r['summary']['conflict_count']>=1
    assert ir.tensors['z'].get('shape_conflict')
    assert next(f for f in r['facts'] if f['tensor_name']=='y')['axes'][1]['status']=='CONFLICT'
    ir=chain(tmp_path,[1,'c',8,8]);r=refine_shapes(ir)
    assert type(ir.tensors['y']['shape'][1])is str and r['unknown_origins']

@pytest.mark.parametrize('start,end,step',[(-100,100,1),(1,7,2),(7,-9,-1),(100,100,1),(7,-100,-2)])
def test_slice_matches_independent_numpy(start,end,step):
    shape,_=slice_indices([8],[start],[end],[0],[step]);assert shape==[np.arange(8)[start:end:step].size]

@pytest.mark.parametrize('source,target,want', [([2,3],[0,-1],[2,3]),([2,3],[3,2],[3,2]),([0,3],[0,3],[0,3])])
def test_reshape(source,target,want):assert reshape_shape(source,target)==want

@pytest.mark.parametrize('source,target',[([2,3],[-1,-1]),([2,3],[4,-1]),([2,3],[0,0,0])])
def test_reshape_conflicts(source,target):
    with pytest.raises(ValueError):reshape_shape(source,target)

def test_cycle_budget():
    from rdkx5_doctor.graph_ir import GraphIR
    ir=GraphIR({},[dict(id='a',op_type='Relu',domain='',inputs=['x'],outputs=['x'],attributes={})],{'x':dict(name='x',shape=None,dtype='float32',producer='a',consumers=['a'])},[dict(source='a',target='a',tensor='x')],[])
    r=refine_shapes(ir);assert r['summary']['budget_exceeded'] and r['unknown_origins']


def test_rank_conflict_is_not_counted_again_each_pass(tmp_path):
    ir=chain(tmp_path);ir.tensors['y']['shape']=[1,32,8]
    r=refine_shapes(ir)
    assert r['summary']['conflict_count']==1
    assert ir.tensors['y']['shape']==[1,32,8] and ir.tensors['z'].get('shape_conflict')
