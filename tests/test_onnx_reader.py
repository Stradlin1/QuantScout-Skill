import onnx
import numpy as np
import pytest
from onnx import helper as h, TensorProto as T, numpy_helper
from rdkx5_doctor.onnx_reader import read_model,ModelError

def test_stable_ids_and_real_tensor_edges(make_model):
    ir=read_model(make_model(branch=True,name=''))
    assert [n['id'] for n in ir.nodes]==[f'main/node_{i:06d}' for i in range(5)]
    assert ir.nodes[1]['original_name']==ir.nodes[2]['original_name']
    assert ir.tensors['c']['consumers']==['main/node_000001','main/node_000002']
    assert len(ir.topology().edges)==6
    assert ir.model['outputs'][0]['producer']=='main/node_000003'

def test_dynamic_shape_preserved(make_model):
    ir=read_model(make_model(shape=['batch',2,None,'width']))
    assert ir.tensors['x']['shape']==['batch',2,None,'width']
    assert ir.model['opset_imports']==[{'domain':'','version':11}]

def test_invalid_file(tmp_path):
    p=tmp_path/'broken.onnx';p.write_text('not protobuf')
    with pytest.raises(ModelError,match='无法读取'):
        read_model(p)
    with pytest.raises(ModelError,match='.onnx'):
        read_model(tmp_path/'missing.onnx')

def test_missing_external_weights(tmp_path):
    w=numpy_helper.from_array(np.ones((2,2,3,3),np.float32),name='w')
    model=h.make_model(h.make_graph([h.make_node('Conv',['x','w'],['y'])],'external',[h.make_tensor_value_info('x',T.FLOAT,[1,2,8,8])],[h.make_tensor_value_info('y',T.FLOAT,[1,2,6,6])],[w]),opset_imports=[h.make_opsetid('',11)])
    path=tmp_path/'external.onnx'
    onnx.save_model(model,path,save_as_external_data=True,all_tensors_to_one_file=True,location='weights.bin',size_threshold=0)
    (tmp_path/'weights.bin').unlink()
    ir=read_model(path)
    assert ir.model['validation']=='metadata_only' and ir.model['missing_external_weights']==['w']
    assert ir.tensors['w']['shape']==[2,2,3,3]
    assert ir.warnings

def test_external_missing_does_not_mask_invalid_graph(tmp_path):
    w=numpy_helper.from_array(np.ones((1,),np.float32),name='w')
    model=h.make_model(h.make_graph([h.make_node('Add',['unknown','w'],['y'])],'external',[],[h.make_tensor_value_info('y',T.FLOAT,[1])],[w]))
    path=tmp_path/'external.onnx'
    onnx.save_model(model,path,save_as_external_data=True,location='w.bin',size_threshold=0)
    (tmp_path/'w.bin').unlink()
    with pytest.raises(ModelError,match='结构校验失败'):
        read_model(path)

def test_nested_graph_explicit_warning(tmp_path):
    branch=h.make_graph([h.make_node('Identity',['x'],['out'])],'branch',[],[h.make_tensor_value_info('out',T.FLOAT,[1])])
    model=h.make_model(h.make_graph([h.make_node('If',['cond'],['y'],then_branch=branch,else_branch=branch)],'main',[h.make_tensor_value_info('x',T.FLOAT,[1]),h.make_tensor_value_info('cond',T.BOOL,[])],[h.make_tensor_value_info('y',T.FLOAT,[1])]),opset_imports=[h.make_opsetid('',11)])
    path=tmp_path/'nested.onnx';onnx.save(model,path)
    assert any('子图解析未覆盖' in x for x in read_model(path).warnings)

def test_constant_weight_and_empty_optional_input(tmp_path):
    weight=numpy_helper.from_array(np.zeros((1,1,3,3),np.float32))
    nodes=[h.make_node('Constant',[],['w'],value=weight),h.make_node('Conv',['x','w',''],['y'])]
    model=h.make_model(h.make_graph(nodes,'constant',[h.make_tensor_value_info('x',T.FLOAT,[1,1,5,5])],[h.make_tensor_value_info('y',T.FLOAT,[1,1,3,3])]),opset_imports=[h.make_opsetid('',11)])
    path=tmp_path/'constant.onnx';onnx.save(model,path)
    ir=read_model(path)
    assert '' not in ir.tensors and ir.tensors['w']['kind']=='constant'
    assert ir.tensors['w']['shape']==[1,1,3,3]
