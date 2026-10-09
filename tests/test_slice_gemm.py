import numpy as np
import onnx
from onnx import helper as h, TensorProto as T, numpy_helper
import pytest
from rdkx5_doctor.onnx_reader import read_model
from rdkx5_doctor.operator_facts import extract_operator
from test_operator_facts import fake_ir, diag

@pytest.mark.parametrize('shape,steps', [([4],None),([2,4],[-1]),([1,2,3,4],[1])])
def test_slice_bounded_inputs_and_defaults(tmp_path,shape,steps):
    values={'starts':[0] if steps is None or steps[0]>0 else [3], 'ends':[2] if steps is None or steps[0]>0 else [-5]}
    if steps is not None:values.update(axes=[len(shape)-1],steps=steps)
    inits=[numpy_helper.from_array(np.array(v,np.int64),name=k) for k,v in values.items()]
    node=h.make_node('Slice',['x']+list(values),['y'])
    model=h.make_model(h.make_graph([node],'slice',[h.make_tensor_value_info('x',T.FLOAT,shape)],
           [h.make_tensor_value_info('y',T.FLOAT,[None]*len(shape))],inits),opset_imports=[h.make_opsetid('',11)])
    onnx.checker.check_model(model);path=tmp_path/'slice.onnx';onnx.save(model,path)
    ir=read_model(path);fact=extract_operator(ir,ir.nodes[0])
    assert fact['fields']['slice_parameters_fixed'] is True
    assert fact['field_evidence']['slice_parameters_fixed']['data_dtype']=='float32'
    assert not fact['issues']
    assert diag(ir)['status']=='NEEDS_VERIFICATION'  # capability is not an applicable hard constraint
    assert not any(r['status']=='FAIL' for r in diag(ir)['results'])

def test_slice_dynamic_and_old_schema():
    ir=fake_ir('Slice',[[4],[1],[1]])
    fact=extract_operator(ir,ir.nodes[0]);assert fact['fields']['slice_parameters_fixed'] is False
    assert diag(ir)['status']=='NEEDS_VERIFICATION'
    ir.model['opset_imports'][0]['version']=9;assert diag(ir)['status']=='NOT_COVERED'

def test_slice_overridable_and_invalid_metadata():
    ir=fake_ir('Slice',[[4],[1],[1]])
    ir.small_constants={'x1':{'status':'UNKNOWN','values':None,'reason':'overridable'},'x2':{'status':'KNOWN','values':[4],'shape':[1],'dtype':'int64'}}
    assert extract_operator(ir,ir.nodes[0])['fields']['slice_parameters_fixed'] is False
    ir.small_constants.update(x1={'status':'KNOWN','values':[0],'shape':[1],'dtype':'int64'},x3={'status':'KNOWN','values':[0],'shape':[1],'dtype':'int64'},x4={'status':'KNOWN','values':[0],'shape':[1],'dtype':'int64'})
    ir.nodes[0]['inputs']+=['x3','x4'];ir.tensors.update(x3={'name':'x3'},x4={'name':'x4'})
    assert 'ONNX Slice step 不能为 0' in extract_operator(ir,ir.nodes[0])['issues']
    assert not any(r['status']=='FAIL' for r in diag(ir)['results'])

@pytest.mark.parametrize('attrs,a,b,mkn', [({},[2,3],[3,4],(2,3,4)),
    ({'transA':1,'transB':1,'alpha':2.0,'beta':0.5},[3,2],[4,3],(2,3,4)),
    ({'transB':1},['batch',3],[4,3],('batch',3,4))])
def test_gemm_logical_metadata_not_conv(attrs,a,b,mkn):
    ir=fake_ir('Gemm',[a,b,[4]],output=[2,4],attrs=attrs,fixed=['x1','x2'])
    fact=extract_operator(ir,ir.nodes[0]);assert tuple(fact['gemm'][k] for k in ('M','K','N'))==mkn
    assert fact['gemm']['onnx_matrix_compatible'] is True
    assert fact['gemm']['bias_compatible'] is True
    assert fact['conversion_target']=='Conv' and not fact['conversion_layout_known']
    assert diag(ir)['status']=='NEEDS_VERIFICATION'
    assert all(r['status']=='UNKNOWN' for r in diag(ir)['results'])

def test_gemm_unknown_and_onnx_mismatch():
    ir=fake_ir('Gemm',[[2,3],None]);assert diag(ir)['status']=='NEEDS_VERIFICATION'
    ir=fake_ir('Gemm',[[2,3],[5,4]])
    fact=extract_operator(ir,ir.nodes[0]);assert fact['gemm']['onnx_matrix_compatible'] is False
    assert any('ONNX_GEMM_K_MISMATCH' in x for x in fact['issues'])
    assert not any(r['status']=='FAIL' for r in diag(ir)['results'])
