from fractions import Fraction
import pytest
from onnx import helper as h,TensorProto as T
from rdkx5_doctor.graph_ir import GraphIR
from rdkx5_doctor.rules import check_operator
from rdkx5_doctor.small_numeric_constants import decode_numeric_tensor
from rdkx5_doctor.resize_op_extractor import nearest_enlargement

def check(rules,scales=(1,1,2,2),mode='nearest',coord='asymmetric',layout=True,shape=[1,3,8,8],opset=11,sizes=None,roi=None):
    n=dict(id='main/node_000000',original_name='resize',op_type='Resize',domain='',inputs=['x','roi','scales','sizes'] if opset!=10 else ['x','scales'],outputs=['y'],attributes={'mode':mode,**({'coordinate_transformation_mode':coord} if opset!=10 else {})})
    ts={name:dict(name=name,shape=shape if name=='x' else None,dtype='float32',producer=None,consumers=[],**({'layout':'NCHW'} if name=='x' and layout else {})) for name in n['inputs']+['y']}
    numeric={'scales':dict(status='KNOWN',values=list(scales),shape=[len(scales)],dtype='float32')} if scales is not None else {}
    if roi is not None:numeric['roi']=dict(status='KNOWN',values=roi,shape=[len(roi)],dtype='float32')
    small={'sizes':dict(status='KNOWN',values=sizes,shape=[len(sizes)],dtype='int64')} if sizes else {}
    ir=GraphIR({'opset_imports':[{'domain':'','version':opset}]},[n],ts,[],[],small,numeric)
    return check_operator(ir,n,rules),n

@pytest.mark.parametrize('scales,mode,coord,status',[((1,1,2,2),'nearest','asymmetric','NO_VIOLATION_FOUND'),((1,1,2,4),'nearest','half_pixel','NO_VIOLATION_FOUND'),((1,1,4,2),'nearest','asymmetric','VIOLATION'),((1,1,3,3),'nearest','asymmetric','VIOLATION'),((1,1,.5,.25),'nearest','asymmetric','NO_VIOLATION_FOUND'),((1,1,3,3),'linear','align_corners','NO_VIOLATION_FOUND'),((1,1,2,2),'cubic','half_pixel','VIOLATION'),((1,1,2,2),'nearest','tf_half_pixel_for_nn','VIOLATION'),((2,1,2,2),'nearest','asymmetric','VIOLATION')])
def test_resize_patterns(rules,scales,mode,coord,status):assert check(rules,scales,mode,coord)[0]['status']==status

def test_unknown_layout_sizes_and_version(rules):
    assert check(rules,layout=False)[0]['status']=='NEEDS_VERIFICATION'
    assert check(rules,scales=None,sizes=[1,3,16,16])[0]['status']=='NO_VIOLATION_FOUND'
    assert check(rules,scales=None)[0]['status']=='NEEDS_VERIFICATION'
    assert check(rules,opset=10)[0]['status']=='NEEDS_VERIFICATION'
    assert check(rules,shape=[1,3,8],scales=[1,1,2])[0]['status']=='VIOLATION'

def test_crop_roi(rules):
    assert check(rules,coord='tf_crop_and_resize',roi=[0,0,0,0,1,1,1,1])[0]['status']=='NO_VIOLATION_FOUND'
    assert check(rules,coord='tf_crop_and_resize',roi=[0,0,.5,0,1,1,1,1])[0]['status']=='VIOLATION'
    assert check(rules,coord='tf_crop_and_resize',roi=None)[0]['status']=='NEEDS_VERIFICATION'

@pytest.mark.parametrize('dt',[T.FLOAT,T.DOUBLE])
def test_float_decode(dt):
    t=h.make_tensor('v',dt,[4],[1,1,2,2]);assert decode_numeric_tensor(t)['values']==[1,1,2,2]
    t.data_location=T.EXTERNAL;assert decode_numeric_tensor(t)['reason_code']=='EXTERNAL_DATA_NOT_READ'
    t.data_location=T.DEFAULT;t.dims[0]=65;assert decode_numeric_tensor(t)['reason_code']=='CONSTANT_BUDGET_EXCEEDED'

def test_nonfinite_and_encoded_count():
    t=h.make_tensor('v',T.FLOAT,[1],[float('nan')]);assert decode_numeric_tensor(t)['status']=='UNKNOWN'
    t.ClearField('float_data');t.raw_data=b'12';assert decode_numeric_tensor(t)['status']=='UNKNOWN'

def test_mixed_factors_unknown():assert nearest_enlargement(list(map(Fraction,[1,1,.5,2]))) is None

def test_overridable_float_initializer_not_static(tmp_path):
    import onnx
    from rdkx5_doctor.small_numeric_constants import collect_numeric_constants
    n=h.make_node('Resize',['x','roi','scales'],['y'],mode='nearest')
    m=h.make_model(h.make_graph([n],'override',[h.make_tensor_value_info('x',T.FLOAT,[1,3,8,8]),h.make_tensor_value_info('scales',T.FLOAT,[4])],[h.make_tensor_value_info('y',T.FLOAT,[1,3,16,16])],[h.make_tensor('scales',T.FLOAT,[4],[1,1,2,2]),h.make_tensor('roi',T.FLOAT,[0],[])]),opset_imports=[h.make_opsetid('',11)])
    r=collect_numeric_constants(m)['scales'];assert r['values'] is None and r['reason_code']=='OVERRIDABLE_INITIALIZER'
