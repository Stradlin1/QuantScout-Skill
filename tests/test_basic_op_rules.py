import onnx
from onnx import helper as h, TensorProto as T
import pytest
from rdkx5_doctor.onnx_reader import read_model
from rdkx5_doctor.rules import check_operator


def fixture(tmp_path, rules, op, attrs=None, shape=None, version=11, domain='', target=None, layout=False, missing_rank=False):
    shape=[1,8,1024,1024] if shape is None else shape
    inputs=[h.make_tensor_value_info('x',T.FLOAT,shape)];names=['x'];inits=[]
    if op=='Reshape':
        target=shape if target is None else target
        inits=[h.make_tensor('target',T.INT64,[len(target)],target)];names+=['target'];out_rank=len(target)
    else:out_rank=len(shape)
    outputs=['y','z'] if op=='Split' else ['y']
    model=h.make_model(h.make_graph([h.make_node(op,names,outputs,name='actual_node',domain=domain,**(attrs or {}))], 'basic',inputs,[h.make_tensor_value_info(n,T.FLOAT,[None]*out_rank) for n in outputs],inits),opset_imports=[h.make_opsetid('',version)]+([h.make_opsetid(domain,1)] if domain else []))
    onnx.checker.check_model(model)
    path=tmp_path/'basic.onnx';onnx.save(model,path);ir=read_model(path)
    if layout:ir.tensors['x']['layout']='NCHW'
    if missing_rank:ir.tensors['x']['shape']=None
    return check_operator(ir,ir.nodes[0],rules)


def result(d,field):return next(x for x in d['results'] if x['field']==field)

@pytest.mark.parametrize('shape,target,field,value,status',[([2,3],[3,2],'input_rank',2,'PASS'),([],[1],'input_rank',0,'FAIL'),([1]*10,[1]*10,'output_rank',10,'PASS'),([1]*11,[1]*11,'output_rank',11,'FAIL')])
def test_reshape_ranks(tmp_path,rules,shape,target,field,value,status):
    d=fixture(tmp_path,rules,'Reshape',shape=shape,target=target);c=result(d,field)
    assert c['actual']==value and c['status']==status and c['node_id']=='main/node_000000'
    assert c['source']['column']=='X5 BPU 支持约束'
    assert result(d,'reshape_conversion_verified')['status']=='UNKNOWN'

@pytest.mark.parametrize('axis,shape,sizes,field,expected',[(-3,[1,8,2,2],[4,4],'split_axis_not_batch','PASS'),(0,[4,8,2,2],[2,2],'split_axis_not_batch','FAIL'),(1,[1,8,2,2],[2,6],'split_lengths_divide_input','FAIL'),(1,[1,8,2,2],[4,4],'split_count_divides_input','UNKNOWN')])
def test_split_conditions(tmp_path,rules,axis,shape,sizes,field,expected):
    d=fixture(tmp_path,rules,'Split',attrs={'axis':axis,'split':sizes},shape=shape,layout=True)
    c=result(d,field);assert c['status']==expected and c['evidence']['split_sizes']==sizes
    assert c['source']['section'].startswith('6.3.3.3 X5 ONNX / Split')

def test_split_default_and_unknown_layout(tmp_path,rules):
    d=fixture(tmp_path,rules,'Split',attrs={'axis':1})
    assert result(d,'split_lengths_divide_input')['status']=='PASS'
    assert result(d,'split_axis_not_batch')['actual'] is None
    assert d['status']=='NEEDS_VERIFICATION'

def test_split_dynamic_size(tmp_path,rules):
    d=fixture(tmp_path,rules,'Split',attrs={'axis':1},shape=[1,'channels',2,2])
    assert result(d,'split_lengths_divide_input')['status']=='UNKNOWN'

@pytest.mark.parametrize('attrs,field,actual,status',[
    ({'kernel_shape':[256,256]},'pool_kernel_max',256,'PASS'),
    ({'kernel_shape':[257,2]},'pool_kernel_max',257,'FAIL'),
    ({'kernel_shape':[2,2],'strides':[256,256]},'pool_stride_max',256,'PASS'),
    ({'kernel_shape':[2,2],'strides':[257,1]},'pool_stride_max',257,'FAIL'),
    ({'kernel_shape':[257,257],'pads':[256]*4},'pool_padding_max',256,'PASS'),
    ({'kernel_shape':[258,258],'pads':[257]*4},'pool_padding_max',257,'FAIL'),
    ({'kernel_shape':[2,2]},'pool_no_dilation',True,'PASS'),
    ({'kernel_shape':[2,2],'dilations':[1,1]},'pool_no_dilation',True,'PASS'),
    ({'kernel_shape':[2,2],'dilations':[2,1]},'pool_no_dilation',False,'FAIL')])
def test_maxpool_boundaries(tmp_path,rules,attrs,field,actual,status):
    d=fixture(tmp_path,rules,'MaxPool',attrs);c=result(d,field)
    assert c['actual']==actual and c['status']==status and c['evidence']['attributes']==attrs

@pytest.mark.parametrize('attrs,field,actual,status',[
    ({'kernel_shape':[1,1]},'pool_kernel_area',1,'FAIL'),
    ({'kernel_shape':[64,128]},'pool_kernel_area',8192,'PASS'),
    ({'kernel_shape':[65,127]},'pool_kernel_area',8255,'FAIL'),
    ({'kernel_shape':[256,2]},'pool_kernel_h',256,'PASS'),
    ({'kernel_shape':[257,2]},'pool_kernel_h',257,'FAIL'),
    ({'kernel_shape':[2,2],'strides':[256,1]},'pool_stride_h',256,'PASS'),
    ({'kernel_shape':[2,2],'strides':[257,1]},'pool_stride_h',257,'FAIL'),
    ({'kernel_shape':[256,256],'pads':[255]*4},'pool_padding_max',255,'PASS'),
    ({'kernel_shape':[257,257],'pads':[256]*4},'pool_padding_max',256,'FAIL')])
def test_averagepool_boundaries(tmp_path,rules,attrs,field,actual,status):
    d=fixture(tmp_path,rules,'AveragePool',attrs);c=result(d,field)
    assert c['actual']==actual and c['status']==status and c['source']['version']=='1.1.2'

@pytest.mark.parametrize('op',['Reshape','Split','MaxPool','AveragePool'])
@pytest.mark.parametrize('version,domain',[(10,''),(12,''),(14,''),(11,'custom')])
def test_new_pack_scope(tmp_path,rules,op,version,domain):
    attrs={'kernel_shape':[2,2]} if 'Pool' in op else {'axis':1} if op=='Split' else {}
    d=fixture(tmp_path,rules,op,attrs,version=version,domain=domain)
    assert d['status']=='NOT_COVERED' and not d['results']

@pytest.mark.parametrize('op',['MaxPool','AveragePool'])
def test_pool_auto_pad_and_rank_unknown(tmp_path,rules,op):
    d=fixture(tmp_path,rules,op,{'kernel_shape':[2,2],'auto_pad':'SAME_UPPER'})
    assert result(d,'pool_padding_max')['status']=='UNKNOWN'
    d=fixture(tmp_path,rules,op,{'kernel_shape':[2,2]},missing_rank=True)
    assert all(c['status']=='UNKNOWN' for c in d['results'])
    d=fixture(tmp_path,rules,op,{'kernel_shape':[2]},shape=[1,2,1024])
    assert d['status']=='NOT_COVERED'

@pytest.mark.parametrize('op,attrs',[('Relu',{}),('Transpose',{'perm':[1,0,2,3]}),('Transpose',{})])
def test_no_fake_relu_transpose_rules(tmp_path,rules,op,attrs):
    d=fixture(tmp_path,rules,op,attrs)
    assert d['status']=='NOT_COVERED' and op not in rules.by_operator


def test_unequal_split_count_wording_is_review_only(tmp_path,rules):
    # Each 32/32/64 block divides 128 although output count 3 does not.
    # The official wording is ambiguous; neither observation proves FAIL.
    attrs={'axis':2,'split':[32,32,64]}
    shape=[1,2,128,4]
    outputs=['a','b','c']
    model=h.make_model(h.make_graph([h.make_node('Split',['x'],outputs,**attrs)],'unequal',[h.make_tensor_value_info('x',T.FLOAT,shape)],[h.make_tensor_value_info(n,T.FLOAT,[None]*4) for n in outputs]),opset_imports=[h.make_opsetid('',11)])
    onnx.checker.check_model(model);path=tmp_path/'split3.onnx';onnx.save(model,path);ir=read_model(path)
    d=check_operator(ir,ir.nodes[0],rules)
    assert result(d,'split_lengths_divide_input')['status']=='PASS'
    count=result(d,'split_count_divides_input')
    assert count['actual'] is False and count['status']=='UNKNOWN' and count['blocking'] is False
    assert d['status']=='NEEDS_VERIFICATION'  # N/layout independently unknown.
