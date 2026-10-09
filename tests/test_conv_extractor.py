import pytest
from rdkx5_doctor.onnx_reader import read_model
from rdkx5_doctor.conv_extractor import extract_conv

def extract(path):
    ir=read_model(path)
    return extract_conv(ir.nodes[0],ir.tensors)

def test_default_and_omitted_kernel(make_model):
    f=extract(make_model())['fields']
    assert (f['kernel_h'],f['kernel_w'],f['strides_h'],f['dilation_w'],f['group']) == (3,3,1,1,1)
    assert [f['pads_'+x] for x in ('top','left','bottom','right')] == [0]*4
    assert f['kernel_elements_per_group'] == 18

@pytest.mark.parametrize('cin,cout,group,cpg',[(8,12,4,2),(8,8,8,1)])
def test_group_and_depthwise(make_model,cin,cout,group,cpg):
    e=extract(make_model(cin=cin,cout=cout,group=group))
    assert e['fields']['in_channels_per_group']==cpg
    assert e['fields']['kernel_elements_per_group']==cpg*9
    assert e['fields']['out_channels_per_group']==cout//group
    assert not e['issues']

@pytest.mark.parametrize('mode,pads',[('VALID',[0,0,0,0]),('SAME_UPPER',[0,0,1,1]),('SAME_LOWER',[1,1,0,0])])
def test_auto_pad_semantics(make_model,mode,pads):
    f=extract(make_model(shape=[1,2,40,40],attrs={'strides':[2,2],'auto_pad':mode}))['fields']
    assert [f['pads_'+x] for x in ('top','left','bottom','right')]==pads

def test_dynamic_auto_pad(make_model):
    f=extract(make_model(shape=['batch',2,'height','width'],attrs={'auto_pad':'SAME_UPPER'}))['fields']
    assert f['input_n']=='batch' and f['pads_top'] is None
    assert f['kernel_h']==3

def test_weight_unknown(make_model):
    f=extract(make_model(unknown_weight=True))['fields']
    assert f['kernel_h'] is None and f['kernel_elements_per_group'] is None

def test_non_conv2d(make_model):
    assert extract(make_model(kernel=(3,3,3),dims=3))['scope']=='not_applicable'

def test_padding_order(make_model):
    f=extract(make_model(attrs={'pads':[1,2,3,4]}))['fields']
    assert [f['pads_'+x] for x in ('top','left','bottom','right')]==[1,2,3,4]

def test_attribute_conflict(make_model):
    e=extract(make_model(attrs={'kernel_shape':[1,1]}))
    assert e['fields']['kernel_h'] is None
    assert any('不一致' in x for x in e['issues'])
