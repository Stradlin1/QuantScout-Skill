import copy
import hashlib
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T, numpy_helper
import pytest
from rdkx5_doctor.onnx_reader import read_model
from rdkx5_doctor.optimization_candidates import detect_optimization_candidates, resolve_reshape
from rdkx5_doctor.small_constants import decode_integer_tensor, MAX_ELEMENTS, MAX_ENCODED_BYTES

@pytest.fixture
def graph(tmp_path):
    def make(nodes,shape=[1,2,3,4],initializers=None,outputs=None,opset=15,custom=False):
        model=h.make_model(h.make_graph(nodes,'patterns',[h.make_tensor_value_info('x',T.FLOAT,shape)],
              outputs or [h.make_tensor_value_info(nodes[-1].output[0],T.FLOAT,None)],initializers or []),
              opset_imports=[h.make_opsetid('',opset)]+([h.make_opsetid('custom',1)] if custom else []))
        # Supply public shape metadata without assuming intermediate connectivity.
        for output in model.graph.output:
            if not output.type.tensor_type.HasField('shape'):
                output.type.tensor_type.shape.CopyFrom(h.make_tensor_value_info('placeholder',T.FLOAT,[None]*len(shape)).type.tensor_type.shape)
        onnx.checker.check_model(model)
        path=tmp_path/f'graph{len(list(tmp_path.glob("*.onnx")))}.onnx'
        onnx.save(model,path)
        before=hashlib.sha256(path.read_bytes()).hexdigest()
        ir=read_model(path)
        assert before==hashlib.sha256(path.read_bytes()).hexdigest()
        return ir
    return make

def candidates(ir,pattern):
    return [c for c in detect_optimization_candidates(ir) if c['pattern']==pattern]

def test_identity_internal_boundary_and_duplicate_names(graph):
    ir=graph([h.make_node('Identity',['x'],['internal'],name='same'),h.make_node('Identity',['internal'],['y'],name='same')])
    cs=candidates(ir,'IDENTITY')
    assert len(cs)==2 and cs[0]['classification']=='SEMANTICALLY_REDUNDANT'
    assert cs[1]['classification']=='REVIEW_REQUIRED'
    assert any('Public graph output' in s for s in cs[1]['blockers'])
    assert cs[0]['node_names']==cs[1]['node_names'] and cs[0]['node_ids']!=cs[1]['node_ids']
    assert cs==candidates(ir,'IDENTITY')
    assert cs[0]['evidence']['input_tensor']=='x'

@pytest.mark.parametrize('a,b,found', [([0,2,3,1],[0,3,1,2],True),([0,2,3,1],[0,2,3,1],False),(None,None,True)])
def test_transpose_composition(graph,a,b,found):
    nodes=[h.make_node('Transpose',['x'],['mid'],**({'perm':a} if a else {})),
           h.make_node('Transpose',['mid'],['back'],**({'perm':b} if b else {})),h.make_node('Relu',['back'],['y'])]
    cs=candidates(graph(nodes),'TRANSPOSE_INVERSE_PAIR')
    assert bool(cs)==found
    if found:
        assert cs[0]['evidence']['composite_perm']==[0,1,2,3]
        assert cs[0]['classification']=='SEMANTICALLY_REDUNDANT'
        assert cs[0]['evidence']['connections'][0]['tensor']=='mid'

def test_transpose_fanout_boundary_and_unknown_rank(graph):
    nodes=[h.make_node('Transpose',['x'],['mid'],perm=[0,2,3,1]),
           h.make_node('Transpose',['mid'],['back'],perm=[0,3,1,2]),
           h.make_node('Relu',['mid'],['branch'])]
    outputs=[h.make_tensor_value_info('back',T.FLOAT,[1,2,3,4]),h.make_tensor_value_info('branch',T.FLOAT,[1,3,4,2]),h.make_tensor_value_info('mid',T.FLOAT,[1,3,4,2])]
    ir=graph(nodes,outputs=outputs)
    c=candidates(ir,'TRANSPOSE_INVERSE_PAIR')[0]
    assert c['classification']=='REVIEW_REQUIRED' and c['evidence']['intermediate_consumer_count']==2
    assert any('fanout' in s for s in c['blockers']) and any('mid' in s for s in c['blockers'])
    ir.tensors['x']['shape']=None
    c=candidates(ir,'TRANSPOSE_INVERSE_PAIR')[0]
    assert c['classification']=='INSUFFICIENT_INFORMATION'

@pytest.mark.parametrize('to,found',[(T.FLOAT,True),(T.INT8,False)])
def test_cast_same_dtype(graph,to,found):
    out=h.make_tensor_value_info('y',to,[1,2,3,4])
    ir=graph([h.make_node('Cast',['x'],['y'],to=to)],outputs=[out])
    assert bool(candidates(ir,'CAST_SAME_DTYPE'))==found

def test_cast_unknown_and_custom_domain(graph):
    ir=graph([h.make_node('Cast',['x'],['y'],to=T.FLOAT)])
    ir.tensors['x']['dtype']=None
    assert candidates(ir,'CAST_SAME_DTYPE')[0]['classification']=='INSUFFICIENT_INFORMATION'
    custom=graph([h.make_node('Identity',['x'],['y'],domain='custom')],custom=True)
    assert not detect_optimization_candidates(custom)

@pytest.mark.parametrize('target,found', [([1,2,3,4],True),([0,2,3,4],True),([-1,2,3,4],True),([1,24],False)])
def test_reshape_target_semantics(graph,target,found):
    init=numpy_helper.from_array(np.array(target,np.int64),name='target')
    ir=graph([h.make_node('Reshape',['x','target'],['reshaped']),h.make_node('Relu',['reshaped'],['y'])],initializers=[init],outputs=[h.make_tensor_value_info('y',T.FLOAT,[None]*len(target))])
    cs=candidates(ir,'RESHAPE_NOOP')
    assert bool(cs)==found
    if found:
        assert cs[0]['classification']=='SEMANTICALLY_REDUNDANT'
        assert cs[0]['evidence']['resolved_target']==[1,2,3,4]

@pytest.mark.parametrize('shape,target,allowzero,expected',[([],[],0,[]),([0,2],[0,2],1,[0,2]),([0,2],[-1,0],0,[0,2]),([0,2],[0,-1],0,None),([2,3],[0,-1],0,[2,3]),([2,3],[0,-1],1,None),([2,3],[-1,-1],0,None),([2,3],[0,0,0],0,None),([2,3],[4,2],0,None),(['N',3],[-1,3],0,None)])
def test_reshape_edge_cases(shape,target,allowzero,expected):
    resolved,reason=resolve_reshape(shape,target,allowzero=allowzero)
    assert resolved==expected
    assert bool(reason)==(expected is None)

def test_constant_node_shape_and_dynamic_target(graph):
    value=numpy_helper.from_array(np.array([1,2,3,4],np.int64))
    ir=graph([h.make_node('Constant',[],['target'],value=value),h.make_node('Reshape',['x','target'],['y'])])
    assert candidates(ir,'RESHAPE_NOOP')[0]['evidence']['resolved_target']==[1,2,3,4]
    dynamic=graph([h.make_node('Shape',['x'],['target']),h.make_node('Reshape',['x','target'],['y'])])
    assert candidates(dynamic,'RESHAPE_NOOP')[0]['classification']=='INSUFFICIENT_INFORMATION'
    ir.tensors['x']['shape']=['N',2,3,4]
    assert candidates(ir,'RESHAPE_NOOP')[0]['classification']=='INSUFFICIENT_INFORMATION'

def bn_graph(graph,opset=15,training=False,fanout=False,group=1):
    w=numpy_helper.from_array(np.zeros((2,2//group,1,1),np.float32),name='w')
    params=[numpy_helper.from_array(np.ones(2,np.float32),name=n) for n in ('scale','bias','mean','variance')]
    bnattrs={'training_mode':int(training)} if opset>=14 else {'is_test':int(not training)} if opset<=6 else {}
    outputs=['y','running_mean','running_var'] if training and opset>=14 else ['y']
    nodes=[h.make_node('Conv',['x','w'],['conv'],group=group),h.make_node('BatchNormalization',['conv','scale','bias','mean','variance'],outputs,**bnattrs)]
    public=[h.make_tensor_value_info('y',T.FLOAT,[1,2,3,4])]
    if training and opset>=14:
        public += [h.make_tensor_value_info(n,T.FLOAT,[2]) for n in outputs[1:]]
    if fanout:
        nodes.append(h.make_node('Relu',['conv'],['branch']))
        public.append(h.make_tensor_value_info('branch',T.FLOAT,[1,2,3,4]))
    return graph(nodes,initializers=[w]+params,opset=opset,outputs=public)

@pytest.mark.parametrize('opset',[6,7,9,13,14,15,23])
def test_bn_versioned_inference(graph,opset):
    ir=bn_graph(graph,opset=opset)
    c=candidates(ir,'CONV_BN_FUSION_REVIEW')[0]
    assert c['classification']=='REVIEW_REQUIRED' and c['evidence']['inference_mode']
    assert c['evidence']['channels']==2 and not c['evidence']['parameter_values_read']
    assert c['evidence']['connections'][0]['tensor']=='conv'

@pytest.mark.parametrize('opset',[6,15])
def test_training_bn_not_fusion(graph,opset):
    assert not candidates(bn_graph(graph,opset=opset,training=True),'CONV_BN_FUSION_REVIEW')

def test_bn_fanout_and_group_blockers(graph):
    c=candidates(bn_graph(graph,fanout=True,group=2),'CONV_BN_FUSION_REVIEW')[0]
    assert any('other consumers' in s for s in c['blockers'])
    assert any('Group convolution' in s for s in c['blockers'])

def test_overlap_and_nonadjacent_node_order(graph):
    # Node-list order is intentionally not the chain order for consecutive Transpose recognition.
    nodes=[h.make_node('Transpose',['x'],['t1'],perm=[0,2,3,1]),h.make_node('Relu',['x'],['branch']),
           h.make_node('Transpose',['t1'],['t2'],perm=[0,3,1,2]),h.make_node('Transpose',['t2'],['y'],perm=[0,2,3,1])]
    ir=graph(nodes,outputs=[h.make_tensor_value_info('branch',T.FLOAT,[1,2,3,4]),h.make_tensor_value_info('y',T.FLOAT,[1,3,4,2])])
    cs=candidates(ir,'TRANSPOSE_INVERSE_PAIR')
    assert len(cs)==2 and cs[0]['overlap_with']==[cs[1]['candidate_id']]
    assert cs[0]['node_ids']==['main/node_000000','main/node_000002']

def test_large_or_bad_constant_is_not_materialized():
    huge=h.make_tensor('shape',T.INT64,[MAX_ELEMENTS+1],list(range(MAX_ELEMENTS+1)))
    assert decode_integer_tensor(huge)['status']=='UNKNOWN'
    ext=copy.deepcopy(huge);ext.data_location=T.EXTERNAL
    assert 'External' in decode_integer_tensor(ext)['reason']
    bad=h.make_tensor('shape',T.INT64,[2],[1,2]);del bad.int64_data[:];bad.raw_data=b'\x00'
    assert 'truncated' in decode_integer_tensor(bad)['reason']
    inflated=h.make_tensor('x'*MAX_ENCODED_BYTES,T.INT64,[1],[1])
    assert 'byte' in decode_integer_tensor(inflated)['reason']

def test_missing_external_shape_constant(graph,tmp_path):
    init=numpy_helper.from_array(np.array([1,2,3,4],np.int64),name='target')
    ir=graph([h.make_node('Reshape',['x','target'],['y'])],initializers=[init])
    path=Path(ir.model['path'])
    model=onnx.load(path)
    external=tmp_path/'external.onnx'
    onnx.save_model(model,external,save_as_external_data=True,location='shape.bin',size_threshold=0)
    (tmp_path/'shape.bin').unlink()
    ir=read_model(external)
    assert ir.warnings
    assert candidates(ir,'RESHAPE_NOOP')[0]['classification']=='INSUFFICIENT_INFORMATION'
    assert ir.small_constants['target']['values'] is None

def test_identity_proof_does_not_require_inferred_metadata(graph):
    ir=graph([h.make_node('Identity',['x'],['internal']),h.make_node('Relu',['internal'],['y'])])
    ir.tensors['x'].update(shape=None,dtype=None)
    c=candidates(ir,'IDENTITY')[0]
    assert c['classification']=='SEMANTICALLY_REDUNDANT' and c['evidence']['input_metadata_missing']


def test_unreviewed_opset_is_not_confirmation(graph):
    ir=graph([h.make_node('Identity',['x'],['y'])])
    ir.model['opset_imports']=[{'domain':'','version':999}]
    assert candidates(ir,'IDENTITY')[0]['classification']=='INSUFFICIENT_INFORMATION'


def test_overridable_shape_initializer_not_constant(graph):
    from rdkx5_doctor.small_constants import collect_shape_constants
    init=numpy_helper.from_array(np.array([1,2,3,4],np.int64),name='target')
    ir=graph([h.make_node('Reshape',['x','target'],['y'])],initializers=[init])
    model=onnx.load(ir.model['path'])
    model.graph.input.append(h.make_tensor_value_info('target',T.INT64,[4]))
    resolved=collect_shape_constants(model)
    assert resolved['target']['status']=='UNKNOWN' and resolved['target']['values'] is None
    assert detect_optimization_candidates(ir,constants=resolved)[0]['classification']=='INSUFFICIENT_INFORMATION'


def test_real_allowzero_and_invalid_transpose_metadata(graph):
    init=numpy_helper.from_array(np.array([0,2],np.int64),name='target')
    ir=graph([h.make_node('Reshape',['x','target'],['y'],allowzero=1)],shape=[0,2],initializers=[init],
             outputs=[h.make_tensor_value_info('y',T.FLOAT,[0,2])],opset=14)
    c=candidates(ir,'RESHAPE_NOOP')[0]
    assert c['evidence']['resolved_target']==[0,2]
    ir=graph([h.make_node('Transpose',['x'],['a']),h.make_node('Transpose',['a'],['y'])])
    ir.nodes[0]['attributes']['perm']=[0,0,2,3]
    assert candidates(ir,'TRANSPOSE_INVERSE_PAIR')[0]['classification']=='INSUFFICIENT_INFORMATION'


def test_bn_unknown_channel_and_parameter_dtype_blocker(graph):
    ir=bn_graph(graph)
    ir.tensors['scale']['dtype']='float16'
    ir.tensors['conv']['shape']=[1,None,3,4]
    c=candidates(ir,'CONV_BN_FUSION_REVIEW')[0]
    assert any('channel metadata unknown' in s for s in c['blockers'])
    assert any('precision' in s for s in c['blockers'])
