from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T, numpy_helper
import pytest
from rdkx5_doctor.cli import DEFAULT_RULESET
from rdkx5_doctor.rules import load_ruleset

@pytest.fixture
def rules():
    return load_ruleset(DEFAULT_RULESET)[0]

@pytest.fixture
def make_model(tmp_path):
    def make(kernel=(3,3), cin=2, cout=2, group=1, shape=None, attrs=None, name='conv', branch=False, unknown_weight=False, dims=2):
        shape = [1,cin]+[40]*dims if shape is None else shape
        wshape = [cout, cin//group] + list(kernel)
        weight = numpy_helper.from_array(np.zeros(wshape,dtype=np.float32),name='w')
        inputs = [h.make_tensor_value_info('x',T.FLOAT,shape)]
        if unknown_weight:
            inputs.append(h.make_tensor_value_info('w',T.FLOAT,[None]*len(wshape)))
        attributes = dict(attrs or {})
        if group != 1:
            attributes['group'] = group
        nodes = [h.make_node('Conv',['x','w'],['c'],name=name,**attributes)]
        output_names = ['c']
        if branch:
            nodes += [h.make_node('Relu',['c'],['a'],name='same'),h.make_node('Identity',['c'],['b'],name='same'),
                h.make_node('Add',['a','b'],['sum']),h.make_node('Concat',['sum','a'],['joined'],axis=1)]
            output_names = ['sum','joined']
        outputs = [h.make_tensor_value_info(n,T.FLOAT,[None]*(dims+2)) for n in output_names]
        model = h.make_model(h.make_graph(nodes,'test',inputs,outputs,[] if unknown_weight else [weight]),opset_imports=[h.make_opsetid('',11)])
        onnx.checker.check_model(model)
        path = tmp_path / ('model_'+str(len(list(tmp_path.glob('*.onnx'))))+'.onnx')
        onnx.save(model,path)
        return path
    return make
