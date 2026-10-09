"""Generate a tiny valid Conv violation with a residual merge and two outputs."""
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T, numpy_helper


def generate(path):
    w=numpy_helper.from_array(np.zeros((2,2,32,1),dtype=np.float32),name='weight')
    nodes=[h.make_node('Conv',['image','weight'],['features'],name='oversized_kernel'),
        h.make_node('Relu',['features'],['activated'],name='branch_relu'),
        h.make_node('Identity',['features'],['skip'],name='residual_skip'),
        h.make_node('Add',['activated','skip'],['prediction'],name='merge'),
        h.make_node('Concat',['prediction','activated'],['auxiliary'],name='aux_head',axis=1)]
    graph=h.make_graph(nodes,'doctor_demo',[h.make_tensor_value_info('image',T.FLOAT,[1,2,40,8])],
        [h.make_tensor_value_info('prediction',T.FLOAT,[1,2,9,8]),h.make_tensor_value_info('auxiliary',T.FLOAT,[1,4,9,8])],[w])
    model=h.make_model(graph,opset_imports=[h.make_opsetid('',11)],producer_name='rdkx5-doctor-demo')
    onnx.checker.check_model(model)
    onnx.save(model,path)

if __name__ == '__main__':
    path=Path(__file__).with_name('demo.onnx')
    generate(path)
    print(path)
