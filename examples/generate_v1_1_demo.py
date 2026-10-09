"""Tiny five-pattern ONNX, no graph rewrite or runtime execution."""
from pathlib import Path
import json
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T, numpy_helper
from rdkx5_doctor.onnx_reader import read_model
from rdkx5_doctor.report import analyze, markdown
from rdkx5_doctor.rules import load_ruleset
from rdkx5_doctor.cli import DEFAULT_RULESET


def generate(path):
    nodes = [h.make_node('Identity',['image'],['copied'],name='input_identity'),
             h.make_node('Transpose',['copied'],['nhwc'],name='to_nhwc',perm=[0,2,3,1]),
             h.make_node('Transpose',['nhwc'],['nchw'],name='back_nchw',perm=[0,3,1,2]),
             h.make_node('Cast',['nchw'],['same_dtype'],name='same_float_cast',to=T.FLOAT),
             h.make_node('Reshape',['same_dtype','target_shape'],['same_shape'],name='same_shape_reshape'),
             h.make_node('Conv',['same_shape','weight'],['features'],name='normal_conv'),
             h.make_node('BatchNormalization',['features','scale','bias','mean','variance'],['prediction'],name='inference_bn')]
    initializers = [numpy_helper.from_array(np.array([0,2,4,4],np.int64),name='target_shape'),
                    numpy_helper.from_array(np.ones((2,2,1,1),np.float32),name='weight')]
    for name,value in [('scale',1),('bias',0),('mean',0),('variance',1)]:
        initializers.append(numpy_helper.from_array(np.full(2,value,np.float32),name=name))
    model = h.make_model(h.make_graph(nodes,'v1_1_demo',[h.make_tensor_value_info('image',T.FLOAT,[1,2,4,4])],
                [h.make_tensor_value_info('prediction',T.FLOAT,[1,2,4,4])],initializers),opset_imports=[h.make_opsetid('',15)],producer_name='rdkx5-doctor-v1.1-demo')
    onnx.checker.check_model(model)
    onnx.save(model,path)


def regenerate():
    folder=Path(__file__).resolve().parent
    path=folder/'v1_1_demo.onnx'
    generate(path)
    ir=read_model(path)
    ir.model['path']='examples/v1_1_demo.onnx'
    rules,manifest=load_ruleset(DEFAULT_RULESET)
    data=analyze(ir,rules,manifest)
    out=folder/'v1_1_demo-report'
    out.mkdir(exist_ok=True)
    (out/'analysis.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (out/'report.md').write_text(markdown(data),encoding='utf-8')
    print(f'{path}\n{out}')

if __name__=='__main__':
    regenerate()
