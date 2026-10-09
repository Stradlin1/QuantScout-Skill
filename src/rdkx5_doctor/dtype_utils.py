"""Single ONNX enum normalization shared by reader and pattern detectors."""
from onnx import TensorProto

def dtype(value):
    return {TensorProto.FLOAT: 'float32', TensorProto.DOUBLE: 'float64',
            TensorProto.FLOAT16: 'float16'}.get(value, TensorProto.DataType.Name(value).lower())
