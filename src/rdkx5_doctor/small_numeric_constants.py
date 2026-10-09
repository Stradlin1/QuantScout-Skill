"""Requested immutable FLOAT32/FLOAT64 metadata only; no feature arrays."""
import math
import struct
from math import prod
from onnx import TensorProto
from .dtype_utils import dtype

def decode_numeric_tensor(t):
    result=dict(status='UNKNOWN',values=None,shape=list(t.dims),dtype=dtype(t.data_type),reason_code=None)
    def fail(code):result['reason_code']=code;return result
    if t.data_location==TensorProto.EXTERNAL or t.external_data:return fail('EXTERNAL_DATA_NOT_READ')
    if t.data_type not in (TensorProto.FLOAT,TensorProto.DOUBLE):return fail('UNSUPPORTED_CONSTANT_DTYPE')
    if len(t.dims)>64 or any(x<0 for x in t.dims) or prod(t.dims)>64 or t.ByteSize()>4096:return fail('CONSTANT_BUDGET_EXCEEDED')
    count=prod(t.dims);width,code=(4,'f') if t.data_type==TensorProto.FLOAT else (8,'d');typed=t.float_data if width==4 else t.double_data
    if t.raw_data:
        if typed or len(t.raw_data)!=count*width:return fail('ONNX_SEMANTIC_CONFLICT')
        values=list(struct.unpack('<'+code*count,t.raw_data))
    else:
        if len(typed)!=count:return fail('ONNX_SEMANTIC_CONFLICT')
        values=list(typed)
    if not all(math.isfinite(x) for x in values):return fail('NONFINITE_CONSTANT')
    result.update(status='KNOWN',values=values);return result

def collect_numeric_constants(model):
    requested={x for n in model.graph.node if n.op_type=='Resize' and n.domain in ('','ai.onnx')
               for x in (n.input[1:3] if len(n.input)>2 else n.input[1:2]) if x}
    results={x:dict(status='UNKNOWN',values=None,reason_code='VALUE_NOT_STATIC') for x in requested}
    graphinputs={v.name for v in model.graph.input}
    for t in model.graph.initializer:
        if t.name in requested:
            results[t.name]=decode_numeric_tensor(t)
            if t.name in graphinputs:results[t.name].update(status='UNKNOWN',values=None,reason_code='OVERRIDABLE_INITIALIZER')
    for n in model.graph.node:
        if n.op_type=='Constant' and n.domain in ('','ai.onnx') and len(n.output)==1 and n.output[0] in requested:
            attrs={a.name:a for a in n.attribute}
            if 'value' in attrs:results[n.output[0]]=decode_numeric_tensor(attrs['value'].t)
    return results
