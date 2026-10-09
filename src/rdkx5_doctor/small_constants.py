"""Decode only requested inline integer shape vectors with hard budgets."""
from math import prod
import struct
from onnx import TensorProto, AttributeProto
from .dtype_utils import dtype

MAX_ELEMENTS = 64
MAX_ENCODED_BYTES = 4096

def decode_integer_tensor(tensor):
    result = {'status': 'UNKNOWN', 'values': None, 'shape': list(tensor.dims),
              'dtype': dtype(tensor.data_type), 'reason': None}
    def fail(reason):
        result['reason'] = reason
        return result
    if tensor.data_location == TensorProto.EXTERNAL or tensor.external_data:
        return fail('External data is never decoded by the small-constant resolver')
    if tensor.data_type not in (TensorProto.INT32, TensorProto.INT64):
        return fail('Only inline signed INT32/INT64 constants are supported')
    if len(tensor.dims) > 64 or any(d < 0 for d in tensor.dims):
        return fail('Malformed constant dimensions')
    count = prod(tensor.dims)
    if count > MAX_ELEMENTS:
        return fail(f'Constant exceeds {MAX_ELEMENTS}-element bound')
    if len(tensor.raw_data) > MAX_ENCODED_BYTES or tensor.ByteSize() > MAX_ENCODED_BYTES:
        return fail(f'Constant exceeds {MAX_ENCODED_BYTES}-byte encoded bound')
    width, code = (4, 'i') if tensor.data_type == TensorProto.INT32 else (8, 'q')
    typed = tensor.int32_data if width == 4 else tensor.int64_data
    if tensor.raw_data:
        if typed or len(tensor.raw_data) != count * width:
            return fail('Ambiguous or truncated integer encoding')
        values = list(struct.unpack('<' + code * count, tensor.raw_data))
    else:
        if len(typed) != count:
            return fail('Inline integer value count does not match dimensions')
        values = list(typed)
    result.update(status='KNOWN', values=values)
    return result

def collect_shape_constants(model):
    requested = {n.input[1] for n in model.graph.node if n.domain in ('', 'ai.onnx')
                 and n.op_type == 'Reshape' and len(n.input) > 1 and n.input[1]}
    requested.update(name for n in model.graph.node if n.domain in ('', 'ai.onnx')
                     and n.op_type == 'Slice' for name in n.input[1:5] if name)
    # Decode only constants in bounded shape-parameter cones. Shape stops
    # the backwards walk: its feature input values are never requested.
    producers={x:n for n in model.graph.node for x in n.output}
    requested.update(x for n in model.graph.node if n.op_type=='Resize' and n.domain in ('','ai.onnx') for x in n.input[3:4] if x)
    stack=[(x,0) for x in requested];seen=set()
    while stack and len(seen)<2048:
        name,depth=stack.pop()
        if name in seen or depth>64:continue
        seen.add(name);node=producers.get(name)
        if node and node.domain in ('','ai.onnx') and node.op_type!='Shape':
            stack.extend((x,depth+1) for x in node.input if x)
    requested=seen
    result = {name: {'status': 'UNKNOWN', 'values': None, 'reason': 'Target is not a supported inline constant'} for name in sorted(requested)}
    inputs = {v.name for v in model.graph.input}
    for init in model.graph.initializer:
        if init.name in requested:
            result[init.name] = decode_integer_tensor(init)
            result[init.name]['source'] = 'initializer'
            if init.name in inputs:
                result[init.name].update(status='UNKNOWN', values=None, reason='Initializer is also a graph input and can be overridden')
    for node in model.graph.node:
        if node.domain not in ('', 'ai.onnx') or node.op_type != 'Constant' or len(node.output) != 1 or node.output[0] not in requested:
            continue
        name = node.output[0]
        attrs = {a.name:a for a in node.attribute}
        value = attrs.get('value')
        if value is not None and value.type == AttributeProto.TENSOR:
            result[name] = decode_integer_tensor(value.t)
            result[name]['source'] = 'Constant.value'
        elif 'value_int' in attrs:
            result[name] = {'status':'KNOWN','values':[attrs['value_int'].i], 'shape':[], 'dtype':'int64','source':'Constant.value_int','reason':None}
        elif 'value_ints' in attrs:
            values = attrs['value_ints'].ints
            if len(values) <= MAX_ELEMENTS and attrs['value_ints'].ByteSize() <= MAX_ENCODED_BYTES:
                result[name] = {'status': 'KNOWN', 'values': list(values), 'shape': [len(values)],
                                'dtype': 'int64', 'source': 'Constant.value_ints', 'reason': None}
            else:
                result[name]['reason'] = 'Constant.value_ints exceeds count/encoded bounds'
    return result
