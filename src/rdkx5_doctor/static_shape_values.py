"""Typed bounded integer expressions; never evaluate neural feature values."""
from dataclasses import dataclass, asdict
from math import prod
import numpy as np
import onnx

LIMITS = dict(max_elements=64, max_encoded_bytes=4096, max_value_bytes=4096,
              max_dependency_depth=64, max_visited_shape_nodes=2048,
              max_propagation_passes=4, max_unknown_trace_nodes=64,
              max_abs_integer_value=2**63-1)
SCHEMAS = {'Shape': {1,13,15,19,21,23}, 'Gather': {1,11,13},
           'Add': {7,13,14}, 'Sub': {7,13,14}, 'Mul': {7,13,14}, 'Div': {7,13,14},
           'Concat': {4,11,13}, 'Unsqueeze': {1,11,13,21,23}, 'Squeeze': {1,11,13,21,23},
           'Cast': {6,9,13,19,21,23}, 'Slice': {10,11,13}, 'Constant': {1,9,11,12,13,19,21,23}}

@dataclass(frozen=True)
class TinyValueFact:
    tensor_name: str
    onnx_dtype: str | None = None
    value_shape: tuple | None = None
    values: tuple | None = None
    status: str = 'UNKNOWN'
    source_node_ids: tuple = ()
    source_tensor_names: tuple = ()
    reason_code: str | None = None
    explanation: str = ''
    depth: int = 0

    def json(self):
        return asdict(self)

def schema_ok(node, opset, table=SCHEMAS):
    if node.get('domain','') not in ('','ai.onnx') or opset is None or not 1 <= opset <= 23:
        return False
    try:
        schema=onnx.defs.get_schema(node['op_type'],opset,'')
        return schema.since_version in table.get(node['op_type'],set()) and not set(node['attributes'])-set(schema.attributes)
    except onnx.onnx_cpp2py_export.defs.SchemaError:
        return False

def tiny(name, dtype, shape, values, node=None, dependencies=(), reason=None):
    sources=list(dict.fromkeys([s for d in dependencies for s in d.source_node_ids]+([node['id']] if node else [])))
    tensors=list(dict.fromkeys([d.tensor_name for d in dependencies]))
    depth=1+max((d.depth for d in dependencies),default=0)
    if depth>LIMITS['max_dependency_depth']:
        reason='SHAPE_PROPAGATION_BUDGET_EXCEEDED';values=None
    if shape is not None and (len(shape)>64 or any(type(x)is not int or x<0 for x in shape) or prod(shape)>64):
        reason='CONSTANT_BUDGET_EXCEEDED';values=None
    if values is not None:
        if len(values)>64 or shape is None or prod(shape)!=len(values):reason='CONSTANT_BUDGET_EXCEEDED';values=None
        elif dtype not in ('int32','int64'):reason='INTEGER_OVERFLOW_OR_CAST_LOSS';values=None
        else:
            bound=2**(31 if dtype=='int32' else 63)
            if any(v is not None and (type(v)is not int or not -bound<=v<bound or abs(v)>LIMITS['max_abs_integer_value']) for v in values):
                reason='INTEGER_OVERFLOW_OR_CAST_LOSS';values=None
    status='UNKNOWN' if values is None else 'PARTIAL' if None in values else 'PROVEN'
    return TinyValueFact(name,dtype,tuple(shape) if shape is not None else None,
                         tuple(values) if values is not None else None,status,tuple(sources[-2048:]),
                         tuple(tensors),reason or ('SYMBOLIC_UPSTREAM_DIM' if status=='PARTIAL' else None),
                         f"{node['op_type'] if node else 'inline constant'}: bounded integer semantics",depth)

def slice_indices(shape, starts, ends, axes=None, steps=None):
    """Python slice.indices agrees with reviewed ONNX Slice-10+ clamping."""
    axes=list(range(len(starts))) if axes is None else axes
    steps=[1]*len(starts) if steps is None else steps
    if not len(starts)==len(ends)==len(axes)==len(steps):raise ValueError('ONNX_SEMANTIC_CONFLICT')
    rank=len(shape); normalized=[];selection=[slice(None)]*rank
    for start,end,axis,step in zip(starts,ends,axes,steps):
        if not all(type(v)is int for v in (start,end,axis,step)):
            raise ValueError('VALUE_NOT_STATIC')
        if not -rank<=axis<rank or step==0:raise ValueError('ONNX_SEMANTIC_CONFLICT')
        axis%=rank
        if axis in normalized:raise ValueError('ONNX_SEMANTIC_CONFLICT')
        normalized.append(axis);selection[axis]=slice(start,end,step)
    result=list(shape)
    for axis in normalized:
        d=shape[axis]
        result[axis]=len(range(*selection[axis].indices(d))) if type(d)is int and d>=0 else None
    return result,tuple(selection)

def evaluate(node, ir, values, opset):
    name=node['outputs'][0];op=node['op_type'];attrs=node['attributes']
    deps=[values.get(n,TinyValueFact(n,reason_code='VALUE_NOT_STATIC')) for n in node['inputs'] if n]
    def out(dt=None,shape=None,vs=None,reason=None):return tiny(name,dt,shape,vs,node,deps,reason)
    if not schema_ok(node,opset):return out(reason='UNSUPPORTED_OPERATOR_VERSION')
    if op=='Constant':
        c=ir.small_constants.get(name,{})
        return out(c.get('dtype'),c.get('shape'),c.get('values') if c.get('status')=='KNOWN' else None,c.get('reason_code','VALUE_NOT_STATIC') if c.get('status')!='KNOWN' else None)
    if op=='Shape':
        s=ir.tensors.get(node['inputs'][0],{}).get('shape')
        if s is None:return out(reason='MISSING_TENSOR_METADATA')
        if len(s)>64:return out(reason='CONSTANT_BUDGET_EXCEEDED')
        start=attrs.get('start',0);end=attrs.get('end',len(s));s=s[slice(start,end)]
        return out('int64',[len(s)],[d if type(d)is int else None for d in s])
    if not deps or any(d.values is None for d in deps):return out(reason=next((d.reason_code for d in deps if d.values is None),'VALUE_NOT_STATIC'))
    dt=deps[0].onnx_dtype
    arrays=[np.array(d.values,dtype=object).reshape(d.value_shape) for d in deps]
    try:
        if op in ('Add','Sub','Mul','Div'):
            if len(deps)!=2 or deps[1].onnx_dtype!=dt:return out(reason='ONNX_SEMANTIC_CONFLICT')
            shape=np.broadcast_shapes(*[d.value_shape for d in deps])
            if prod(shape)>64:return out(reason='CONSTANT_BUDGET_EXCEEDED')
            aa,bb=np.broadcast_arrays(*arrays);vs=[]
            for x,y in zip(aa.flat,bb.flat):
                if x is None or y is None:vs.append(None);continue
                if op=='Div':
                    if y==0:return out(reason='DIVISION_BY_ZERO')
                    # ONNX signed integer division truncates towards zero; never float.
                    v=(abs(x)//abs(y))*(-1 if (x<0)!=(y<0) else 1)
                else:v={'Add':lambda:x+y,'Sub':lambda:x-y,'Mul':lambda:x*y}[op]()
                vs.append(int(v))
        elif op=='Gather':
            axis=attrs.get('axis',0);rank=arrays[0].ndim
            if not -rank<=axis<rank or None in deps[1].values:return out(reason='UNSUPPORTED_INDEX_PATTERN')
            if arrays[0].size * arrays[1].size > 64 * max(1, arrays[0].shape[axis]):return out(reason='CONSTANT_BUDGET_EXCEEDED')
            arr=np.asarray(np.take(arrays[0],np.array(deps[1].values,dtype=np.int64).reshape(deps[1].value_shape),axis=axis),dtype=object)
            shape=arr.shape;vs=arr.ravel().tolist()
        elif op=='Concat':
            if sum(x.size for x in arrays)>64:return out(reason='CONSTANT_BUDGET_EXCEEDED')
            if any(d.onnx_dtype!=dt for d in deps):return out(reason='ONNX_SEMANTIC_CONFLICT')
            arr=np.concatenate(arrays,axis=attrs['axis']);shape=arr.shape;vs=arr.ravel().tolist()
        elif op in ('Unsqueeze','Squeeze'):
            axes=list(deps[1].values) if opset>=13 and len(deps)>1 else attrs.get('axes')
            arr=arrays[0]
            if axes is None:
                if op=='Unsqueeze':return out(reason='VALUE_NOT_STATIC')
                arr=np.squeeze(arr)
            else:
                rank=arr.ndim+(len(axes) if op=='Unsqueeze' else 0)
                if any(type(x)is not int or not -rank<=x<rank for x in axes) or len({x%rank for x in axes})!=len(axes):return out(reason='ONNX_SEMANTIC_CONFLICT')
                ax=tuple(x%rank for x in axes)
                arr=np.expand_dims(arr,ax) if op=='Unsqueeze' else np.squeeze(arr,axis=ax)
            shape=arr.shape;vs=arr.ravel().tolist()
        elif op=='Cast':
            dt={onnx.TensorProto.INT32:'int32',onnx.TensorProto.INT64:'int64'}.get(attrs['to'])
            if dt=='int32' and deps[0].onnx_dtype=='int64' and None in deps[0].values:return out(reason='INTEGER_OVERFLOW_OR_CAST_LOSS')
            shape=deps[0].value_shape;vs=list(deps[0].values)
        elif op=='Slice':
            params=[list(d.values) for d in deps[1:]]
            _,sel=slice_indices(deps[0].value_shape,*params);arr=arrays[0][sel];shape=arr.shape;vs=arr.ravel().tolist()
        else:return out(reason='UNSUPPORTED_SHAPE_OPERATOR')
        return out(dt,list(shape),vs)
    except (ValueError,IndexError,TypeError,OverflowError) as exc:
        return out(reason=str(exc) if str(exc) in ('VALUE_NOT_STATIC','ONNX_SEMANTIC_CONFLICT') else 'ONNX_SEMANTIC_CONFLICT')

def dependency_cone(ir):
    """Visit unique producer nodes only, with explicit global/depth bounds."""
    nodes={n['id']:n for n in ir.nodes}
    wanted=set();stack=[]
    for n in ir.nodes:
        if n['domain'] not in ('','ai.onnx'):continue
        if n['op_type'] in ('Slice','Reshape','Resize'):
            stack.extend((x,0) for x in (n['inputs'][1:5] if n['op_type']=='Slice' else n['inputs'][1:2] if n['op_type']=='Reshape' else n['inputs'][3:4]) if x)
        if n['op_type']=='Shape':stack.extend((x,0) for x in n['outputs'])
    seen=set();limited=False
    while stack:
        name,depth=stack.pop()
        if name in seen:continue
        if depth>64 or len(seen)>=2048:limited=True;continue
        seen.add(name);p=nodes.get(ir.tensors.get(name,{}).get('producer'))
        if not p:continue
        wanted.add(p['id'])
        # Shape reads metadata, not feature-map data; stop here.
        if p['op_type']!='Shape':stack.extend((x,depth+1) for x in p['inputs'] if x)
    return seen,wanted,limited
