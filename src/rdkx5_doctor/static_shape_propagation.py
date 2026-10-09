"""In-memory shape overlay, bounded fixed point, no ONNX persistence."""
from copy import deepcopy
from collections import Counter
from math import prod
import onnx
import networkx as nx
from onnx import helper as h
from .dtype_utils import dtype
from .operator_facts import imported_opset
from .tensor_resource import analyze_tensor_resources
from .static_shape_values import LIMITS,TinyValueFact,tiny,evaluate,dependency_cone,schema_ok,slice_indices
from .shape_provenance import unknown_origins

# Metadata-only upstream inference for these reviewed shape transfers. No
# initializer bytes or feature values are passed to the temporary model.
TRANSFER={'Conv','Sigmoid','Relu','Tanh','Identity','Add','Sub','Mul','Div','Concat',
          'Split','Transpose','Flatten','MaxPool','AveragePool','Gemm','MatMul','Softmax'}

class _UnavailableShapeParameter(ValueError):
    """An unproved parameter is not evidence of an ONNX semantic conflict."""

def reshape_shape(source,target,allowzero=False):
    if len(target)>64 or any(type(x)is not int or x < -1 for x in target) or target.count(-1)>1:
        raise ValueError('ONNX_SEMANTIC_CONFLICT')
    result=list(target)
    for i,d in enumerate(result):
        if d==0 and not allowzero:
            if source is None or i>=len(source):raise ValueError('VALUE_NOT_STATIC')
            result[i]=source[i] if type(source[i])is int else None
    if -1 in result:
        i=result.index(-1);others=[x for x in result if x!=-1]
        if any(type(x)is not int for x in (source or [])+others) or source is None:result[i]=None
        else:
            denominator=prod(others);total=prod(source)
            if denominator==0 or total%denominator:raise ValueError('ONNX_SEMANTIC_CONFLICT')
            result[i]=total//denominator
    elif source is not None and all(type(x)is int for x in source+result) and prod(source)!=prod(result):
        raise ValueError('ONNX_SEMANTIC_CONFLICT')
    return result

def transfer(node,ir,values,opset):
    op=node['op_type'];attrs=node['attributes'];names=node['inputs'];outputs=node['outputs']
    shapes=[ir.tensors.get(n,{}).get('shape') for n in names if n]
    if node['domain'] not in ('','ai.onnx') or opset not in (10,11,12,13):return None,'UNSUPPORTED_OPERATOR_VERSION'
    def vector(n):
        f=values.get(n)
        if not f or f.status!='PROVEN' or f.value_shape!=(len(f.values),):raise _UnavailableShapeParameter(f.reason_code if f and f.reason_code else 'VALUE_NOT_STATIC')
        return list(f.values)
    try:
        if op=='Slice' and shapes and shapes[0] is not None:
            starts,ends=vector(names[1]),vector(names[2]);axes=vector(names[3]) if len(names)>3 and names[3] else None;steps=vector(names[4]) if len(names)>4 and names[4] else None
            if not schema_ok(node,opset):return None,'UNSUPPORTED_OPERATOR_VERSION'
            result,_=slice_indices(shapes[0],starts,ends,axes,steps)
            return [result],None
        if op=='Reshape':
            schema=onnx.defs.get_schema(op,opset,'')
            if schema.since_version not in (5,13):return None,'UNSUPPORTED_OPERATOR_VERSION'
            return [reshape_shape(shapes[0],vector(names[1]))],None
        if op in ('Unsqueeze','Squeeze') and shapes and shapes[0] is not None:
            axes=vector(names[1]) if opset>=13 and len(names)>1 else attrs.get('axes');s=shapes[0]
            rank=len(s)+(len(axes) if axes is not None and op=='Unsqueeze' else 0)
            if axes is None:
                if op=='Unsqueeze' or any(type(x)is not int for x in s):return None,'VALUE_NOT_STATIC'
                return [[x for x in s if x!=1]],None
            if not schema_ok(node,opset) or any(type(x)is not int or not -rank<=x<rank for x in axes) or len({x%rank for x in axes})!=len(axes):return None,'ONNX_SEMANTIC_CONFLICT'
            axes={x%rank for x in axes}
            if op=='Unsqueeze':
                it=iter(s);return [[1 if i in axes else next(it) for i in range(rank)]],None
            if any(s[i]!=1 for i in axes):return None,'ONNX_SEMANTIC_CONFLICT'
            return [[x for i,x in enumerate(s) if i not in axes]],None
        if op=='Resize':
            from .resize_op_extractor import resize_parameters
            schema=onnx.defs.get_schema(op,opset,'')
            if set(attrs)-set(schema.attributes):return None,'UNSUPPORTED_OPERATOR_VERSION'
            parameters=resize_parameters(ir,node)
            return ([parameters['output_shape']],None) if parameters['output_shape'] is not None and not parameters['issues'] else (None,'VALUE_NOT_STATIC')
        if op not in TRANSFER:return None,'UNSUPPORTED_SHAPE_OPERATOR'
        if any(s is None or len(s)>64 for s in shapes):return None,'MISSING_TENSOR_METADATA'
        schema=onnx.defs.get_schema(op,opset,'')
        if set(attrs)-set(schema.attributes):return None,'UNSUPPORTED_OPERATOR_VERSION'
        inputs=[]
        for n in dict.fromkeys(names):
            if not n:continue
            t=ir.tensors[n];dt=t.get('dtype')
            typeid=next((i for i in range(1,25) if dtype(i)==dt),None)
            if typeid is None:return None,'MISSING_TENSOR_METADATA'
            inputs.append(h.make_tensor_value_info(n,typeid,t['shape']))
        ns=h.make_node(op,names,outputs,**attrs)
        model=h.make_model(h.make_graph([ns],'metadata_only',inputs,[h.make_empty_tensor_value_info(n) for n in outputs]),opset_imports=[h.make_opsetid('',opset)])
        inferred=onnx.shape_inference.infer_shapes(model,strict_mode=True)
        result=[]
        for t in inferred.graph.output:
            tt=t.type.tensor_type
            result.append([d.dim_value if d.HasField('dim_value') else None for d in tt.shape.dim] if tt.HasField('shape') else None)
        return result,None
    except _UnavailableShapeParameter as exc:
        return None,str(exc)
    except (ValueError,IndexError,KeyError,TypeError,StopIteration,onnx.onnx_cpp2py_export.shape_inference.InferenceError) as exc:
        return None,str(exc) if str(exc) in ('VALUE_NOT_STATIC','ONNX_SEMANTIC_CONFLICT') else 'ONNX_SEMANTIC_CONFLICT'

def refine_shapes(ir):
    """Mutates only the analysis GraphIR's metadata; original facts retained."""
    before=analyze_tensor_resources(ir);snapshot={n:deepcopy(t) for n,t in ir.tensors.items()}
    names,wanted,limited=dependency_cone(ir);values={};opset=imported_opset(ir)
    try:order=list(nx.topological_sort(ir.topology()))
    except nx.NetworkXUnfeasible:
        order=[];limited=True
    byid={n['id']:n for n in ir.nodes};nodes=[byid[i] for i in order[:2048]]
    if len(order)>2048:limited=True
    proofs={};conflicts=[];reasons={};updates=0
    for name in sorted(names):
        t=ir.tensors.get(name,{});c=ir.small_constants.get(name,{})
        if t.get('is_initializer'):
            reason='OVERRIDABLE_INITIALIZER' if t.get('is_graph_input') else ('EXTERNAL_DATA_NOT_READ' if 'External' in str(c.get('reason')) else 'CONSTANT_BUDGET_EXCEEDED' if 'bound' in str(c.get('reason')) else 'VALUE_NOT_STATIC')
            values[name]=tiny(name,c.get('dtype'),c.get('shape'),c.get('values') if c.get('status')=='KNOWN' and not t.get('is_graph_input') else None,reason=None if c.get('status')=='KNOWN' and not t.get('is_graph_input') else reason)
    for passes in range(LIMITS['max_propagation_passes']):
        changed=False
        for node in nodes:
            if any(ir.tensors.get(n,{}).get('shape_conflict') for n in node['inputs'] if n):
                for name in node['outputs']:
                    ir.tensors[name]['shape_conflict']=True;reasons[name]='ONNX_SEMANTIC_CONFLICT'
                    values[name]=TinyValueFact(name,reason_code='ONNX_SEMANTIC_CONFLICT')
                continue
            if node['id'] in wanted:
                value=evaluate(node,ir,values,opset);values[value.tensor_name]=value
                if value.values is not None:shapes=[list(value.value_shape)];reason=None
                else:shapes,reason=transfer(node,ir,values,opset)
            else:shapes,reason=transfer(node,ir,values,opset)
            if shapes is None:
                for name in node['outputs']:reasons[name]=reason
                continue
            for name,derived in zip(node['outputs'],shapes):
                if derived is None or len(derived)>64:continue
                t=ir.tensors[name];old=t.get('shape')
                if old is not None and len(old)!=len(derived):
                    conflict=dict(tensor_name=name,node_id=node['id'],existing=old,derived=derived,reason_code='ONNX_SEMANTIC_CONFLICT')
                    if conflict not in conflicts:conflicts.append(conflict)
                    t['shape_conflict']=True;continue
                final=list(old) if old is not None else [None]*len(derived)
                for axis,d in enumerate(derived):
                    if type(d)is not int or d<0 or d>2**63-1:continue
                    if type(final[axis])is int:
                        if final[axis]!=d:
                            conflict=dict(tensor_name=name,axis=axis,node_id=node['id'],existing=final[axis],derived=d,reason_code='ONNX_SEMANTIC_CONFLICT')
                            if conflict not in conflicts:conflicts.append(conflict)
                            t['shape_conflict']=True
                        continue
                    key=f'{name}::axis{axis}';premises=[{'tensor_name':n,'shape':deepcopy(ir.tensors.get(n,{}).get('shape')),'proof_ids':[f'{n}::axis{i}' for i in range(len(ir.tensors.get(n,{}).get('shape') or [])) if f'{n}::axis{i}' in proofs]} for n in node['inputs'] if n]
                    proof=dict(proof_id=key,tensor_name=name,axis=axis,value=d,status='PROVEN',method='BOUNDED_STATIC_SHAPE_EVALUATION',producer_node_id=node['id'],source_node_ids=[node['id']]+[ir.tensors[n].get('producer') for n in node['inputs'] if n and ir.tensors[n].get('producer')],source_tensor_names=[n for n in node['inputs'] if n],premises=premises,previous_value=final[axis],derivation=f"{node['op_type']} imported schema transfer from recorded input metadata/parameter facts",imported_opset=opset,limits_applied=LIMITS,parameter_facts=[values[n].json() for n in node['inputs'] if n in values])
                    proofs[key]=proof;final[axis]=d;updates+=1;changed=True
                t['shape']=final
        if not changed:break
    else:limited=limited or changed
    # Publish computed integer facts to the same bounded cache used by Slice
    # diagnostics and candidate/resource readers; values never replace graph nodes.
    for name,v in values.items():
        if v.status=='PROVEN':
            ir.small_constants[name]=dict(status='KNOWN',values=list(v.values),shape=list(v.value_shape),dtype=v.onnx_dtype,source='STATIC_DERIVED',reason=None,proof_node_ids=list(v.source_node_ids))
    after=analyze_tensor_resources(ir);b={r['name']:r for r in before['tensor_records']};a={r['name']:r for r in after['tensor_records']}
    facts=[]
    for name,t in ir.tensors.items():
        s=t.get('shape');axes=[]
        for i,x in enumerate(s or []):
            key=f'{name}::axis{i}';origin='STATIC_DERIVED' if key in proofs else 'MODEL_ORIGINAL' if snapshot[name].get('original_shape') is not None and i<len(snapshot[name]['original_shape']) and type(snapshot[name]['original_shape'][i])is int else 'ONNX_INFERRED' if type(x)is int else 'UNKNOWN'
            axes.append(dict(axis=i,value=x,status='CONFLICT' if t.get('shape_conflict') else 'PROVEN' if type(x)is int else 'UNKNOWN',origin=origin,proof_id=key if key in proofs else None))
        status='CONFLICT' if t.get('shape_conflict') else 'UNKNOWN' if s is None else 'PARTIAL' if any(type(x)is not int for x in s) else 'PROVEN'
        facts.append(dict(tensor_name=name,original_shape=snapshot[name].get('original_shape'),original_dtype=snapshot[name].get('original_dtype'),onnx_inferred_shape=snapshot[name].get('shape'),onnx_inferred_dtype=snapshot[name].get('dtype'),final_shape=s,final_dtype=t.get('dtype'),status=status,axes=axes,producer_node_id=t.get('producer'),reason_code=reasons.get(name,'SYMBOLIC_UPSTREAM_DIM') if status in ('PARTIAL','UNKNOWN') else None,payload_became_known=b[name]['raw_bytes'] is None and a[name]['raw_bytes'] is not None))
    origins=unknown_origins(ir,facts,values,reasons)
    summary=dict(before_known_tensor_count=sum(r['raw_bytes'] is not None for r in b.values()),before_unknown_tensor_count=sum(r['raw_bytes'] is None for r in b.values()),after_known_tensor_count=sum(r['raw_bytes'] is not None for r in a.values()),after_unknown_tensor_count=sum(r['raw_bytes'] is None for r in a.values()),newly_proved_tensor_count=sum(f['payload_became_known'] for f in facts),newly_proved_axis_count=len(proofs),conflict_count=len(conflicts),reason_counts=dict(Counter(o['reason_code'] for o in origins)),passes=passes+1,budget_exceeded=limited)
    return dict(schema_version='1.0',method='BOUNDED_STATIC_INFERENCE',summary=summary,facts=facts,proofs=list(proofs.values()),tiny_values=[v.json() for v in values.values()],unknown_origins=origins,conflicts=conflicts,limits=LIMITS,limitations=['Metadata-only refinement; unsupported transfers remain unknown; no feature/weight evaluation.','Symbolic upstream dimensions are not proof of dynamic external input.'])
