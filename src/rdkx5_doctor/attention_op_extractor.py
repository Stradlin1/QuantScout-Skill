"""ONNX semantic checks separate from X5 documented eligibility."""
from .operator_facts import set_fact,rank
from .rule_predicates import broadcast_facts

def all_dims_within(shapes,start,end,lo,hi):
    if any(s is None for s in shapes):return None
    axes=[x for s in shapes for x in s[start:end]]
    if any(type(x)is int and not lo<=x<=hi for x in axes):return False
    return True if all(type(x)is int for x in axes) else None

def matmul_broadcast_pattern_supported(fields):
    shapes=fields['matmul_input_shapes']
    if len(shapes)!=2 or any(s is None or len(s)<2 for s in shapes) or len(shapes[0])!=len(shapes[1]):return None
    a,b=shapes[0][:-2],shapes[1][:-2];ac=[];bc=[]
    for x,y in zip(a,b):
        if x==y and x is not None:ac.append(False);bc.append(False)
        elif type(x)is int and type(y)is int:
            if x==1:ac.append(True);bc.append(False)
            elif y==1:ac.append(False);bc.append(True)
            else:return None  # ONNX illegal; not a hardware broadcast violation
        else:ac.append(None);bc.append(None)
    # Partial concrete evidence can already establish a contradiction.
    if True in ac and False in ac:return False
    if any(bc[i] is True and bc[j] is False for i in range(len(bc)) for j in range(i+1,len(bc))):return False
    if None in ac or None in bc:return None
    return True

def extract_attention(ir,node,facts):
    inputs=facts['input_tensors'];outputs=facts['output_tensors'];attrs=node['attributes']
    if node['op_type']=='MatMul':
        if len(inputs)!=2 or len(outputs)!=1:facts['issues'].append('ONNX MatMul input/output count invalid');return
        shapes=[t.get('shape') for t in inputs];ranks=[rank(t) for t in inputs]
        evidence={'input_tensors':[t['name'] for t in inputs],'input_shapes':shapes,'input_ranks':ranks}
        facts['fields']['matmul_input_shapes']=shapes
        set_fact(facts,'matmul_rank_relation_ok',ranks[0]==ranks[1] if None not in ranks else None,evidence)
        matrix=all_dims_within(shapes,-2,None,1,8192) if all(s is not None and len(s)>=2 for s in shapes) else None
        higher=all_dims_within(shapes,0,-2,1,4096) if all(s is not None and len(s)>=2 for s in shapes) else None
        set_fact(facts,'matmul_matrix_dims_within_limits',matrix,{**evidence,'bounds':[1,8192],'axes':'trailing two'})
        set_fact(facts,'matmul_higher_dims_within_limits',higher,{**evidence,'bounds':[1,4096],'axes':'all leading'})
        set_fact(facts,'matmul_broadcast_pattern_supported',matmul_broadcast_pattern_supported(facts['fields']),
                 {**evidence,'leading_axis_pairs':list(zip(shapes[0][:-2],shapes[1][:-2])) if all(s is not None for s in shapes) else None,'policy':'A uniform broadcast class; B nonbroadcast prefix then broadcast suffix'})
        if all(s is not None and len(s)>=2 for s in shapes):
            k1,k2=shapes[0][-1],shapes[1][-2]
            comp,lead,_=broadcast_facts([s[:-2] for s in shapes]);facts['onnx_matrix_product']={'k_a':k1,'k_b':k2,'leading_compatible':comp,'leading_shape':lead}
            if type(k1)is int and type(k2)is int and k1!=k2 or comp is False:facts['issues'].append('ONNX_SEMANTIC_CONFLICT: MatMul K or leading broadcast incompatible')
        facts['unverified'].append('Compiler layout/allocation and actual BPU placement unknown')
    else:
        if len(inputs)!=1 or len(outputs)!=1:facts['issues'].append('ONNX Softmax input/output count invalid');return
        r=rank(inputs[0]);raw=attrs.get('axis',1 if facts['imported_opset']<13 else -1)
        axis=raw%r if type(raw)is int and r and -r<=raw<r else None
        if r is not None and axis is None:facts['issues'].append('ONNX_SEMANTIC_CONFLICT: Softmax axis invalid')
        ev={'input_tensor':inputs[0]['name'],'shape':inputs[0].get('shape'),'axis_raw':raw,'axis_normalized':axis,'opset':facts['imported_opset']}
        # Source path is reviewed for opset10/11; modern ONNX semantics are
        # shown but do not establish toolchain support for newer versions.
        reviewed=facts['imported_opset'] in (10,11)
        set_fact(facts,'softmax_input_rank',r if reviewed else None,ev)
        set_fact(facts,'softmax_axis',axis if reviewed else None,ev)
        set_fact(facts,'run_on_bpu_verified',None,{'configuration':'not supplied or inspected','runtime_placement':'UNKNOWN'})
        path='UNKNOWN' if not reviewed or r is None or axis is None else 'ELIGIBLE_WITH_COMPILER_CONFIGURATION' if r==4 and axis==3 else 'NOT_ELIGIBLE_UNDER_DOCUMENTED_ONNX_PATH'
        facts.update(axis_raw=raw,axis_normalized=axis,onnx_semantics='suffix flattened to 2D before normalization' if facts['imported_opset']<13 else 'normalization along axis only',static_bpu_path=path,compiler_configuration_verified=False,actual_runtime_placement='UNKNOWN',converter_provenance='standard ONNX domain; PyTorch converter path not established')
