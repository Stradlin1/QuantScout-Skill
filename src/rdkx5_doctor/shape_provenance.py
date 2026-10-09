"""Bounded, cycle-safe backwards blocker discovery from real Tensor links."""

def unknown_origins(ir,facts,values,reasons):
    nodes={n['id']:n for n in ir.nodes};results=[]
    for f in facts:
        if f['status'] not in ('PARTIAL','UNKNOWN','CONFLICT'):continue
        stack=[f['tensor_name']];seen=set();blockers=[]
        while stack and len(seen)<64:
            name=stack.pop()
            if name in seen:continue
            seen.add(name);t=ir.tensors.get(name,{});p=nodes.get(t.get('producer'));s=t.get('shape')
            value=values.get(name)
            if s is not None and all(type(x)is int for x in s) and (value is None or value.status=='PROVEN'):continue
            reason='ONNX_SEMANTIC_CONFLICT' if t.get('shape_conflict') else value.reason_code if value and value.status=='UNKNOWN' else reasons.get(name)
            if not p or reason in ('UNSUPPORTED_SHAPE_OPERATOR','UNSUPPORTED_OPERATOR_VERSION','OVERRIDABLE_INITIALIZER','EXTERNAL_DATA_NOT_READ','CONSTANT_BUDGET_EXCEEDED','DIVISION_BY_ZERO','INTEGER_OVERFLOW_OR_CAST_LOSS','ONNX_SEMANTIC_CONFLICT'):
                blockers.append(dict(tensor_name=name,node_id=t.get('producer'),reason_code=reason or 'SYMBOLIC_UPSTREAM_DIM'));continue
            upstream=[n for n in p['inputs'] if n]
            if not upstream:blockers.append(dict(tensor_name=name,node_id=p['id'],reason_code=reason or 'VALUE_NOT_STATIC'))
            else:stack.extend(upstream)
        if not blockers:blockers=[dict(tensor_name=f['tensor_name'],node_id=f['producer_node_id'],reason_code='SHAPE_PROPAGATION_BUDGET_EXCEEDED' if stack else f['reason_code'] or 'SYMBOLIC_UPSTREAM_DIM')]
        results.append(dict(tensor_name=f['tensor_name'],reason_code=blockers[0]['reason_code'],first_blocking_sources=blockers,visited_tensor_names=sorted(seen),truncated=bool(stack),max_trace_nodes=64))
    return results
