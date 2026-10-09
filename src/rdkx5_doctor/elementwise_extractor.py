"""Add/Mul share shape semantics; X5 lowering is a separate predicate."""
from .operator_facts import rank,set_fact
from .rule_predicates import broadcast_facts, at_most_one_fixed_constant_input, x5_elementwise_broadcast_mergeable

def extract_elementwise(ir,node,facts,node_index=None):
    inputs,outputs=facts['input_tensors'],facts['output_tensors']
    if len(inputs)!=2 or len(outputs)!=1:
        facts['issues'].append('ONNX elementwise 输入输出数量不一致');return
    ranks=[rank(t) for t in inputs]
    shapes=[t.get('shape') for t in inputs]
    statuses=[t['fixed_constant_status'] for t in inputs]
    facts['fields'].update(input_shapes=shapes,fixed_constant_statuses=statuses)
    evidence={'input_tensors':[t['name'] for t in inputs],'shapes':shapes,'ranks':ranks}
    known_ranks=[r for r in ranks if r is not None]
    minimum=min(ranks) if None not in ranks else (0 if 0 in ranks else None)
    maximum=max(ranks) if None not in ranks else (max(known_ranks) if known_ranks and max(known_ranks)>10 else None)
    for field,value in [('min_input_rank',minimum),('max_input_rank',maximum)]:set_fact(facts,field,value,evidence)
    set_fact(facts,'output_rank',rank(outputs[0]),{'tensor':outputs[0]['name'],'shape':outputs[0].get('shape')})
    compatible,output,labels=broadcast_facts(shapes)
    facts['onnx_broadcast']={'compatible':compatible,'derived_shape':output,'axis_classes':labels}
    if compatible is False:facts['issues'].append('ONNX_BROADCAST_INVALID：尾维不相等且都不是 1；不是 BPU 硬约束违规')
    elif compatible is None:facts['issues'].append('ONNX_BROADCAST_UNKNOWN：符号轴或缺失 shape 无法证明广播关系')
    set_fact(facts,'constant_count_ok',at_most_one_fixed_constant_input(facts['fields']),
             {'inputs':[{k:t.get(k) for k in ('name','producer','is_initializer','is_graph_input','fixed_constant_status','graph_input_override')} for t in inputs]})
    set_fact(facts,'broadcast_mergeable',x5_elementwise_broadcast_mergeable(facts['fields']),
             {**evidence,'axis_classes':labels,'derived_output':output,'policy':'remove output-1 axes; merge adjacent identical broadcast classes; <=4 groups'})
    producers=node_index if node_index is not None else {n['id']:n for n in ir.nodes}
    if node['op_type']=='Add':
        set_fact(facts,'shortcut_review',None,{'producer_node_ids':[t.get('producer') for t in inputs]})
    else:
        for sigmoid_index in (0,1):
            sig=producers.get(inputs[sigmoid_index].get('producer'),{})
            other=inputs[1-sigmoid_index]
            if sig.get('op_type')=='Sigmoid' and sig.get('domain') in ('','ai.onnx') and sig['inputs']==[other['name']]:
                facts['pattern_evidence']={'pattern':'SIGMOID_MUL_SILU_LIKE',
                   'sigmoid_node_id':sig['id'],'mul_node_id':node['id'],'shared_tensor':other['name'],
                   'shared_tensor_producer':other.get('producer'),'compiler_fusion_verified':False}
