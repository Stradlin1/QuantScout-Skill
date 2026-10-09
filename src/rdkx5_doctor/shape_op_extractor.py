"""Ranks, axis provenance, and bounded Slice parameter observations."""
from .operator_facts import rank, set_fact

def batch_basis(ir, name):
    """Find explicit NCHW metadata or a standard rank-4 Conv data connection.

    Only rank-preserving paths are followed. Reshape/Transpose/Unsqueeze do
    not justify declaring an arbitrary first axis to be batch.
    """
    nodes={n['id']:n for n in ir.nodes}
    todo=[(name,[])];seen=set()
    while todo:
        current,path=todo.pop()
        if current in seen or len(path)>64:continue
        seen.add(current);t=ir.tensors.get(current,{})
        if rank(t)!=4:continue
        if t.get('layout')=='NCHW':return {'tensor':current,'basis':'explicit NCHW metadata','path':path}
        producer=nodes.get(t.get('producer'),{})
        if producer.get('domain','') in ('','ai.onnx') and producer.get('op_type')=='Conv':
            return {'tensor':current,'basis':'standard ONNX Conv2D output','node_id':producer['id'],'path':path}
        for consumer in t.get('consumers',[]):
            n=nodes.get(consumer,{})
            if n.get('op_type')=='Conv' and n.get('domain') in ('','ai.onnx') and n['inputs'][0]==current:
                return {'tensor':current,'basis':'standard ONNX Conv2D data input','node_id':consumer,'path':path}
        if producer.get('domain','') in ('','ai.onnx') and producer.get('op_type') in ('Sigmoid','Mul','Add','Slice','Concat','Relu','Identity'):
            names=producer['inputs'][:1] if producer['op_type']=='Slice' else producer['inputs']
            todo.extend((source,path+[producer['id']]) for source in names if source)
    return None

def extract_shape_op(ir,node,facts):
    inputs,outputs=facts['input_tensors'],facts['output_tensors']
    if node['op_type']=='Sigmoid':
        if len(inputs)!=1 or len(outputs)!=1:
            facts['issues'].append('Sigmoid 输入输出数量不一致');return
        for field,t in [('input_rank',inputs[0]),('output_rank',outputs[0])]:
            set_fact(facts,field,rank(t),{'tensor':t['name'],'shape':t.get('shape'),'dtype':t.get('dtype')})
    elif node['op_type']=='Concat':
        ranks=[rank(t) for t in inputs]
        r=ranks[0] if ranks and all(x is not None and x==ranks[0] for x in ranks) else None
        axis=node['attributes'].get('axis')
        normalized=axis % r if type(axis)is int and r and -r<=axis<r else None
        if axis is None or (r is not None and normalized is None):facts['issues'].append('ONNX axis 缺失或越界')
        if ranks and len(set(ranks))>1:facts['issues'].append('ONNX Concat 输入 rank 不一致')
        if type(axis)is int and axis<0 and facts['schema_version']<11:
            normalized=None;facts['issues'].append('当前 schema 未审查负轴语义')
        # The BPU row says N, not arbitrary axis zero. Only the documented
        # NCHW rank-4 interpretation is applied; other layouts remain unknown.
        bases=[batch_basis(ir,t['name']) for t in inputs] if r==4 else []
        batch_known=r==4 and bool(bases) and all(bases)
        set_fact(facts,'axis_not_batch',normalized!=0 if normalized is not None and batch_known else None,
                 {'axis_raw':axis,'axis_normalized':normalized,'input_ranks':ranks,
                  'input_tensors':[t['name'] for t in inputs],'batch_axis':0 if batch_known else None,
                  'layout_basis':bases if batch_known else 'N/batch layout not established'})
        facts['axis_raw']=axis;facts['axis_normalized']=normalized
    elif node['op_type']=='Slice':
        params=[]
        labels=['starts','ends','axes','steps']
        values={}
        fixed=True
        for i,label in enumerate(labels,1):
            name=node['inputs'][i] if len(node['inputs'])>i else ''
            if not name and i>=3:
                params.append({'parameter':label,'tensor':None,'default':'range(len(starts))' if i==3 else 'ones(len(starts))'})
                continue
            constant=ir.small_constants.get(name,{'status':'UNKNOWN','values':None,'reason':'Parameter values not decoded'})
            tensor=ir.tensors.get(name,{})
            valid=(constant.get('status')=='KNOWN' and constant.get('dtype') in ('int32','int64')
                   and constant.get('shape')==[len(constant.get('values',[]))])
            fixed=fixed and valid
            params.append({'parameter':label,'tensor':name,'shape':tensor.get('shape'), 'dtype':tensor.get('dtype'),
                           'producer':tensor.get('producer'),'constant':constant})
            if valid:values[label]=constant['values']
        if fixed:
            length=len(values['starts'])
            if len(values['ends'])!=length or any(len(values[k])!=length for k in ('axes','steps') if k in values):
                facts['issues'].append('ONNX Slice 参数长度不一致')
            r=rank(inputs[0])
            axes=values.get('axes',list(range(length)))
            steps=values.get('steps',[1]*length)
            if any(x==0 for x in steps):facts['issues'].append('ONNX Slice step 不能为 0')
            if r is not None:
                if any(not -r<=x<r for x in axes):facts['issues'].append('ONNX Slice axes 越界')
                elif len(set(x%r for x in axes))!=len(axes):facts['issues'].append('ONNX Slice axes 重复')
        set_fact(facts,'slice_parameters_fixed',fixed,
                 {'data_tensor':inputs[0]['name'],'data_shape':inputs[0].get('shape'),
                  'data_dtype':inputs[0].get('dtype'),'parameters':params})
        facts['unverified'].append('Slice 文档无附加数值限制；动态索引/实际转换未由能力陈述证明。')
