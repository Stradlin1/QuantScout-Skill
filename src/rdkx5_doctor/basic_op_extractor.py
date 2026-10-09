"""Reviewed Opset-11 basic operator metadata; no feature/weight decoding."""
from math import prod
from .operator_facts import rank, set_fact
from .shape_op_extractor import batch_basis

def extract_basic(ir,node,facts):
    op=node['op_type'];attrs=node['attributes'];inputs=facts['input_tensors'];outputs=facts['output_tensors']
    if not inputs or not outputs:
        facts['issues'].append('Missing operator input/output metadata');return
    source=inputs[0];s=source.get('shape');r=rank(source)
    evidence=dict(input_tensor=source['name'],input_shape=s,output_tensors=[t['name'] for t in outputs],attributes=attrs,schema_version=facts['schema_version'])
    def put(field,value,extra=None):set_fact(facts,field,value,{**evidence,**(extra or {})})
    if op=='Reshape':
        put('input_rank',r);put('output_rank',rank(outputs[0]),{'output_shape':outputs[0].get('shape')})
        name=node['inputs'][1] if len(node['inputs'])>1 else ''
        parameter=ir.small_constants.get(name,{'status':'UNKNOWN','reason':'Value not statically established'})
        put('reshape_conversion_verified',None,{'target_tensor':name,'bounded_target_fact':parameter})
        facts['unverified'].append('Reshape runtime lowering/folding is unverified even when ranks and bounded target values are known.')
    elif op=='Split':
        axis=attrs.get('axis',0);axis=axis%r if type(axis)is int and r and -r<=axis<r else None
        basis=batch_basis(ir,source['name']) if r==4 else None
        length=s[axis] if axis is not None else None
        length=length if type(length)is int and length>0 else None
        n=len(outputs);sizes=attrs.get('split')
        if sizes is None and length is not None and n and length%n==0:sizes=[length//n]*n
        valid=isinstance(sizes,list) and len(sizes)==n and all(type(x)is int and x>0 for x in sizes)
        if sizes is not None and (not valid or length is not None and sum(sizes)!=length):
            facts['issues'].append('ONNX Split sizes/output count/sum conflict')
        extra=dict(axis_normalized=axis,input_axis_length=length,split_sizes=sizes,output_count=n,layout_basis=basis)
        put('split_axis_not_batch',axis!=0 if axis is not None and basis else None,{**extra,'batch_axis':0 if basis else None})
        put('split_lengths_divide_input',all(length%x==0 for x in sizes) if valid and length is not None else None,extra)
        put('split_count_divides_input',length%n==0 if length is not None and n else None,extra)
    elif op in ('MaxPool','AveragePool'):
        # These reviewed hardware rows describe H/W. Other ranks remain uncovered;
        # CPU support for 5D is not evidence of the same BPU limits.
        if r is not None and r!=4:
            facts['scope']='not_covered';facts['issues'].append('Only reviewed 2D pooling context');return
        def vector(key,default,positive=True):
            v=attrs.get(key,default)
            return v if isinstance(v,list) and len(v)==len(default) and all(type(x)is int and x>=(1 if positive else 0) for x in v) else None
        kernel=vector('kernel_shape',[None,None]);strides=vector('strides',[1,1]);dilations=vector('dilations',[1,1])
        pads=vector('pads',[0,0,0,0],False) if attrs.get('auto_pad','NOTSET')=='NOTSET' else None
        extra=dict(kernel=kernel,strides=strides,dilations=dilations,pads=pads,auto_pad=attrs.get('auto_pad','NOTSET'),context='ONNX rank-4 N,C,H,W pooling')
        if r is None:kernel=strides=dilations=pads=None
        if op=='MaxPool':
            for field,v in [('pool_kernel_max',kernel),('pool_stride_max',strides),('pool_padding_max',pads)]:put(field,max(v) if v is not None else None,extra)
            put('pool_no_dilation',all(x==1 for x in dilations) if dilations is not None else None,extra)
        else:
            for field,v,index in [('pool_kernel_h',kernel,0),('pool_kernel_w',kernel,1),('pool_stride_h',strides,0),('pool_stride_w',strides,1)]:put(field,v[index] if v is not None else None,extra)
            put('pool_kernel_area',prod(kernel) if kernel is not None else None,extra)
            put('pool_padding_min',min(pads) if pads is not None else None,extra);put('pool_padding_max',max(pads) if pads is not None else None,extra)
        facts['unverified'].append('General BPU bounds, auxiliary outputs, ceil_mode/storage_order and actual placement are not verified by these numeric rules; auto_pad padding is unknown.')
