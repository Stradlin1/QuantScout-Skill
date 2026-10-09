"""Gemm logical matrices are never treated as a compiler's Conv layout."""
from .operator_facts import set_fact

def extract_gemm(ir,node,facts):
    inputs=facts['input_tensors'];attrs=node['attributes']
    logical={k:attrs.get(k,default) for k,default in [('alpha',1.0),('beta',1.0),('transA',0),('transB',0)]}
    logical.update(M=None,K=None,N=None,onnx_matrix_compatible=None,bias_compatible=None)
    if len(inputs)<2:
        facts['issues'].append('ONNX Gemm A/B 缺失')
    else:
        a,b=inputs[0].get('shape'),inputs[1].get('shape')
        if a is not None and b is not None and len(a)==len(b)==2 and logical['transA'] in (0,1) and logical['transB'] in (0,1):
            m,k=a[::-1] if logical['transA'] else a
            kb,n=b[::-1] if logical['transB'] else b
            logical.update(M=m,K=k,N=n)
            known=type(k)is int and type(kb)is int
            compatible=k==kb if known or (type(k)is str and k==kb) else None
            logical['onnx_matrix_compatible']=compatible
            if compatible is False:facts['issues'].append('ONNX_GEMM_K_MISMATCH：矩阵内维不匹配；不是 BPU 违规')
            if len(inputs)>2:
                c=inputs[2].get('shape')
                if c is not None and len(c)<=2:
                    aligned=[1]*(2-len(c))+c
                    checks=[x==1 or (type(x)is int and type(y)is int and x==y) or (type(x)is str and x==y) for x,y in zip(aligned,[m,n])]
                    if all(checks):logical['bias_compatible']=True
                    elif any(type(x)is int and type(y)is int and x not in (1,y) for x,y in zip(aligned,[m,n])):
                        logical['bias_compatible']=False;facts['issues'].append('ONNX_GEMM_BIAS_MISMATCH：单向广播不匹配')
                elif c is not None: facts['issues'].append('ONNX Gemm C rank >2')
        else:
            facts['issues'].append('Gemm 矩阵 rank/属性/元信息未确认')
    facts['gemm']=logical
    facts.update(conversion_target='Conv',conversion_layout_known=False)
    set_fact(facts,'conversion_layout_known',False,
             {'inputs':[{k:t.get(k) for k in ('name','shape','dtype','fixed_constant_status')} for t in inputs],
              'attributes':{k:logical[k] for k in ('alpha','beta','transA','transB')},
              'logical_matrices':logical,'conversion_target':'Conv','conversion_layout_known':False})
