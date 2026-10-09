"""Bounded Resize factors, actual attrs, and independently proved layout."""
from fractions import Fraction
from .operator_facts import rank,set_fact
from .shape_op_extractor import batch_basis

COORDINATES=['half_pixel','pytorch_half_pixel','asymmetric','align_corners','tf_crop_and_resize']

def resize_parameters(ir,node):
    names=node['inputs'];opset=next((x['version'] for x in ir.model['opset_imports'] if x['domain'] in ('','ai.onnx')),None)
    scales_name=names[1] if opset==10 and len(names)>1 else names[2] if len(names)>2 else ''
    sizes_name=names[3] if len(names)>3 else '';roi_name=names[1] if opset!=10 and len(names)>1 else ''
    scale=ir.numeric_constants.get(scales_name,{});sizes=ir.small_constants.get(sizes_name,{})
    roi=ir.numeric_constants.get(roi_name,{})
    x=ir.tensors.get(names[0],{}).get('shape');factors=None;out=None;issues=[]
    ss=scale.get('values') if scale.get('status')=='KNOWN' else None
    sz=sizes.get('values') if sizes.get('status')=='KNOWN' else None
    has_scales=bool(ss);has_sizes=bool(sz)
    if has_scales and has_sizes:issues.append('ONNX_SEMANTIC_CONFLICT: both scales and sizes nonempty')
    elif has_scales:
        if x is not None and scale.get('shape')==[len(x)] and len(ss)==len(x) and all(v>0 for v in ss):
            factors=[Fraction(v) for v in ss]
            out=[(d*f).numerator//(d*f).denominator if type(d)is int and d>=0 else None for d,f in zip(x,factors)]
        else:issues.append('ONNX Resize scale shape/rank/positive factors not proved')
    elif has_sizes:
        if x is not None and sizes.get('shape')==[len(x)] and len(sz)==len(x) and all(type(v)is int and v>0 for v in sz):
            out=list(sz);factors=[Fraction(s,d) if type(d)is int and d>0 else None for s,d in zip(sz,x)]
        else:issues.append('ONNX Resize sizes shape/rank/positive dimensions not proved')
    return dict(scales_tensor=scales_name,sizes_tensor=sizes_name,roi_tensor=roi_name,scales=scale,sizes=sizes,roi=roi,factors=factors,output_shape=out,issues=issues)

def nearest_enlargement(factors):
    if factors is None or len(factors)!=4 or factors[2] is None or factors[3] is None:return None
    h,w=factors[2:]
    if h<=1 and w<=1:return True
    if h<1 or w<1:return None  # mixed enlargement/shrink source not explicit
    def power(v):return v.denominator==1 and v.numerator>0 and v.numerator & (v.numerator-1)==0
    return power(h) and power(w) and h<=w

def extract_resize(ir,node,facts):
    if not facts['input_tensors']:facts['issues'].append('ONNX Resize input missing');return
    x=facts['input_tensors'][0];params=resize_parameters(ir,node);attrs=node['attributes'];r=rank(x)
    mode=attrs.get('mode','nearest');coord=attrs.get('coordinate_transformation_mode','asymmetric' if facts['imported_opset']==10 else 'half_pixel')
    evidence={'input_tensor':x['name'],'shape':x.get('shape'),'dtype':x.get('dtype'),'opset':facts['imported_opset'],'parameters':{k:v for k,v in params.items() if k!='factors'},'exact_factors':[str(v) if v is not None else None for v in params['factors']] if params['factors'] else None,'attributes':attrs}
    set_fact(facts,'resize_input_rank',r,evidence)
    basis=batch_basis(ir,x['name']);layout=False if x.get('layout') not in (None,'NCHW') else True if basis else None
    set_fact(facts,'resize_layout_nchw',layout,{**evidence,'layout_basis':basis or 'LAYOUT_NOT_PROVEN'})
    fs=params['factors'];spatial=None if fs is None or len(fs)!=4 or any(v is None for v in fs[:2]) else fs[0]==1 and fs[1]==1
    set_fact(facts,'resize_spatial_only',spatial,evidence)
    set_fact(facts,'resize_mode',mode,evidence)
    set_fact(facts,'resize_coordinate_mode',coord if facts['imported_opset']==11 else None,{**evidence,'source_opset_scope':'11 only; other toolchain versions unverified'})
    set_fact(facts,'resize_nearest_enlargement_ok',nearest_enlargement(fs),evidence)
    roi_ok=None
    if coord=='tf_crop_and_resize':
        roi=params['roi'];vs=roi.get('values') if roi.get('status')=='KNOWN' else None;s=x.get('shape')
        if vs is not None and len(vs)==8 and roi.get('shape')==[8] and s is not None and len(s)==4:
            unchanged=vs[0]==0 and vs[1]==0 and vs[4]==1 and vs[5]==1
            boundaries=[Fraction(v)*(d-1) for v,d in zip([vs[2],vs[3],vs[6],vs[7]],[s[2],s[3],s[2],s[3]])] if all(type(d)is int and d>0 for d in s[2:]) else None
            roi_ok=False if not unchanged else all(v.denominator==1 for v in boundaries) if boundaries else None
    set_fact(facts,'resize_roi_ok',roi_ok,evidence)
    facts['resize_parameters']={**evidence['parameters'],'exact_factors':evidence['exact_factors']};facts['mode']=mode;facts['coordinate_transformation_mode']=coord;facts['nearest_mode']=attrs.get('nearest_mode','round_prefer_floor') if facts['imported_opset']>=11 else 'opset10 semantics'
    facts['issues'].extend(params['issues']);facts['unverified'].append('Actual Resize compiler conversion, quantized dtype and execution unverified')
