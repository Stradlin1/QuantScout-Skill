from rdkx5_run_validation import root, out, python, run
import json, math, hashlib, collections, platform
import onnx
from onnx import TensorProto

d=json.loads((out/'analysis.json').read_text())
analysis=str(out/'analysis.json')
def query(label, command, *args):
    p=run(label,[python,'-m','rdkx5_doctor',command,'--analysis',analysis,*args,'--json'])
    if p.returncode: raise RuntimeError(label)
    return json.loads(p.stdout)

top=query('tensors_top10','tensors','--kind','intermediate','--sort','bytes','--limit','10')
outputs=query('tensors_outputs','tensors','--kind','output')
unknown=query('tensors_unknown','tensors','--unknown','--limit','0')
query('nodes_all','nodes')
query('nodes_violations','nodes','--status','VIOLATION')
query('nodes_conv','nodes','--search','Conv')
cs=query('candidates_all','candidates')
for pattern in ['IDENTITY','TRANSPOSE_INVERSE_PAIR','CAST_SAME_DTYPE','RESHAPE_NOOP','CONV_BN_FUSION_REVIEW']:
    query('candidates_'+pattern,'candidates','--pattern',pattern)

chosen=set()
for i,t in enumerate(top+outputs):
    query(f'tensor_{i:02d}','tensor','--name',t['name'])
    if t['producer_node_id']:chosen.add(t['producer_node_id'])
    chosen.update(t['consumer_node_ids'])
for c in cs:
    query('candidate_'+c['candidate_id'],'candidate','--id',c['candidate_id'])
    chosen.update(c['node_ids'])
    for name in c['tensor_names']:
        tensor=next(t for t in d['tensors'] if t['name']==name)
        if tensor['producer']:chosen.add(tensor['producer'])
        chosen.update(tensor['consumers'])
# First unknown Slice and its scalar bounds; attention operators; both non-candidate Reshapes.
chosen.update(n['id'] for n in d['nodes'] if n['op_type'] in ('Shape','Gather','Div','Slice','Transpose','MatMul','Softmax','Resize','Reshape','Split'))
for nid in sorted(chosen):
    query('inspect_'+nid.rsplit('/',1)[-1],'inspect','--node',nid)
    query('trace_'+nid.rsplit('/',1)[-1],'trace','--node',nid)

model=onnx.load(d['model']['path'],load_external_data=False)
node_by_id={n['id']:n for n in d['nodes']}
tensors={t['name']:t for t in d['tensors']}
resources=d['resource_analysis']['tensor_records']
assert len(model.graph.node)==len(d['nodes'])==len(d['diagnostics'])
assert len(resources)==len(tensors)==len({t['name'] for t in resources})
expected_edges=[]
for i,n in enumerate(model.graph.node):
    fact=d['nodes'][i]; nid=f'main/node_{i:06d}'
    assert (fact['id'],fact['original_name'],fact['op_type'],fact['domain'],fact['inputs'],fact['outputs'])==(nid,n.name,n.op_type,n.domain,list(n.input),list(n.output))
    for name in n.input:
        if name:
            assert nid in tensors[name]['consumers']
            if tensors[name]['producer']:expected_edges.append((tensors[name]['producer'],nid,name))
    for name in n.output:
        if name:assert tensors[name]['producer']==nid
assert set(expected_edges)=={(e['source'],e['target'],e['tensor']) for e in d['edges']}
widths={'float32':4,'int64':8}
known=0
for r in resources:
    t=tensors[r['name']]
    assert r['shape']==t['shape'] and r['dtype']==t['dtype']
    assert r['consumer_count']==len(set(t['consumers']))
    if r['raw_bytes'] is not None:
        assert all(type(x) is int and x>=0 for x in r['shape'])
        assert r['element_count']==math.prod(r['shape'])
        assert r['element_size_bytes']==widths[r['dtype']]
        assert r['raw_bytes']==math.prod(r['shape'])*widths[r['dtype']]
        assert r['hypothetical_int8_bytes']==math.prod(r['shape'])
        known+=1
    else:
        assert r['unknown_reason'] and r['element_count'] is None

def spec(v):
    t=v.type.tensor_type
    return {'name':v.name,'dtype':onnx.TensorProto.DataType.Name(t.elem_type),
            'shape':[x.dim_value if x.HasField('dim_value') else x.dim_param or None for x in t.shape.dim] if t.HasField('shape') else None}
original=[spec(v) for v in list(model.graph.input)+list(model.graph.output)+list(model.graph.value_info)]
inferred=onnx.shape_inference.infer_shapes(model,strict_mode=True)
propagated=onnx.shape_inference.infer_shapes(model,strict_mode=True,data_prop=True)
def static_count(m):
    rows=[spec(v) for v in list(m.graph.input)+list(m.graph.output)+list(m.graph.value_info)]
    return {'value_info_count':len(m.graph.value_info),'static_metadata_count':sum(v['shape'] is not None and all(type(x) is int for x in v['shape']) for v in rows),'symbolic_metadata_count':sum(v['shape'] is not None and any(type(x) is str for x in v['shape']) for v in rows)}
shape_comparison={n:static_count(m) for n,m in [('original',model),('default_inference',inferred),('data_prop_experiment',propagated)]}
rule_status=collections.Counter(r['status'] for x in d['diagnostics'] for r in x['results'])
rule_counts={rule['id']:dict(collections.Counter(r['status'] for x in d['diagnostics'] for r in x['results'] if r['rule_id']==rule['id'])) for rule in d['ruleset']['rules']}
candidate_review=[]
for c in cs:
    n=node_by_id[c['node_ids'][0]];x=tensors[n['inputs'][0]];target=c['evidence']['target_constant']['values']
    candidate_review.append({'candidate_id':c['candidate_id'],'node_id':n['id'],'input_rank':len(x['shape']),'target_rank':len(target),
                             'rank_change_proves_not_noop':len(x['shape'])!=len(target),'input_shape':x['shape'],'target':target})
digest=hashlib.sha256()
with open(d['model']['path'],'rb') as f:
    for b in iter(lambda:f.read(1048576),b''):digest.update(b)
assert digest.hexdigest()==d['model']['sha256']==json.loads((out/'validation_metadata.json').read_text())['sha256_before']
audit={'python':platform.python_version(),'platform':platform.platform(),'original_metadata':original,
       'shape_inference_comparison':shape_comparison,'external_initializer_count':sum(t.data_location==TensorProto.EXTERNAL for t in model.graph.initializer),
       'custom_nodes':[n['id'] for n in d['nodes'] if n['domain'] not in ('','ai.onnx')],
       'original_symbolic_metadata':[v for v in original if v['shape'] is not None and any(type(x) is str for x in v['shape'])],
       'known_resource_count':known,'unknown_resource_count':len(unknown),'rule_status_counts':dict(rule_status),'rule_counts':rule_counts,
       'candidate_manual_review':candidate_review,'nodes_inspected':sorted(chosen),'connectivity_audit':'PASS',
       'sha256_after':digest.hexdigest(),'unknown_reason_counts':dict(collections.Counter(t['unknown_reason'] for t in unknown))}
(out/'independent_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
print('AUDIT',json.dumps({k:v for k,v in audit.items() if k not in ('original_metadata','unknown_reason_counts','nodes_inspected')},ensure_ascii=False,indent=2))
