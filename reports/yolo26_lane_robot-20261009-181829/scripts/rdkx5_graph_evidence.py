from rdkx5_run_validation import out, run, python
import json, collections
import networkx as nx
import onnx
from rdkx5_doctor.small_constants import decode_integer_tensor

d=json.loads((out/'analysis.json').read_text())
before=json.loads((out/'before_fix/analysis.json').read_text())
assert all(d[k]==before[k] for k in d if k!='optimization_candidates')
nodes={n['id']:n for n in d['nodes']}
g=nx.DiGraph()
g.add_nodes_from(nodes)
g.add_edges_from((e['source'],e['target']) for e in d['edges'])
relevant={c['node_ids'][0] for c in before['optimization_candidates']['candidates']}
relevant.update(['main/node_000000','main/node_000002','main/node_000008','main/node_000019','main/node_000022','main/node_000179','main/node_000181','main/node_000182','main/node_000183'])
for nid in sorted(relevant):
    n=nodes[nid]
    for command in ('inspect','trace'):
        p=run(command+'_'+nid.rsplit('/',1)[1],[python,'-m','rdkx5_doctor',command,'--analysis',str(out/'analysis.json'),'--node',nid,'--json'])
        assert p.returncode==0
    if nid in ('main/node_000181','main/node_000182','main/node_000183'):
        p=run('tensor_attention_'+nid.rsplit('/',1)[1],[python,'-m','rdkx5_doctor','tensor','--analysis',str(out/'analysis.json'),'--name',n['outputs'][0],'--json'])
        assert p.returncode==0
paths=[]
for source in ('main/node_000002','main/node_000008'):
    for target in sorted(relevant):
        if source!=target and nx.has_path(g,source,target):
            ids=nx.shortest_path(g,source,target)
            paths.append({'source':source,'target':target,'nodes':ids,
                          'edges':[e for a,b in zip(ids,ids[1:]) for e in d['edges'] if e['source']==a and e['target']==b]})
m=onnx.load(d['model']['path'],load_external_data=False)
constants={}
for i,n in enumerate(m.graph.node):
    if 9<=i<=22 and n.op_type=='Constant':
        for a in n.attribute:
            if a.name=='value':constants[f'main/node_{i:06d}']=decode_integer_tensor(a.t)
unknown=[r for r in d['resource_analysis']['tensor_records'] if r['raw_bytes'] is None]
shared=[r for r in d['resource_analysis']['tensor_records'] if r['resource_category']=='intermediate_activation' and r['consumer_count']>=2]
evidence={'unchanged_sections_after_fix':[k for k in d if k!='optimization_candidates'],
          'selected_shortest_paths':paths,'first_slice_constant_values':constants,
          'unknown_tensor_details':unknown,'multi_consumer_tensor_details':shared,
          'unknowns_by_producer_op':dict(collections.Counter(nodes[r['producer_node_id']]['op_type'] for r in unknown)),
          'known_top10_global_max_unverified_due_to_unknowns':True}
ts={t['name']:t for t in d['tensors']}
evidence['symbolic_conv_input_count']=sum(any(type(v)is str for v in ts[n['inputs'][0]]['shape']) for n in d['nodes'] if n['op_type']=='Conv')
(out/'graph_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
print('graph audit PASS; unknown by producer:',evidence['unknowns_by_producer_op'])
