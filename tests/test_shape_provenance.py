from rdkx5_doctor.graph_ir import GraphIR
from rdkx5_doctor.static_shape_values import dependency_cone
from rdkx5_doctor.shape_provenance import unknown_origins

def graph(count):
    nodes=[];ts={}
    for i in range(count):
        name=f'v{i}';nid=f'n{i}'
        nodes.append(dict(id=nid,op_type='Add',domain='',inputs=[f'v{i-1}',f'v{i-1}'] if i else [],outputs=[name],attributes={}))
        ts[name]=dict(name=name,shape=[None],producer=nid,consumers=[])
    nodes.append(dict(id='reshape',op_type='Reshape',domain='',inputs=['feature',f'v{count-1}'],outputs=['y'],attributes={}))
    return GraphIR({},nodes,ts,[],[])

def test_diamond_paths_visited_once():
    names,nodes,limited=dependency_cone(graph(30))
    assert len(nodes)==30 and len(names)==30 and not limited

def test_expression_depth_budget():
    names,nodes,limited=dependency_cone(graph(100))
    assert limited and len(names)<=65 and len(nodes)<=65

def test_unknown_trace_budget_and_shared_roots():
    ir=graph(100)
    facts=[dict(tensor_name='v99',status='PARTIAL',producer_node_id='n99',reason_code='SYMBOLIC_UPSTREAM_DIM')]
    records=unknown_origins(ir,facts,{}, {})
    assert records[0]['truncated'] and len(records[0]['visited_tensor_names'])==64
    assert records[0]['reason_code']=='SHAPE_PROPAGATION_BUDGET_EXCEEDED'
