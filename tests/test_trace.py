from rdkx5_doctor.onnx_reader import read_model
from rdkx5_doctor.trace import trace_node
from rdkx5_doctor.graph_ir import GraphIR

def test_branch_merge_double_outputs(make_model):
    ir=read_model(make_model(branch=True,kernel=(32,1)))
    t=trace_node(ir,'main/node_000000')
    assert t['reachable_outputs']==['sum','joined']
    assert len(t['representative_paths'])==2 and t['paths_truncated']
    assert t['successors']==['main/node_000001','main/node_000002']
    assert t['representative_paths'][0]['edges'][0]['tensors']==['c']

def test_large_diamond_graph_is_bounded():
    nodes=[{'id':'source'}];edges=[];last='source'
    for i in range(250):
        a,b,c=f'a{i}',f'b{i}',f'c{i}'
        nodes.extend({'id':n} for n in (a,b,c))
        edges.extend({'source':s,'target':t,'tensor':s+'_'+t} for s,t in [(last,a),(last,b),(a,c),(b,c)])
        last=c
    ir=GraphIR({'outputs':[{'name':'out','producer':last}]},nodes,{},edges,[])
    t=trace_node(ir,'source')
    assert len(t['representative_paths'])==1 and t['paths_truncated']
    assert len(t['downstream_nodes'])==750

def test_dead_branch_and_direct_output():
    ir=GraphIR({'outputs':[{'name':'out','producer':'other'}]},[{'id':'source'},{'id':'other'}],{},[],[])
    t=trace_node(ir,'source')
    assert not t['reachable_outputs'] and '死分支' in t['reason']
    ir.model['outputs']=[{'name':'out','producer':'source'}]
    t=trace_node(ir,'source')
    assert t['representative_paths'][0]['nodes']==['source']

def test_parallel_tensor_labels():
    ir=GraphIR({'outputs':[{'name':'out','producer':'b'}]},[{'id':'a'},{'id':'b'}],{},[{'source':'a','target':'b','tensor':'x'},{'source':'a','target':'b','tensor':'y'}],[])
    assert trace_node(ir,'a')['representative_paths'][0]['edges'][0]['tensors']==['x','y']
