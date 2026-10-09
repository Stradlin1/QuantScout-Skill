from pathlib import Path
import onnx
from onnx import helper as h, TensorProto as T
from rdkx5_doctor.report import analyze,markdown
from rdkx5_doctor.onnx_reader import read_model
from rdkx5_doctor.rules import load_ruleset
from rdkx5_doctor.cli import DEFAULT_RULESET

def report(path):
    registry,manifest=load_ruleset(DEFAULT_RULESET)
    data=analyze(read_model(path),registry,manifest)
    return data,markdown(data)

def test_normal_nodes_do_not_repeat_advice(make_model):
    data,text=report(make_model(branch=True))
    assert '无需修改' not in text
    assert '完全兼容' not in text
    assert all(f'## {n}.' in text for n in range(9))
    assert '算子覆盖' in text and 'Conv' in text and 'Add' in text
    assert 'producer' in text and '已知大小' in text and '峰值' in text
    assert text==markdown(data)
    assert str(Path(data['model']['path']).parent) not in text

def test_violation_cap_and_complete_json(tmp_path):
    nodes=[h.make_node('Sigmoid',['x' if i==0 else f'y{i-1}'],[f'y{i}'],name=f'n{i}') for i in range(51)]
    model=h.make_model(h.make_graph(nodes,'scalar-chain',[h.make_tensor_value_info('x',T.FLOAT,[])],
           [h.make_tensor_value_info('y50',T.FLOAT,[])]),opset_imports=[h.make_opsetid('',11)])
    path=tmp_path/'chain.onnx';onnx.save(model,path)
    data,text=report(path)
    assert data['summary']['all_status_counts']['VIOLATION']==51
    assert len(data['traces'])==51
    assert '其余 1 个违规节点' in text
    assert text.count('### main/node_')==50
    assert len([d for d in data['diagnostics'] if d['status']=='VIOLATION'])==51

def test_markdown_name_injection_is_escaped(make_model):
    data,_=report(make_model(kernel=(32,1),name='evil`](https://evil.example)\x1b[31m'))
    text=markdown(data)
    assert '\x1b' not in text and '](https://evil.example)' not in text
    assert '&#96;' in text and '&#93;' in text

def test_unknown_groups_are_bounded():
    from test_operator_facts import fake_ir
    from rdkx5_doctor.reporting_sections import unknown_groups
    from rdkx5_doctor.rules import check_operator
    reg,_=load_ruleset(DEFAULT_RULESET)
    ir=fake_ir('Gemm',[[1,2],[2,3]])
    d=check_operator(ir,ir.nodes[0],reg)
    copies=[]
    for i in range(23):
        copy={**d,'node_id':f'main/node_{i:06d}'}
        copies.append(copy)
    groups=unknown_groups({'diagnostics':copies})
    assert len(groups)==1 and len(groups[0]['node_ids'])==23
