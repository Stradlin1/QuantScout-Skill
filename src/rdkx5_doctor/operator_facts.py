"""Standard ONNX facts with explicit provenance, without weight arrays."""
import onnx

VERIFIED_SCHEMAS = {
    'Reshape': {5}, 'Split': {11}, 'MaxPool': {11}, 'AveragePool': {11},
    'MatMul': {9,13}, 'Softmax': {1,11,13}, 'Resize': {10,11,13},
    'Sigmoid': {6, 13}, 'Concat': {4, 11, 13}, 'Slice': {10, 11, 13},
    'Add': {7, 13, 14}, 'Mul': {7, 13, 14}, 'Gemm': {7, 9, 11, 13},
}

def imported_opset(ir):
    versions = [x['version'] for x in ir.model.get('opset_imports', []) if x['domain'] in ('', 'ai.onnx')]
    return versions[0] if len(versions) == 1 and type(versions[0]) is int else None

def fixed_constant_status(tensor, nodes):
    if tensor.get('is_initializer'):
        return 'UNKNOWN' if tensor.get('is_graph_input') else 'FIXED'
    if tensor.get('is_graph_input'):
        return 'FEATURE'
    producer = nodes.get(tensor.get('producer'))
    if producer:
        return 'FIXED' if producer['op_type'] == 'Constant' and producer['domain'] in ('', 'ai.onnx') else 'FEATURE'
    return 'UNKNOWN'

def common_facts(ir, node, node_index=None):
    nodes = node_index if node_index is not None else {n['id']:n for n in ir.nodes}
    inputs = [{**ir.tensors.get(name, {'name':name, 'shape':None, 'dtype':None}),
               'fixed_constant_status':fixed_constant_status(ir.tensors.get(name,{}),nodes),
               'graph_input_override': bool(ir.tensors.get(name,{}).get('is_graph_input') and ir.tensors.get(name,{}).get('is_initializer'))}
              for name in node['inputs'] if name]
    outputs = [ir.tensors.get(name, {'name':name, 'shape':None, 'dtype':None}) for name in node['outputs'] if name]
    version = imported_opset(ir)
    facts = {'node_id':node['id'], 'original_name':node['original_name'], 'operator':node['op_type'],
             'domain':node['domain'], 'imported_opset':version, 'input_tensors':inputs,
             'output_tensors':outputs, 'attributes':node['attributes'], 'fields':{},
             'field_evidence':{}, 'issues':[], 'scope':'standard_onnx', 'schema_version':None,
             'unverified':['实际工具链设备分配、转换和量化精度未验证；int16 能力不是原始 FP32 限制。']}
    if node['domain'] not in ('', 'ai.onnx') or version is None or not 1 <= version <= 23:
        facts.update(scope='not_covered', issues=['标准域或已审查 opset 未匹配'])
        return facts
    schema = onnx.defs.get_schema(node['op_type'], version, '')
    if schema.since_version not in VERIFIED_SCHEMAS[node['op_type']] or set(node['attributes'])-set(schema.attributes):
        facts.update(scope='not_covered', issues=['未审查的 schema 或属性'])
        return facts
    facts['schema_version']=schema.since_version
    return facts

def set_fact(facts, field, value, evidence):
    facts['fields'][field]=value
    facts['field_evidence'][field]=evidence

def rank(tensor):
    shape=tensor.get('shape')
    return len(shape) if isinstance(shape,list) else None

def extract_operator(ir, node, node_index=None):
    from .shape_op_extractor import extract_shape_op
    from .elementwise_extractor import extract_elementwise
    from .gemm_extractor import extract_gemm
    facts=common_facts(ir,node,node_index)
    if facts['scope']=='not_covered':return facts
    if node['op_type'] in ('Sigmoid','Concat','Slice'):extract_shape_op(ir,node,facts)
    elif node['op_type'] in ('Add','Mul'):extract_elementwise(ir,node,facts,node_index)
    elif node['op_type']=='Gemm':extract_gemm(ir,node,facts)
    elif node['op_type'] in ('MatMul','Softmax'):
        from .attention_op_extractor import extract_attention
        extract_attention(ir,node,facts)
    elif node['op_type']=='Resize':
        from .resize_op_extractor import extract_resize
        extract_resize(ir,node,facts)
    elif node['op_type'] in ('Reshape','Split','MaxPool','AveragePool'):
        from .basic_op_extractor import extract_basic
        extract_basic(ir,node,facts)
    return facts
