"""Load metadata without materializing large weights or overwriting the model."""
from pathlib import Path
from collections import Counter
import hashlib
import onnx
from onnx import AttributeProto, TensorProto, helper
from .graph_ir import GraphIR

class ModelError(ValueError):
    pass

def dtype(value):
    return {TensorProto.FLOAT: 'float32', TensorProto.DOUBLE: 'float64',
            TensorProto.FLOAT16: 'float16'}.get(value, TensorProto.DataType.Name(value).lower())

def tensor_metadata(value):
    t = value.type.tensor_type
    shape = None
    if t.HasField('shape'):
        shape = [d.dim_value if d.HasField('dim_value') else d.dim_param or None for d in t.shape.dim]
    return shape, dtype(t.elem_type) if t.elem_type else None

def attribute_metadata(attr):
    if attr.type == AttributeProto.TENSOR:
        return {'shape': list(attr.t.dims), 'dtype': dtype(attr.t.data_type)}
    if attr.type in (AttributeProto.GRAPH, AttributeProto.GRAPHS):
        return {'subgraph': 'not_covered'}
    if attr.type == AttributeProto.TENSORS:
        return [{'shape': list(t.dims), 'dtype': dtype(t.data_type)} for t in attr.tensors]
    if attr.type == AttributeProto.SPARSE_TENSOR:
        return {'shape': list(attr.sparse_tensor.dims), 'sparse': True}
    val = helper.get_attribute_value(attr)
    if isinstance(val, bytes):
        return val.decode('utf-8', errors='replace')
    if isinstance(val, (tuple, list)):
        return [v.decode('utf-8', errors='replace') if isinstance(v, bytes) else v for v in val]
    if isinstance(val, (str, int, float)):
        return val
    return {'unsupported_attribute_type': attr.type}

def read_model(path):
    path = Path(path).resolve()
    if path.suffix.lower() != '.onnx' or not path.is_file():
        raise ModelError(f'需要可读的 .onnx 文件：{path}')
    try:
        model = onnx.load(path, load_external_data=False)
    except Exception as exc:
        raise ModelError(f'无法读取 ONNX：{exc}') from exc
    warnings = []
    external = [t for t in model.graph.initializer if t.data_location == TensorProto.EXTERNAL]
    missing = []
    for t in external:
        location = next((e.value for e in t.external_data if e.key == 'location'), '')
        candidate = (path.parent / location).resolve()
        # Do not open arbitrary external paths supplied by the model.
        if not location or not candidate.is_relative_to(path.parent) or not candidate.is_file():
            missing.append(t.name)
    try:
        if not missing:
            onnx.checker.check_model(str(path))
        else:
            # Checker validates external-file availability before graph consistency.
            # Validate a metadata-only copy with external tensors exposed as inputs.
            copy = onnx.ModelProto()
            copy.CopyFrom(model)
            input_names = {v.name for v in copy.graph.input}
            kept = []
            for t in copy.graph.initializer:
                if t.data_location == TensorProto.EXTERNAL:
                    if t.name not in input_names:
                        copy.graph.input.append(helper.make_tensor_value_info(t.name, t.data_type, list(t.dims)))
                        input_names.add(t.name)
                else:
                    kept.append(t)
            del copy.graph.initializer[:]
            copy.graph.initializer.extend(kept)
            onnx.checker.check_model(copy)
            warnings.append('外部权重缺失或路径不安全；仅验证元信息结构，未完整校验原模型：' + ', '.join(missing))
    except Exception as exc:
        raise ModelError(f'ONNX 结构校验失败：{exc}') from exc
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    except Exception as exc:
        inferred = model
        warnings.append(f'形状推断失败；保留原始元信息：{exc}')
    tensors = {}
    def ensure(name):
        return tensors.setdefault(name, {'name': name, 'shape': None, 'dtype': None,
            'producer': None, 'consumers': [], 'kind': 'intermediate',
            'is_graph_input': False, 'is_graph_output': False, 'is_initializer': False})
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        shape, dt = tensor_metadata(value)
        ensure(value.name).update(shape=shape, dtype=dt)
    for value in inferred.graph.input:
        ensure(value.name).update(is_graph_input=True, kind='graph_input')
    for value in inferred.graph.output:
        ensure(value.name).update(is_graph_output=True)
    for init in inferred.graph.initializer:
        ensure(init.name).update(shape=list(init.dims), dtype=dtype(init.data_type),
                                 is_initializer=True, kind='initializer')
    nodes = []
    for i, node in enumerate(inferred.graph.node):
        nid = f'main/node_{i:06d}'
        attrs = {a.name: attribute_metadata(a) for a in node.attribute}
        nodes.append({'id': nid, 'original_name': node.name, 'op_type': node.op_type,
            'domain': node.domain, 'inputs': list(node.input), 'outputs': list(node.output), 'attributes': attrs})
        for a in node.attribute:
            if a.type in (AttributeProto.GRAPH, AttributeProto.GRAPHS):
                warnings.append(f'{nid}：嵌套子图解析未覆盖')
        for name in node.input:
            if name and nid not in ensure(name)['consumers']:
                ensure(name)['consumers'].append(nid)
        for name in node.output:
            if name:
                t = ensure(name)
                t['producer'] = nid
                if node.op_type == 'Constant' and node.domain in ('', 'ai.onnx'):
                    t['kind'] = 'constant'
                    meta = attrs.get('value', {})
                    if isinstance(meta, dict) and 'shape' in meta:
                        t.update(shape=meta['shape'], dtype=meta['dtype'])
    edges = []
    for name, tensor in tensors.items():
        if tensor['producer']:
            for consumer in tensor['consumers']:
                edges.append({'source': tensor['producer'], 'target': consumer, 'tensor': name})
        elif not tensor['is_graph_input'] and not tensor['is_initializer']:
            tensor['kind'] = 'unknown'
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else None
    if digest is None:
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
        digest = h.hexdigest()
    meta = {'path': str(path), 'sha256': digest, 'sha256_scope': 'ONNX protobuf only; external weight contents excluded',
        'opset_imports': [{'domain': x.domain, 'version': x.version} for x in model.opset_import],
        'inputs': [tensors[x.name] for x in model.graph.input],
        'outputs': [tensors[x.name] for x in model.graph.output], 'node_count': len(nodes),
        'operator_counts': dict(Counter((n['domain'] + '::' if n['domain'] else '') + n['op_type'] for n in nodes)),
        'initializer_count': len(model.graph.initializer), 'missing_external_weights': missing,
        'validation': 'metadata_only' if missing else 'checker_passed'}
    return GraphIR(meta, nodes, tensors, edges, warnings)
