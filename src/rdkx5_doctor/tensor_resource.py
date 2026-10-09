"""Integer arithmetic over logical dense payloads, never runtime allocations."""
from math import prod

WIDTHS = {'float16': 2, 'bfloat16': 2, 'float32': 4, 'float64': 8,
          'int8': 1, 'uint8': 1, 'int16': 2, 'uint16': 2,
          'int32': 4, 'uint32': 4, 'int64': 8, 'uint64': 8, 'bool': 1}
NUMERIC = set(WIDTHS) - {'bool'}
CATEGORIES = {'initializer': 'initializers', 'model_output': 'model_outputs',
              'model_input': 'model_inputs', 'constant': 'constants',
              'intermediate_activation': 'intermediate_activations', 'other_unknown': 'other_unknown'}
LIMITATIONS = [
    'Theoretical logical dense Tensor payload only; not measured runtime/BPU/DDR memory.',
    'Known sums are not peak memory: lifetimes, aliasing, reuse, layout, padding and fusion are unknown.',
    'Hypothetical INT8 raw-payload scenario only; not actual quantization, placement, accuracy or performance.',
    'bool uses an assumed one logical byte representation; packed/string/sparse/container types unsupported.',
    'Each Tensor is counted once; initializer/output overlaps use initializer as primary category.'
]

def static_elements(shape):
    if shape is None:
        return None, 'UNKNOWN_SHAPE', 'Rank or shape unavailable'
    if not isinstance(shape, list):
        return None, 'UNKNOWN_SHAPE', 'Shape is not a dimension list'
    reasons = []
    for i, dim in enumerate(shape):
        if isinstance(dim, str):
            reasons.append(f'dim[{i}] symbolic: {dim}')
        elif dim is None:
            reasons.append(f'dim[{i}] unresolved')
        elif type(dim) is not int or dim < 0:
            reasons.append(f'dim[{i}] malformed or unresolved: {dim}')
    if reasons:
        return None, 'UNKNOWN_SHAPE', '; '.join(reasons)
    return prod(shape), 'KNOWN', None

def category(tensor):
    if tensor.get('is_initializer'):
        return 'initializer'
    if tensor.get('is_graph_output'):
        return 'model_output'
    if tensor.get('is_graph_input'):
        return 'model_input'
    if tensor.get('kind') == 'constant':
        return 'constant'
    if tensor.get('producer'):
        return 'intermediate_activation'
    return 'other_unknown'

def record(tensor):
    elements, shape_status, shape_reason = static_elements(tensor.get('shape'))
    dt = tensor.get('dtype')
    dense = tensor.get('storage_kind', 'dense') == 'dense'
    width = WIDTHS.get(dt) if dense else None
    dtype_reason = None if width is not None else f'Unknown/unsupported dtype or representation: {dt} / {tensor.get("storage_kind", "dense")}'
    reasons = [r for r in (shape_reason, dtype_reason) if r]
    status = shape_status if shape_status != 'KNOWN' else 'UNKNOWN_OR_UNSUPPORTED_DTYPE' if width is None else 'KNOWN'
    raw = elements * width if elements is not None and width is not None else None
    # Floating-point display fields can overflow for hostile metadata; exact B stays authoritative.
    def units(divisor):
        if raw is None:
            return None
        try:
            return raw / divisor
        except OverflowError:
            return None
    consumers = list(dict.fromkeys(tensor.get('consumers', [])))
    overlap = bool(tensor.get('is_initializer') and tensor.get('is_graph_output'))
    return {'name': tensor['name'], 'resource_category': category(tensor), 'shape': tensor.get('shape'),
            'dtype': dt, 'storage_kind': tensor.get('storage_kind', 'dense'),
            'element_count': elements, 'element_size_bytes': width, 'raw_bytes': raw,
            'mib': units(1048576), 'mb': units(1000000),
            'hypothetical_int8_bytes': elements if elements is not None and dense and dt in NUMERIC else None,
            'size_status': status, 'shape_status': shape_status,
            'dtype_status': 'KNOWN' if width is not None else 'UNKNOWN_OR_UNSUPPORTED_DTYPE',
            'unknown_reason': '; '.join(reasons) or None,
            'producer_node_id': tensor.get('producer'), 'consumer_node_ids': consumers,
            'consumer_count': len(consumers), 'shared_by_multiple_consumers': len(consumers) >= 2,
            'is_graph_output': bool(tensor.get('is_graph_output')), 'is_graph_input': bool(tensor.get('is_graph_input')),
            'is_initializer': bool(tensor.get('is_initializer')), 'original_kind': tensor.get('kind'),
            'classification_note': 'Exported initializer: primary initializer category, not counted again as output' if overlap else None}

def analyze_tensor_resources(ir):
    records = [record(t) for t in ir.tensors.values()]
    summary = {}
    for cat, key in CATEGORIES.items():
        members = [t for t in records if t['resource_category'] == cat]
        unknown = sum(t['raw_bytes'] is None for t in members)
        summary[key] = {'tensor_count': len(members), 'known_bytes_sum': sum(t['raw_bytes'] for t in members if t['raw_bytes'] is not None),
                        'unknown_tensor_count': unknown, 'completeness': 'PARTIAL' if unknown else 'COMPLETE'}
    def largest(cat):
        return [t['name'] for t in sorted((t for t in records if t['resource_category'] == cat and t['raw_bytes'] is not None),
                key=lambda t: (-t['raw_bytes'], t['name']))[:10]]
    summary.update(largest_intermediates=largest('intermediate_activation'), largest_outputs=largest('model_output'),
                   multi_consumer_intermediate_count=sum(t['resource_category'] == 'intermediate_activation' and t['consumer_count'] >= 2 for t in records),
                   unknown_tensor_names=[t['name'] for t in records if t['raw_bytes'] is None], limitations=LIMITATIONS)
    return {'schema_version': '1.0', 'units': {'bytes': 'B', 'mib_divisor': 1048576, 'mb_divisor': 1000000},
            'hypothetical_int8_label': 'Hypothetical INT8 raw-payload scenario', 'tensor_records': records, 'summary': summary}
