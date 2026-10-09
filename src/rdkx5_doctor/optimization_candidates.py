"""Five conservative structural patterns. Never modifies an ONNX graph."""
from collections import Counter, defaultdict
from math import prod
from .dtype_utils import dtype
from .tensor_resource import WIDTHS, static_elements

PATTERNS = ['IDENTITY', 'TRANSPOSE_INVERSE_PAIR', 'CAST_SAME_DTYPE', 'RESHAPE_NOOP', 'CONV_BN_FUSION_REVIEW']
SOURCES = {op: f'https://onnx.ai/onnx/operators/onnx__{op}.html' for op in ('Identity','Transpose','Cast','Reshape','BatchNormalization')}
VERIFIED_SCHEMAS = {'Identity': {1,13,14,16,19,21,23}, 'Transpose': {1,13,21,23},
                    'Cast': {6,9,13,19,21,23}, 'Reshape': {5,13,14,19,21,23},
                    'BatchNormalization': {1,6,7,9,14,15}}
VERIFY = ['ONNX checker on a future rewritten copy', 'Compare input/output shapes, dtypes and public output names',
          'ONNX Runtime numerical comparison on representative inputs before any future rewrite; check task metrics where relevant']


def understood_version(ir, node):
    if node['domain'] not in ('', 'ai.onnx'):
        return None, 'Custom operator domain: standard semantics not assumed'
    versions = [v['version'] for v in ir.model.get('opset_imports', []) if v['domain'] in ('', 'ai.onnx')]
    if not versions or len(set(versions)) != 1 or not 1 <= versions[0] <= 23:
        return None, 'Missing, conflicting or unreviewed ONNX opset (supported imported versions: 1..23)'
    import onnx
    try:
        schema = onnx.defs.get_schema(node['op_type'], versions[0], '')
    except Exception:
        return None, 'Operator schema unavailable'
    if schema.since_version not in VERIFIED_SCHEMAS.get(node['op_type'], set()):
        return None, f'Operator schema {schema.since_version} is not reviewed for this detector'
    if set(node['attributes']) - set(schema.attributes):
        return None, 'Unrecognized attributes for the imported operator version'
    return schema.since_version, None


def unary(node):
    return len(node['inputs']) == 1 and bool(node['inputs'][0]) and len(node['outputs']) == 1 and bool(node['outputs'][0])


def permutation(node, rank):
    if rank is None:
        return None, 'Input rank unknown; default/explicit permutation cannot be proven'
    perm = node['attributes'].get('perm', list(reversed(range(rank))))
    if not isinstance(perm, list) or len(perm) != rank or any(type(x) is not int for x in perm) or sorted(perm) != list(range(rank)):
        return None, 'Invalid permutation for known input rank'
    return perm, None


def resolve_reshape(input_shape, target, *, allowzero=0):
    elements, status, reason = static_elements(input_shape)
    if status != 'KNOWN':
        return None, reason
    if allowzero not in (0, 1) or type(allowzero) is not int:
        return None, 'Invalid allowzero'
    if not isinstance(target, list) or len(target) > 64 or any(type(x) is not int or x < -1 for x in target) or target.count(-1) > 1:
        return None, 'Invalid or oversized target shape vector'
    if allowzero == 1 and 0 in target and -1 in target:
        return None, 'allowzero=1 cannot combine a zero and -1'
    resolved = list(target)
    for i, value in enumerate(target):
        if value == 0 and not allowzero:
            if i >= len(input_shape):
                return None, 'Zero-copy axis exceeds input rank'
            resolved[i] = input_shape[i]
    if -1 in resolved:
        known_product = prod(x for x in resolved if x != -1)
        if known_product == 0 or elements % known_product:
            return None, 'Inferred dimension ambiguous or violates element count'
        resolved[resolved.index(-1)] = elements // known_product
    if prod(resolved) != elements:
        return None, 'Target does not preserve input element count'
    return resolved, None


def detect_optimization_candidates(ir, *, constants=None):
    constants = ir.small_constants if constants is None else constants
    nodes = {n['id']:n for n in ir.nodes}
    order = {n['id']:i for i,n in enumerate(ir.nodes)}
    candidates = []
    edges_by_source = defaultdict(list)
    for edge in ir.edges:
        edges_by_source[edge['source']].append(edge)
    seen = set()
    def add(pattern, chain, tensor_names, evidence, *, classification='SEMANTICALLY_REDUNDANT', blockers=None, reason=None):
        ids = [n['id'] for n in chain]
        names = list(dict.fromkeys(name for name in tensor_names if name))
        key = (pattern, tuple(ids), tuple(names))
        if key in seen:
            return
        seen.add(key)
        blockers = list(blockers or [])
        # Public boundaries can never be treated as globally removable interfaces.
        for n in chain:
            for name in n['outputs']:
                t = ir.tensors.get(name,{})
                if t.get('is_graph_output'):
                    blockers.append(f'Public graph output must retain name/interface: {name}')
        if blockers and classification == 'SEMANTICALLY_REDUNDANT':
            classification = 'REVIEW_REQUIRED'
        evidence['connections'] = [dict(e) for nid in ids for e in edges_by_source[nid] if e['target'] in ids]
        evidence['tensor_interfaces'] = [{'name':name, 'is_graph_input':bool(ir.tensors.get(name,{}).get('is_graph_input')),
            'is_graph_output':bool(ir.tensors.get(name,{}).get('is_graph_output')),
            'producer':ir.tensors.get(name,{}).get('producer'), 'consumers':ir.tensors.get(name,{}).get('consumers',[])} for name in names]
        candidate = {'pattern': pattern, 'node_ids': ids, 'node_names': [n['original_name'] for n in chain],
                     'tensor_names': names, 'classification': classification, 'evidence': evidence,
                     'explanation': reason or 'Supported local operator semantics establish an unchanged value/shape transformation.',
                     'conditions': ['Preserve public input/output names, shapes, dtypes and all consumer connections',
                                    'Apply to a separate model copy only after validating all blockers and overlapping candidates'],
                     'blockers': list(dict.fromkeys(blockers)),
                     'benefit_hint': 'Potentially fewer graph operations if retained by the compiler; no measured speed or memory improvement',
                     'verification_needed': VERIFY,
                     'reachable_outputs': None, 'reachability_status': 'ON_DEMAND',
                     'overlap_with': [], 'source_urls': [SOURCES[n['op_type']] for n in chain if n['op_type'] in SOURCES],
                     'limitations': ['No model rewrite performed', 'No compiler fusion, CPU/BPU placement or performance measurement']}
        candidates.append(candidate)

    for node in ir.nodes:
        op = node['op_type']
        if op not in VERIFIED_SCHEMAS or node['domain'] not in ('', 'ai.onnx'):
            continue
        version, version_reason = understood_version(ir, node)
        if op == 'BatchNormalization':
            if len(node['inputs']) != 5 or not all(node['inputs']) or not node['outputs'] or not node['outputs'][0]:
                continue
            mid = ir.tensors.get(node['inputs'][0], {})
            conv = nodes.get(mid.get('producer'), {})
            if conv.get('op_type') != 'Conv' or conv.get('domain') not in ('', 'ai.onnx') or node['inputs'][0] not in conv['outputs']:
                continue
            if version is None:
                add('CONV_BN_FUSION_REVIEW', [conv,node], conv['inputs']+node['inputs']+node['outputs'],
                    {'operator_version':None}, classification='INSUFFICIENT_INFORMATION', blockers=[version_reason])
                continue
            outputs = [x for x in node['outputs'] if x]
            attrs = node['attributes']
            inference = (attrs.get('is_test', 0) == 1 if version <= 6 else
                         len(outputs) == 1 if version in (7,9) else attrs.get('training_mode',0) == 0)
            if not inference or len(outputs) != 1:
                continue
            blockers = []
            if len(mid.get('consumers',[])) != 1:
                blockers.append('Conv output has other consumers: do not fold the whole Conv branch globally')
            if attrs.get('spatial',1) != 1:
                blockers.append('Older non-spatial BN semantics require separate review')
            shape = mid.get('shape')
            channels = shape[1] if shape is not None and len(shape) >= 2 else None
            if type(channels) is not int:
                blockers.append('Conv output channel metadata unknown')
            weight = ir.tensors.get(conv['inputs'][1],{}) if len(conv['inputs']) > 1 else {}
            ws = weight.get('shape')
            if ws is None or not ws or ws[0] != channels:
                blockers.append('Conv weight/output channel metadata not demonstrably consistent')
            if not (weight.get('is_initializer') or weight.get('kind') == 'constant') or weight.get('is_graph_input'):
                blockers.append('Conv weights are dynamic or overridable; fixed-weight folding not established')
            conv_dtype = mid.get('dtype')
            if weight.get('dtype') != conv_dtype or conv_dtype not in ('float16','bfloat16','float32','float64'):
                blockers.append('Conv weight/output precision not verified')
            output_meta = ir.tensors.get(node['outputs'][0],{})
            if output_meta.get('shape') != shape or output_meta.get('dtype') != conv_dtype:
                blockers.append('BN output shape/dtype compatibility not verified')
            if len(conv['inputs']) > 2 and conv['inputs'][2]:
                bias = ir.tensors.get(conv['inputs'][2],{})
                if bias.get('shape') != [channels] or bias.get('dtype') != conv_dtype or bias.get('is_graph_input') or not (bias.get('is_initializer') or bias.get('kind') == 'constant'):
                    blockers.append('Conv bias shape/dtype/fixed-value metadata not verified')
            params = []
            for name in node['inputs'][1:]:
                t = ir.tensors.get(name,{})
                params.append({'name':name, 'shape':t.get('shape'), 'dtype':t.get('dtype'),
                               'is_constant': bool(t.get('is_initializer') or t.get('kind') == 'constant')})
                if type(channels) is not int or t.get('shape') != [channels]:
                    blockers.append(f'BN parameter shape/channel not verified: {name}')
                if not (t.get('is_initializer') or t.get('kind') == 'constant') or t.get('is_graph_input'):
                    blockers.append(f'BN parameter is dynamic/overridable: {name}')
                if t.get('dtype') != mid.get('dtype') or t.get('dtype') not in ('float16','bfloat16','float32','float64'):
                    blockers.append(f'Mixed/unknown parameter precision needs validation: {name}')
            if conv['attributes'].get('group',1) != 1:
                blockers.append('Group convolution folding requires explicit numerical verification')
            add('CONV_BN_FUSION_REVIEW', [conv,node], conv['inputs']+node['inputs']+node['outputs'],
                {'bn_schema_version':version,'inference_mode':True,'conv_output_tensor':mid.get('name',node['inputs'][0]),
                 'conv_output_consumer_count':len(mid.get('consumers',[])), 'channels':channels,
                 'bn_parameters':params,'epsilon':attrs.get('epsilon',1e-5), 'parameter_values_read':False},
                classification='REVIEW_REQUIRED',blockers=blockers,
                reason='An actual Conv output directly enters inference BN; fixed-parameter fusion is a review opportunity, not evidence of compiler fusion.')
            continue
        if op == 'Reshape':
            if len(node['inputs']) != 2 or not all(node['inputs']) or len(node['outputs']) != 1:
                continue
            x = ir.tensors.get(node['inputs'][0],{})
            y = ir.tensors.get(node['outputs'][0],{})
            constant = constants.get(node['inputs'][1],{})
            reason = version_reason
            if version is not None and version < 14 and 'allowzero' in node['attributes']:
                reason = 'allowzero is not defined for this imported operator version'
            if constant.get('status') != 'KNOWN':
                reason = reason or constant.get('reason','Target shape is not a known small constant')
            elif constant.get('shape') != [len(constant.get('values',[]))] or constant.get('dtype') != 'int64':
                reason = reason or 'Reshape target must be a one-dimensional INT64 shape vector'
            resolved = None
            if reason is None:
                resolved, reason = resolve_reshape(x.get('shape'),constant['values'],allowzero=node['attributes'].get('allowzero',0))
            if reason:
                add('RESHAPE_NOOP',[node],node['inputs']+node['outputs'],
                    {'input_shape':x.get('shape'),'target_constant':constant,'resolved_target':None},
                    classification='INSUFFICIENT_INFORMATION',blockers=[reason],reason='Reshape near-match only; no no-op has been proven.')
                continue
            if resolved != x.get('shape'):
                continue
            ys = y.get('shape')
            if ys is not None and (len(ys) != len(resolved) or any(type(v) is int and v != resolved[i] for i,v in enumerate(ys))):
                add('RESHAPE_NOOP',[node],node['inputs']+node['outputs'],{'resolved_target':resolved,'output_shape':ys},
                    classification='INSUFFICIENT_INFORMATION',blockers=['Output metadata conflicts with resolved target'])
                continue
            blockers = ['Output shape metadata unavailable'] if ys is None else []
            add('RESHAPE_NOOP',[node],node['inputs']+node['outputs'],
                {'input_shape':x['shape'],'target_values':constant['values'],'resolved_target':resolved,
                 'allowzero':node['attributes'].get('allowzero',0),'schema_version':version},blockers=blockers)
            continue
        if not unary(node):
            continue
        x = ir.tensors.get(node['inputs'][0],{})
        y = ir.tensors.get(node['outputs'][0],{})
        if op == 'Identity':
            if version is None:
                add('IDENTITY',[node],node['inputs']+node['outputs'],{},classification='INSUFFICIENT_INFORMATION',blockers=[version_reason])
            else:
                blockers = []
                metadata_missing = x.get('shape') is None or x.get('dtype') is None
                if len(y.get('consumers',[])) > 1:
                    blockers.append('Identity output fanout: preserve every consumer when rewiring')
                add('IDENTITY',[node],node['inputs']+node['outputs'],
                    {'value_transformation':'identity','input_tensor':node['inputs'][0],
                     'output_tensor':node['outputs'][0],'output_consumer_count':len(y.get('consumers',[])), 'schema_version':version,
                     'input_metadata_missing':metadata_missing},blockers=blockers)
        elif op == 'Cast':
            to = node['attributes'].get('to')
            try:
                destination = dtype(to) if type(to) is int else None
            except ValueError:
                destination = None
            if version is None or x.get('dtype') not in WIDTHS or destination not in WIDTHS:
                add('CAST_SAME_DTYPE',[node],node['inputs']+node['outputs'],{'input_dtype':x.get('dtype'),'destination_dtype':destination},
                    classification='INSUFFICIENT_INFORMATION',blockers=[version_reason or 'Input/destination dtype is unknown or unsupported'])
            elif destination == x.get('dtype'):
                add('CAST_SAME_DTYPE',[node],node['inputs']+node['outputs'],
                    {'input_dtype':x['dtype'],'destination_dtype':destination,'to':to,'schema_version':version})
        elif op == 'Transpose':
            for consumer in y.get('consumers',[]):
                other = nodes[consumer]
                if other['op_type'] != 'Transpose' or other['domain'] not in ('', 'ai.onnx') or not unary(other) or other['inputs'][0] != node['outputs'][0]:
                    continue
                other_version, other_reason = understood_version(ir,other)
                rank = len(x['shape']) if x.get('shape') is not None else None
                a, a_reason = permutation(node,rank)
                b, b_reason = permutation(other,rank)
                reason = version_reason or other_reason or a_reason or b_reason
                if reason:
                    add('TRANSPOSE_INVERSE_PAIR',[node,other],node['inputs']+node['outputs']+other['outputs'],
                        {'perm_a':a,'perm_b':b},classification='INSUFFICIENT_INFORMATION',blockers=[reason])
                    continue
                composite = [a[b[i]] for i in range(rank)]
                if composite != list(range(rank)):
                    continue
                blockers = []
                if y.get('consumers',[]) != [other['id']]:
                    blockers.append('Intermediate Transpose Tensor has fanout: first node cannot be globally removed')
                add('TRANSPOSE_INVERSE_PAIR',[node,other],node['inputs']+node['outputs']+other['outputs'],
                    {'perm_a':a,'perm_b':b,'composite_perm':composite,'intermediate_consumer_count':len(y.get('consumers',[])),
                     'intermediate_is_graph_output':bool(y.get('is_graph_output'))},blockers=blockers)
    candidates.sort(key=lambda c:(min(order[n] for n in c['node_ids']),c['pattern'],tuple(c['node_ids']),tuple(c['tensor_names'])))
    shared = defaultdict(list)
    for i,c in enumerate(candidates,1):
        c['candidate_id'] = f'OPT-{i:04d}'
        for nid in c['node_ids']:
            shared[nid].append(c['candidate_id'])
    for c in candidates:
        c['overlap_with'] = sorted({cid for nid in c['node_ids'] for cid in shared[nid]} - {c['candidate_id']})
        if c['overlap_with']:
            c['limitations'].append('Overlapping candidates cannot be assumed jointly applicable; benefits are not summed')
    return candidates


def analyze_optimization_candidates(ir):
    candidates = detect_optimization_candidates(ir)
    return {'schema_version':'1.0','candidates':candidates,
            'summary':{'candidate_count':len(candidates), 'counts_by_pattern':dict(Counter(c['pattern'] for c in candidates)),
                       'counts_by_classification':dict(Counter(c['classification'] for c in candidates)),
                       'confirmed_local_redundancy_count':sum(c['classification']=='SEMANTICALLY_REDUNDANT' for c in candidates),
                       'insufficient_information_count':sum(c['classification']=='INSUFFICIENT_INFORMATION' for c in candidates)},
            'limitations':['Structural/semantic candidates are not X5 operator compatibility rules.',
                           'No rewrite, compiler fusion verification or measured performance gain.',
                           'Reachability is computed on demand from saved Tensor edges. Imported opset >23 is unreviewed.']}
