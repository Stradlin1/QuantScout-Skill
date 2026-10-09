"""Reviewed, bounded predicates; YAML cannot supply executable code."""

def broadcast_facts(shapes):
    if len(shapes) != 2 or any(s is None for s in shapes):
        return None, None, None
    a, b = shapes
    rank = max(len(a), len(b))
    a = [1] * (rank - len(a)) + a
    b = [1] * (rank - len(b)) + b
    labels, output = [], []
    uncertain = False
    for x, y in zip(a, b):
        if x == y and x is not None:
            output.append(x); labels.append('NONE')
        elif x == 1:
            output.append(y); labels.append('A')
        elif y == 1:
            output.append(x); labels.append('B')
        elif type(x) is int and type(y) is int and x >= 0 and y >= 0:
            return False, None, None
        else:
            output.append(None); labels.append(None); uncertain = True
    return None if uncertain else True, output, labels

def x5_elementwise_broadcast_mergeable(fields):
    compatible, output, labels = broadcast_facts(fields['input_shapes'])
    if compatible is not True:
        return None
    if len(output) <= 4:
        return True
    # Zero/symbolic axes cannot be used to prove a hardware lowering.
    if any(type(x) is not int or x <= 0 for x in output):
        return None
    groups = []
    for dim, label in zip(output, labels):
        if dim == 1:
            continue
        if not groups or label != groups[-1]:
            groups.append(label)
    return len(groups) <= 4

def at_most_one_fixed_constant_input(fields):
    statuses = fields['fixed_constant_statuses']
    known = statuses.count('FIXED')
    unknown = statuses.count('UNKNOWN')
    if known > 1:
        return False
    if known + unknown <= 1:
        return True
    return None

PREDICATES = {
    'x5_elementwise_broadcast_mergeable': x5_elementwise_broadcast_mergeable,
    'at_most_one_fixed_constant_input': at_most_one_fixed_constant_input,
}

def matmul_broadcast_pattern_supported(fields):
    from .attention_op_extractor import matmul_broadcast_pattern_supported as predicate
    return predicate(fields)
PREDICATES['matmul_broadcast_pattern_supported']=matmul_broadcast_pattern_supported
