"""ONNX NCHW semantics; every unavailable scalar stays None."""
from math import ceil

FIELDS = set('input_rank input_n input_c input_h input_w output_rank output_n output_c output_h output_w weight_shape out_channels in_channels_per_group out_channels_per_group kernel_h kernel_w kernel_elements_per_group strides_h strides_w dilation_h dilation_w pads_top pads_left pads_bottom pads_right auto_pad group input_dtype output_dtype has_dilation dilation_h_divisible dilation_w_divisible quantized_output_dtype quantized_subgraph_last shortcut_fusion'.split())

def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)

def extract_conv(node, tensors):
    f = dict.fromkeys(FIELDS)
    issues = []
    attrs = node['attributes']
    def tensor(index, output=False):
        names = node['outputs'] if output else node['inputs']
        return tensors.get(names[index], {}) if len(names) > index else {}
    x, w, y = tensor(0), tensor(1), tensor(0, True)
    xs, ws, ys = x.get('shape'), w.get('shape'), y.get('shape')
    for prefix, shape in [('input', xs), ('output', ys)]:
        f[prefix + '_rank'] = len(shape) if shape is not None else None
        if shape is not None and len(shape) == 4:
            for suffix, value in zip(('n', 'c', 'h', 'w'), shape):
                f[prefix + '_' + suffix] = value
    f.update(weight_shape=ws, group=attrs.get('group', 1), auto_pad=attrs.get('auto_pad', 'NOTSET'),
             input_dtype=x.get('dtype'), output_dtype=y.get('dtype'))
    if node['domain'] not in ('', 'ai.onnx'):
        scope = 'unknown'
        issues.append('非标准 ONNX 域，未应用标准 Conv2D 约束')
    elif (xs is not None and len(xs) != 4) or (ws is not None and len(ws) != 4):
        scope = 'not_applicable'
        issues.append('不是 rank-4 Conv2D，V1 未覆盖')
    elif xs is None or ws is None:
        scope = 'unknown'
        issues.append('输入或权重 rank 未知，无法确认 Conv2D 范围')
    else:
        scope = 'conv2d'
    if ws is not None and len(ws) == 4:
        f['out_channels'], f['in_channels_per_group'] = ws[:2]
        kernel = attrs.get('kernel_shape', ws[2:])
        if list(kernel) != list(ws[2:]):
            issues.append('kernel_shape 与权重维度不一致')
            kernel = [None, None]
        f['kernel_h'], f['kernel_w'] = kernel if len(kernel) == 2 else [None, None]
    elif len(attrs.get('kernel_shape', [])) == 2:
        f['kernel_h'], f['kernel_w'] = attrs['kernel_shape']
    for attr, keys, default in [('strides', ('strides_h', 'strides_w'), [1, 1]),
                                ('dilations', ('dilation_h', 'dilation_w'), [1, 1])]:
        vals = attrs.get(attr, default)
        if len(vals) != 2:
            issues.append(f'{attr} 长度不是 2')
            vals = [None, None]
        f.update(zip(keys, vals))
    pads = attrs.get('pads', [0, 0, 0, 0])
    if len(pads) != 4:
        issues.append('pads 长度不是 4')
        pads = [None] * 4
    mode = f['auto_pad']
    if mode == 'VALID':
        pads = [0] * 4
    elif mode in ('SAME_UPPER', 'SAME_LOWER'):
        pads = [None] * 4
        for i, axis in enumerate(('h', 'w')):
            size, kernel, dilation, stride = (f['input_' + axis], f['kernel_' + axis], f['dilation_' + axis], f['strides_' + axis])
            if all(integer(v) and v > 0 for v in (size, kernel, dilation, stride)):
                total = max(0, (ceil(size / stride) - 1) * stride + (kernel - 1) * dilation + 1 - size)
                low, high = total // 2, total - total // 2
                pads[i], pads[i + 2] = (low, high) if mode == 'SAME_UPPER' else (high, low)
    elif mode != 'NOTSET':
        issues.append('未知 auto_pad 模式')
        pads = [None] * 4
    f.update(zip(('pads_top', 'pads_left', 'pads_bottom', 'pads_right'), pads))
    vals = [f[k] for k in ('in_channels_per_group', 'kernel_h', 'kernel_w')]
    if all(integer(v) for v in vals):
        f['kernel_elements_per_group'] = vals[0] * vals[1] * vals[2]
    g, cin, cpg, cout = f['group'], f['input_c'], f['in_channels_per_group'], f['out_channels']
    if not integer(g) or g <= 0:
        issues.append('group 必须为正整数')
    else:
        if integer(cout):
            if cout % g:
                issues.append('输出通道不能被 group 整除')
            else:
                f['out_channels_per_group'] = cout // g
        if integer(cin) and integer(cpg) and cin != cpg * g:
            issues.append('输入通道与权重每组通道 × group 不一致')
    dh, dw = f['dilation_h'], f['dilation_w']
    if integer(dh) and integer(dw):
        f['has_dilation'] = dh > 1 or dw > 1
    for axis in ('h', 'w'):
        size, dilation = f['input_' + axis], f['dilation_' + axis]
        if integer(size) and integer(dilation) and dilation > 0:
            f['dilation_' + axis + '_divisible'] = size % dilation == 0
    return {'scope': scope, 'fields': f, 'issues': issues}
