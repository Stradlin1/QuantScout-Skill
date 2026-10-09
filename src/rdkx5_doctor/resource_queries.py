"""Terminal views over saved facts only, never reopening the ONNX model."""
import json
from .optimization_candidates import PATTERNS
from .text_utils import json_text

KIND_MAP = {'all':None, 'output':'model_output', 'intermediate':'intermediate_activation',
            'input':'model_input', 'initializer':'initializer', 'constant':'constant', 'unknown':'other_unknown'}


def register_commands(sub):
    listing = sub.add_parser('tensors', help='Tensor 理论字节列表/过滤/排序（不是 BPU/DDR 内存）')
    listing.add_argument('--analysis', required=True)
    listing.add_argument('--kind', choices=list(KIND_MAP), default='all')
    listing.add_argument('--sort', choices=['bytes','name'], default='bytes')
    listing.add_argument('--limit', type=nonnegative, default=10, help='最多展示条数，0 表示全部')
    listing.add_argument('--unknown', action='store_true', help='只显示尺寸未知项，独立于 category')
    listing.add_argument('--search', default='', help='Tensor 名称子串，忽略大小写')
    listing.add_argument('--json', action='store_true')
    tensor = sub.add_parser('tensor', help='查看一个 Tensor 的资源事实')
    tensor.add_argument('--analysis', required=True)
    tensor.add_argument('--name', required=True)
    tensor.add_argument('--json', action='store_true')
    candidates = sub.add_parser('candidates', help='列出结构优化候选及信息不足观察')
    candidates.add_argument('--analysis', required=True)
    candidates.add_argument('--pattern', choices=PATTERNS)
    candidates.add_argument('--json', action='store_true')
    candidate = sub.add_parser('candidate', help='候选证据、条件和按需依赖溯源')
    candidate.add_argument('--analysis', required=True)
    candidate.add_argument('--id', required=True)
    candidate.add_argument('--json', action='store_true')


def nonnegative(value):
    import argparse
    try:
        number = int(value)
        if number < 0:
            raise ValueError()
        return number
    except ValueError:
        raise argparse.ArgumentTypeError('limit 必须为非负整数（0 表示全部）')


def section(data, key):
    result = data.get(key)
    if not isinstance(result, dict) or result.get('schema_version') != '1.0':
        raise ValueError(f'报告缺少有效 {key}；请使用 V1.1 重跑 analyze (rerun analyze)')
    list_key = 'tensor_records' if key == 'resource_analysis' else 'candidates'
    if not isinstance(result.get(list_key), list) or not isinstance(result.get('summary'),dict):
        raise ValueError(f'{key} 结构无效；请重跑 analyze (rerun analyze)')
    return result


def run_query(args, data, *, display, node_trace):
    if args.command in ('tensors','tensor'):
        resources = section(data,'resource_analysis')
        records = resources['tensor_records']
        if args.command == 'tensor':
            matches = [t for t in records if t['name'] == args.name]
            if len(matches) != 1:
                raise ValueError(f'未找到唯一 Tensor：{args.name}')
            result = matches[0]
            if args.json:
                print(json_text(result,ensure_ascii=False,indent=2,allow_nan=False))
            else:
                print(json_text(result,ensure_ascii=False,indent=2,allow_nan=False))
                print('Hypothetical INT8 raw-payload scenario；理论原始载荷，不是实际量化或 BPU/DDR 分配。')
            return 0
        cat = KIND_MAP[args.kind]
        filtered = [r for r in records if (cat is None or r['resource_category'] == cat)
                    and (not args.unknown or r['raw_bytes'] is None)
                    and args.search.casefold() in r['name'].casefold()]
        if args.sort == 'bytes':
            filtered.sort(key=lambda r:(r['raw_bytes'] is None, -(r['raw_bytes'] or 0), r['name']))
        else:
            filtered.sort(key=lambda r:r['name'])
        rows = filtered[:args.limit] if args.limit else filtered
        if args.json:
            print(json_text(rows,ensure_ascii=False,indent=2,allow_nan=False))
        else:
            print('Tensor\tcategory\tshape\tdtype\texact B\traw MiB\tconsumers')
            for r in rows:
                mib = f"{r['mib']:.2f}" if r['mib'] is not None else 'unknown'
                print(f"{display(r['name'])}\t{r['resource_category']}\t{display(r['shape'])}\t{display(r['dtype'])}\t{r['raw_bytes'] if r['raw_bytes'] is not None else 'unknown'}\t{mib}\t{r['consumer_count']}")
            unknown = sum(r['raw_bytes'] is None for r in filtered)
            print(f'显示 {len(rows)} / {len(filtered)}；Unknown-size Tensors: {unknown}（本次过滤全集，未因 limit 隐藏未知数）')
            print('未知项：tensors --analysis <analysis.json> --unknown --limit 0；详情：tensor --analysis <analysis.json> --name <名称>')
            print('Note: theoretical logical raw Tensor payload, not actual RDK X5 BPU/DDR allocation.')
        return 0
    candidates = section(data,'optimization_candidates')['candidates']
    if args.command == 'candidates':
        rows = [c for c in candidates if not args.pattern or c['pattern'] == args.pattern]
        if args.json:
            print(json_text(rows,ensure_ascii=False,indent=2,allow_nan=False))
        else:
            print('ID\tpattern\tclassification\tnode IDs\t原始名称')
            for c in rows:
                print(f"{c['candidate_id']}\t{c['pattern']}\t{c['classification']}\t{display(c['node_ids'])}\t{display(c['node_names'])}")
            print(f'匹配 {len(rows)} 个候选/待验证观察；INSUFFICIENT_INFORMATION 未证明冗余。没有改写模型，没有编译器融合或性能测量。')
        return 0
    matches = [c for c in candidates if c['candidate_id'] == args.id]
    if len(matches) != 1:
        raise ValueError(f'未找到唯一候选 ID：{args.id}')
    # At most the nominated candidate's last node is traced, never every candidate.
    result = dict(matches[0])
    node = next(n for n in data['nodes'] if n['id'] == result['node_ids'][-1])
    trace = node_trace(data,node)
    result.update(reachable_outputs=trace['reachable_outputs'], reachability_status='COMPUTED_ON_DEMAND', trace=trace)
    print(json_text(result,ensure_ascii=False,indent=2,allow_nan=False))
    if not args.json:
        print('依赖路径不是性能影响证明；没有改写模型。')
    return 0
