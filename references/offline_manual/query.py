"""Read an integrity-checked official manual snapshot; never infer BPU verdicts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
HEADERS = ['**ONNX算子名称**', '**CPU计算/BPU加速**',
           '**X5 BPU支持约束**', '**CPU支持约束**']


def load_snapshot(root=ROOT):
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['schema_version'] != '1.0':
        raise ValueError('Unsupported snapshot manifest schema')
    for name in ('supported_op_list.md', 'LICENSE_UPSTREAM'):
        data = (root / name).read_bytes()
        expected = manifest['files'][name]
        if len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
            raise ValueError(f'Snapshot integrity check failed: {name}')
    lines = (root / 'supported_op_list.md').read_text(encoding='utf-8').splitlines()
    heading = manifest['section']
    # Never fall back to X3, Caffe, Ultra, or the first table matching a name.
    if heading != '## RDK X5支持的ONNX算子列表' or lines.count(heading) != 1:
        raise ValueError('Missing or ambiguous X5 ONNX section')
    start = lines.index(heading)
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith('## ')), len(lines))
    rows = []
    header_seen = False
    for i in range(start + 1, end):
        line = lines[i]
        if not line.startswith('|'):
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        # Two upstream rows have an extra empty trailing cell. Nonempty extras fail.
        while len(cells) > 4 and not cells[-1]:
            cells.pop()
        if len(cells) != 4:
            raise ValueError(f'Ambiguous table columns at source line {i + 1}')
        if not header_seen:
            if cells != HEADERS:
                raise ValueError('Unexpected X5 ONNX column headers')
            header_seen = True
            continue
        if all(c and set(c) <= set('-: ') for c in cells):
            continue
        if not cells[0] or not cells[1]:
            raise ValueError(f'Missing operator or execution label at source line {i + 1}')
        rows.append(dict(operator_label=cells[0], execution_label=cells[1],
                         x5_bpu_constraints=cells[2], cpu_constraints=cells[3],
                         source_line=i + 1, raw_row=line))
    if len(rows) != manifest['expected_operator_rows'] or len({r['operator_label'] for r in rows}) != len(rows):
        raise ValueError('Incomplete or duplicate X5 ONNX table')
    usage_start = lines.index('## 使用限制说明')
    usage_end = next(i for i in range(usage_start + 1, len(lines)) if lines[i].startswith('## '))
    usage = dict(source_start_line=usage_start + 1, source_end_line=usage_end,
                 text='\n'.join(lines[usage_start:usage_end]))
    return manifest, rows, usage


def query(operator=None, mode='operator', root=ROOT):
    manifest, rows, usage = load_snapshot(root)
    result = dict(lookup_status='CACHED', checked_at=datetime.now(timezone.utc).isoformat(),
                  source=manifest, scope='RDK X5 / ONNX table; literal source labels',
                  runtime_placement_verified=False,
                  notes=['This is a pinned upstream source snapshot, not a live website check.',
                         'Raw cells and context require Agent review; no PASS/FAIL or YAML is generated.',
                         'A missing exact label does not prove hardware or toolchain non-support.'])
    if mode == 'list':
        result.update(query_status='LISTED', operators=[r['operator_label'] for r in rows], count=len(rows))
    elif mode == 'verify':
        result.update(query_status='INTEGRITY_VERIFIED', count=len(rows))
    elif mode == 'usage':
        result.update(query_status='USAGE_CONTEXT', usage_restrictions=usage)
    else:
        matches = [r for r in rows if r['operator_label'] == operator]
        result.update(query_status='EXACT_ENTRY_FOUND' if matches else 'NO_EXACT_LABEL_IN_SNAPSHOT',
                      operator=operator, entries=matches, usage_restrictions=usage)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--operator', help='Exact, case-sensitive label in the official X5 ONNX table')
    action.add_argument('--list', action='store_true', help='List all source labels')
    action.add_argument('--usage', action='store_true', help='Read the complete usage restriction context')
    action.add_argument('--verify', action='store_true', help='Verify source/license hashes and table completeness')
    parser.add_argument('--json', action='store_true', help='Output structured JSON')
    args = parser.parse_args(argv)
    mode = 'list' if args.list else 'usage' if args.usage else 'verify' if args.verify else 'operator'
    try:
        result = query(args.operator, mode)
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        # Broken/missing cache is SOURCE_UNAVAILABLE, never a successful missing-op result.
        print(json.dumps(dict(lookup_status='FAILED', query_status='SOURCE_UNAVAILABLE',
                              error=str(exc)), ensure_ascii=False), file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"CACHED | {result['query_status']} | {result['source']['upstream_revision']}")
        if 'operators' in result:
            print('\n'.join(result['operators']))
        for entry in result.get('entries', []):
            print(f"{entry['operator_label']} (source line {entry['source_line']})")
            print(f"Official execution label: {entry['execution_label']}")
            print(f"X5 BPU column: {entry['x5_bpu_constraints']}")
            print(f"CPU column: {entry['cpu_constraints']}")
        if 'usage_restrictions' in result:
            print(result['usage_restrictions']['text'])
        print('Pinned source snapshot; requires version/context review; runtime placement unverified.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
