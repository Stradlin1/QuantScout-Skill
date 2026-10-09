"""Test a standalone manual reader without networking or installed package dependencies."""
import hashlib
import importlib.util
import json
import re
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANUAL = ROOT / 'references/offline_manual'
spec = importlib.util.spec_from_file_location('offline_manual_query', MANUAL / 'query.py')
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


def altered(tmp_path, transform):
    shutil.copytree(MANUAL, tmp_path / 'manual')
    root = tmp_path / 'manual'
    path = root / 'supported_op_list.md'
    path.write_text(transform(path.read_text()), encoding='utf-8')
    manifest = json.loads((root / 'manifest.json').read_text())
    manifest['files']['supported_op_list.md'] = dict(bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (root / 'manifest.json').write_text(json.dumps(manifest))
    return root


def test_official_snapshot_integrity_and_exact_scope():
    result = reader.query(mode='list')
    assert result['count'] == 159 and len(set(result['operators'])) == 159
    assert result['lookup_status'] == 'CACHED'
    assert result['runtime_placement_verified'] is False
    entry = reader.query('Transpose')['entries'][0]
    assert entry['source_line'] == 456
    assert '任意输入维度' in entry['x5_bpu_constraints']
    assert 'perm：[0, 3, 1, 2]' in entry['cpu_constraints']
    assert entry['raw_row'] == (MANUAL / 'supported_op_list.md').read_text().splitlines()[455]


def test_usage_context_and_conditional_folding_preserved():
    result = reader.query('Shape')
    assert '常量折叠' in result['usage_restrictions']['text']
    assert 'input_batch ≤ 128' in result['usage_restrictions']['text']
    assert 'Caffe' in result['usage_restrictions']['text']
    # Upstream labels Shape as BPU acceleration but qualifies constant folding;
    # preserve the discrepancy instead of inferring direct BPU execution.
    assert result['entries'][0]['execution_label'] == 'BPU加速'
    assert '常量折叠' in result['entries'][0]['x5_bpu_constraints']
    assert 'knowledge_status' not in result and 'supported' not in result


@pytest.mark.parametrize('name', ['transpose', 'MissingOp', 'GridSample'])
def test_missing_exact_label_does_not_claim_unsupported(name):
    result = reader.query(name)
    assert result['query_status'] == 'NO_EXACT_LABEL_IN_SNAPSHOT'
    assert result['entries'] == [] and 'diagnostics' not in result
    assert reader.query('GridSample（PyTorch）')['entries']


@pytest.mark.parametrize('name', ['supported_op_list.md', 'LICENSE_UPSTREAM'])
def test_tampering_fails_instead_of_missing_operator(tmp_path, name):
    shutil.copytree(MANUAL, tmp_path / 'manual')
    (tmp_path / 'manual' / name).write_text('broken')
    with pytest.raises(ValueError, match='integrity'):
        reader.query('Relu', root=tmp_path / 'manual')


@pytest.mark.parametrize('change', [
    lambda s: s.replace('## RDK X5支持的ONNX算子列表', '## RDK X3 ONNX only'),
    lambda s: s.replace('## RDK X5支持的ONNX算子列表', '## RDK X5支持的ONNX算子列表\n## RDK X5支持的ONNX算子列表'),
    lambda s: s.replace('**X5 BPU支持约束**', '**CPU column mislabeled**'),
    lambda s: re.sub(r'\| (Abs|Acos)\s+\|', '| AbsDuplicate |', s),
])
def test_bad_table_never_falls_back_to_other_chip(tmp_path, change):
    root = altered(tmp_path, change)
    with pytest.raises(ValueError):
        reader.query('Relu', root=root)


def test_cli_runs_offline_outside_repository(tmp_path):
    proc = subprocess.run([sys.executable, str(MANUAL / 'query.py'), '--operator', 'Relu', '--json'],
                          cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result['query_status'] == 'EXACT_ENTRY_FOUND'
    assert result['entries'][0]['operator_label'] == 'Relu'


def test_broken_cache_cli_reports_unavailable(tmp_path):
    shutil.copytree(MANUAL, tmp_path / 'manual')
    (tmp_path / 'manual/supported_op_list.md').unlink()
    proc = subprocess.run([sys.executable, str(tmp_path / 'manual/query.py'), '--list', '--json'],
                          capture_output=True, text=True)
    assert proc.returncode == 2 and not proc.stdout
    assert json.loads(proc.stderr)['query_status'] == 'SOURCE_UNAVAILABLE'
