"""Reproducible public assets; never claims independent Agent E2E."""
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys

import yaml
from rdkx5_doctor.summary_facts import build_fact_package
from rdkx5_doctor.report import markdown

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / 'examples/demo-report'


def test_public_demo_schema_facts_report_summary_agree():
    data = json.loads((DEMO / 'analysis.json').read_text())
    assert data['schema_version'] == '1.3'
    assert data['model']['path'] == 'examples/demo.onnx'
    assert data['model']['node_count'] == 5
    assert data['model']['sha256'] == hashlib.sha256((ROOT / data['model']['path']).read_bytes()).hexdigest()
    pack = build_fact_package(DEMO / 'analysis.json')
    assert pack == json.loads((DEMO / 'summary_facts.json').read_text())
    assert markdown(data) == (DEMO / 'report.md').read_text()
    text = html.unescape((DEMO / 'summary.md').read_text())
    assert text.count('\n## ') == 4
    for item in ('NO_VIOLATION_FOUND 2', 'VIOLATION 1', 'NEEDS_VERIFICATION 0', 'NOT_COVERED 2', 'main/node_000000', 'X5-CONV2D-KERNEL-H', '实际值 32', '未提供 Profile', '嵌套子图未逐节点展开'):
        assert item in text
    assert not any(word in text for word in ('建议', '优化方案', '原因推测'))


def test_public_regeneration_reproducible_without_model_write():
    paths = [ROOT / 'examples/demo.onnx', DEMO / 'analysis.json', DEMO / 'report.md', DEMO / 'summary.md', DEMO / 'summary_facts.json']
    before = {p: p.read_bytes() for p in paths}
    for _ in range(2):
        run = subprocess.run([sys.executable, 'examples/regenerate_report.py'], cwd=ROOT, capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
    assert all(p.read_bytes() == raw for p, raw in before.items())
    assert json.loads((ROOT / 'examples/legacy-demo-report/analysis.json').read_text())['schema_version'] == '1.0'


def test_ci_offline_matrix_and_permissions():
    # BaseLoader preserves the YAML key 'on' instead of treating it as bool.
    workflow = yaml.load((ROOT/'.github/workflows/ci.yml').read_text(),Loader=yaml.BaseLoader)
    assert set(workflow['on']) == {'push','pull_request'}
    assert workflow['permissions'] == {'contents':'read'}
    job=workflow['jobs']['tests']
    assert job['strategy']['matrix']['python-version'] == ['3.10','3.12']
    commands=[s['run'] for s in job['steps'] if 'run' in s]
    assert 'python -m pytest -q' in commands and 'python -m build' in commands
    assert 'python -m rdkx5_doctor rules validate' in commands
    assert not any(word in '\n'.join(commands).lower() for word in ('docker','hb_mapper','api_key','upload-artifact'))
