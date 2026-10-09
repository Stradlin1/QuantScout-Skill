"""Generated reports stay local; distributable fixtures remain usable."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def test_generated_reports_ignored_and_examples_trackable():
    def ignored(path):
        return subprocess.run(['git', 'check-ignore', '--no-index', '-q', path], cwd=ROOT).returncode == 0
    assert ignored('reports/new-validation/analysis.json')
    assert ignored('reports/new-validation/stdout.log')
    assert ignored('private-model.onnx')
    assert not ignored('examples/demo.onnx')
    assert not ignored('docs/validations/Yolo26_V1_2_Summary.md')
    assert subprocess.check_output(['git', 'ls-files', 'examples/demo.onnx'], cwd=ROOT).strip()
