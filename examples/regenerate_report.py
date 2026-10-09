"""Regenerate relocatable checked-in demo facts and terminal reports."""
from pathlib import Path
import json
from generate_demo import generate
from rdkx5_doctor.onnx_reader import read_model
from rdkx5_doctor.rules import load_ruleset
from rdkx5_doctor.cli import DEFAULT_RULESET
from rdkx5_doctor.report import analyze,markdown

root=Path(__file__).resolve().parent
model=root/'demo.onnx'
generate(model)
ir=read_model(model)
ir.model['path']='examples/demo.onnx'  # Relative to the repository for a relocatable example.
rules,manifest=load_ruleset(DEFAULT_RULESET)
data=analyze(ir,rules,manifest)
out=root/'demo-report'
out.mkdir(exist_ok=True)
(out/'analysis.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
(out/'report.md').write_text(markdown(data),encoding='utf-8')
print(out)
