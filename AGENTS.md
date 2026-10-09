# Repository instructions

- Prioritize `docs/RDK_X5_ONNX_Doctor_V1_3_Shape_Inference_And_Attention_BPU_Development_Spec.md`; follow `docs/TERMINAL_V1_SPEC.md` and `docs/RDK_X5_ONNX_Doctor_V1_1_Development_Spec.md`; preserve V1 compatibility. All interaction is terminal based; do not add HTML, web assets, browser or Netron dependencies.
- Use `.venv/bin/python`; bootstrap with `python3 -m venv .venv` and `.venv/bin/python -m pip install -e '.[dev]'`.
- Run CLI --help, rules validate and `.venv/bin/pytest -q` after relevant changes.
- Canonical rules live in `src/rdkx5_doctor/resources/rulesets`; root rulesets is a symlink. Include rules in wheel/sdist; do not maintain duplicate copies.
- Preserve symbolic/unknown dimensions, tensor connectivity, stable graph-path IDs and official source evidence. No eval or executable rules.
- Unknown compiler/quantization conditions stay UNKNOWN. Conv plus Sigmoid/Concat/Slice/Add/Mul/Gemm/MatMul/Softmax/Resize have versioned rules; other operators stay NOT_COVERED. Never infer performance or accuracy from Conv count.
- Do not modify source models or execute Docker/hb_mapper. Use tiny generated fixtures; never commit real large weights.
- Skill user workflow is maintained only in `.agents/skills/rdk-x5-onnx-doctor/SKILL.md`.

- New analysis uses schema 1.3; node queries also read 1.0/1.1/1.2 and resource/candidate queries read 1.1/1.2/1.3. Theoretical bytes are not runtime/BPU/DDR memory. Candidates never rewrite models. Decode only bounded inline integer shape constants.

- Keep generated reports/logs under ignored reports/. Only concise selected validation summaries belong in docs/validations/. Untracking reports must preserve local files.
- Structure changes belong in the original training/export project; never patch ONNX. Preserve source/opset/toolchain version distinctions and node/Tensor evidence.

- Session preference: model detection reports never sync to GitHub. Development review Markdown may sync during development, but final model validation Markdown stays under ignored reports/; this overrides selected-summary guidance for model-specific results. Do not commit or push automatically.

- V1.3 shape inference is bounded, proof-bearing metadata only. Preserve concrete dimensions, original scalar failures, connectivity and public I/O. Do not search for training projects or generate exact training-code patches; architectural directions only.
