# Repository instructions

- Follow `docs/TERMINAL_V1_SPEC.md` and `docs/RDK_X5_ONNX_Doctor_V1_1_Development_Spec.md`; preserve V1 compatibility. All interaction is terminal based; do not add HTML, web assets, browser or Netron dependencies.
- Use `.venv/bin/python`; bootstrap with `python3 -m venv .venv` and `.venv/bin/python -m pip install -e '.[dev]'`.
- Run CLI --help, rules validate and `.venv/bin/pytest -q` after relevant changes.
- Canonical rules live in `src/rdkx5_doctor/resources/rulesets`; root rulesets is a symlink. Include rules in wheel/sdist; do not maintain duplicate copies.
- Preserve symbolic/unknown dimensions, tensor connectivity, stable graph-path IDs and official source evidence. No eval or executable rules.
- Unknown compiler/quantization conditions stay UNKNOWN. Other operators stay NOT_COVERED. Never infer performance or accuracy from Conv count.
- Do not modify source models or execute Docker/hb_mapper. Use tiny generated fixtures; never commit real large weights.
- Skill user workflow is maintained only in `.agents/skills/rdk-x5-onnx-doctor/SKILL.md`.

- New resource/candidate facts use analysis schema 1.1; existing queries also read 1.0. Theoretical bytes are not runtime/BPU/DDR memory. Candidates never rewrite models. Decode only bounded inline integer shape constants.
