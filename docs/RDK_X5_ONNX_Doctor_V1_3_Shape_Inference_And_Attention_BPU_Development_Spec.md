# RDK X5 ONNX Doctor V1.3 - Codex Implementation Specification

**Track:** bounded static shape inference + MatMul / Softmax / Resize BPU diagnostics

**Repository:** https://github.com/Stradlin1/skillzuoye

**Starting baseline:** main at `dcc7b290` (V1.2, package 0.3.0, analysis schema 1.2, multi-op registry 0.2.0). Check the actual Git HEAD before making changes: a newer commit may exist.

**Primary target model:**

```text
/home/xhm/lianghua_ws/rdktoolchain/horizon_x5_open_explorer_v1.2.8/horizon_x5_open_explorer_v1.2.8-py310_20240926/samples/ai_toolchain/horizon_model_convert_sample/04_detection/15_tasknav/model/yolo26_lane_robot.onnx
```

**Implementation agent:** Codex working in the cloned `skillzuoye` repository on Ubuntu, exclusively through the terminal.

**Deliverable:** actual implemented and tested V1.3, not a plan or pseudocode-only response. Preserve V1, V1.1, and V1.2 behavior unless this document expressly revises it.

---

## 0. Non-negotiable product decisions

1. This project analyzes ONNX; **it NEVER edits, patches, simplifies, replaces, exports, or optimizes the ONNX graph**. Do not add graph surgery, ONNX simplifiers, `onnx.save` output, executable rewrite actions, or automatic optimization pipelines.
2. This repository is NOT the training or export repository. Do not guess its file locations, Python classes, layer names, training flags, or exact patches. Do not search unrelated local directories for the training project.
3. The most that diagnostic advice may suggest is a **high-level architectural direction**. As an illustrative analogy, the user previously maintained `lanerobot` `main` and `quant corrected` branches with a decoupled/simplified output-head approach. Refer to this only as an architectural category, not as evidence of any specific source-code implementation.
4. An acceptable model-level recommendation is: "Review whether the output head can be decoupled into the neural-network outputs and postprocessing; implement in the original training/export project, then re-export and compare outputs." An unacceptable recommendation is: "Edit model/head.py line 120" without access to that source.
5. Primary V1.3 goals: **(A) recover provable intermediate Tensor shapes** and **(B) extend official RDK X5 BPU checks to MatMul, Softmax, and Resize**. No other operator's BPU rule coverage is mandatory in this version.
6. Respect the difference among **ONNX operator legality**, **compliance with a documented static X5 BPU constraint**, **eligibility contingent on compiler options**, and **actual CPU/BPU placement after quantization**. Never conflate them.
7. Terminal-only operation. No GUI, Netron launch, HTML, `graph.html`, frontend, dashboards, or interactive visual controls.
8. Do not run Docker, `hb_mapper`, PTQ, model inference on hardware, RDK deployment, or quantization. Do not estimate FPS, inference accuracy, SRAM/DDR allocation, latency, or final compiler placement.
9. Never fabricate source-backed rules. Use only the **RDK X5 ONNX BPU constraint column**, not X3, RDK Ultra, Caffe, CPU-only, or another chip/version. Source-version and toolchain-version uncertainty remain explicit.
10. Keep the existing official Conv rules byte-for-byte unchanged. V1.2 Add/Mul/Sigmoid/Concat/Slice/Gemm verdicts must be reproducible unless an evidence-backed shape refinement changes an UNKNOWN into a better-founded result. In particular, do not silently erase the five rank-zero Mul findings.
11. Model diagnosis artifacts belong in ignored `reports/`. Do not upload or commit the test model, its complete JSON, command logs, or model-specific validation Markdown. Do not run `git commit` or `git push` without the user's explicit instruction.
12. Treat any unsupported/unknown shape or configuration as **UNKNOWN / NEEDS_VERIFICATION**, rather than making up a plausible dimension or claiming BPU support.

## 1. Why V1.3 is needed: evidence from V1.2 real-model validation

The user's local `REAL_ONNX_V1_2_VALIDATION.md` establishes the baseline below. It is the reference measurement, not a target number to force through implementation.

| Item | Verified V1.2 reference |
| --- | --- |
| Model opset | ONNX 11 |
| Nodes | 281, across 23 op types |
| Already within registered rule scope | 181 nodes, across 7 op types |
| Not registered | 100 nodes |
| Tensor records | 403 |
| Known sizes | 231 |
| Unknown sizes | 172, predominantly propagated symbolic shapes |
| Static rule failures | 5 Mul nodes with rank-zero scalar constants |
| MatMul | 2 nodes; currently not BPU-checked |
| Softmax | 1 node; currently not BPU-checked |
| Resize | 1 node; currently not BPU-checked |
| V1.2 tests | 234 passed according to the user's local validation report |

The earliest non-provable feature shapes were traced to a chain resembling:

```text
Known-shape feature Tensor
    -> Shape
    -> Gather / integer Add / Div / Mul
    -> computed Slice ends
    -> Slice of feature Tensor
    -> symbolic feature shape
    -> downstream Concat / Conv / Add / task branches
```

The original ONNX graph has fixed external input `images: float32[1,3,640,640]`. Some inferred `unk__...` dimensions are **limitations of ONNX shape inference**, not proof that the user exported a dynamic-input model. In-memory ONNX `data_prop=True` was already tried in V1.2 and did not resolve the 172 unknown Tensor sizes.

The five rank-zero Mul failures are located at `main/node_000181` (attention scaling) and `main/node_000228`, `000242`, `000256`, `000270` (four task branches). They are documented **original graph versus X5 static BPU requirement conflicts**, not proof of compilation failure: a compiler may fold or normalize a scalar constant. Do not suppress them in this version.

Known attention structure from previous analysis:

```text
main/node_000178 Transpose
 -> 000179 MatMul
 -> 000181 Mul
 -> 000182 Softmax
 -> 000183 Transpose
 -> 000184 MatMul
```

This is an actual dependency chain, NOT a removable inverse-Transpose pair. Run fresh queries to confirm the node IDs against the current source model; never assume the graph remains identical if the source ONNX changes.

### Acceptance philosophy

- An unknown-size count below 172 is useful **only if each new concrete dimension has a machine-verifiable proof**.
- A reduction to exactly zero is NOT a requirement; leaving irreducible unknowns is correct.
- No new V1.3 rule is considered reliable until both positive and negative boundary tests exist.
- After adding MatMul (2), Softmax (1), and Resize (1), up to four additional real-model nodes enter the rule registry. `185/281` would indicate registered scope, **not 185 verified BPU nodes**.

## 2. Mandatory repository survey before editing

Read, in this order:

```text
AGENTS.md
README.md
pyproject.toml
.agents/skills/rdk-x5-onnx-doctor/SKILL.md
docs/DEVELOPMENT_LOG.md
docs/ANALYSIS_SCHEMA_V1_2.md
docs/REPORT_POLICY_V1_2.md
references/x5_multiop_sources.md
references/x5_multiop_rule_review.md
src/rdkx5_doctor/onnx_reader.py
src/rdkx5_doctor/graph_ir.py
src/rdkx5_doctor/small_constants.py
src/rdkx5_doctor/tensor_resource.py
src/rdkx5_doctor/operator_facts.py
src/rdkx5_doctor/shape_op_extractor.py
src/rdkx5_doctor/elementwise_extractor.py
src/rdkx5_doctor/rules.py
src/rdkx5_doctor/report.py
src/rdkx5_doctor/reporting_sections.py
src/rdkx5_doctor/cli.py
src/rdkx5_doctor/resource_queries.py
src/rdkx5_doctor/resources/rulesets/x5-bayes-e/manifest.yaml
```

Discover any other existing test/helper files before introducing new modules. Do not recreate a component that already exists. In the current baseline, `GraphIR` holds nodes, Tensor dictionaries, edges, warnings, and bounded small constants; `report.analyze` handles registered diagnostics and calls Tensor resource/candidate analysis; `reporting_sections.py` renders the Markdown report.

Baseline commands (run with the repository's environment, not system Python):

```bash
pwd
git status --short
git rev-parse HEAD
.venv/bin/python -m rdkx5_doctor --help
.venv/bin/python -m rdkx5_doctor rules validate
env -u PYTHONPATH .venv/bin/python -m pytest -q
```

The WSL environment previously had ROS `PYTHONPATH` contamination. Isolate that variable for pytest; do not skip, delete, or rewrite project tests to hide such a problem.

Create a fresh ignored baseline report directory before changing code. Preserve the baseline `analysis.json` for exact comparisons after implementation. Record source ONNX SHA256 before and after each real-model validation.

## 3. Proposed architecture and module boundaries

Prefer the following additions or nearby alternatives supported by the actual code layout:

```text
src/rdkx5_doctor/
    static_shape_values.py           # bounded, typed, evidence-carrying tiny integer value facts
    static_shape_propagation.py      # shape-only dependency evaluation, local refinement
    shape_provenance.py              # per-dimension proof and unknown origin traces
    attention_op_extractor.py        # MatMul and Softmax static facts
    resize_op_extractor.py           # Resize static facts and bounded scale/sizes facts
    small_numeric_constants.py      # optional bounded inline FP scales/ROI decoder
    ... existing onnx_reader, graph_ir, operator_facts, rules, report, CLI

src/rdkx5_doctor/resources/rulesets/x5-bayes-e/
    matmul.yaml
    softmax.yaml
    resize.yaml
    manifest.yaml                    # add registered op entries, version and sources

tests/
    test_static_shape_values.py
    test_static_shape_propagation.py
    test_shape_provenance.py
    test_matmul_rules.py
    test_softmax_rules.py
    test_resize_rules.py
    test_v1_3_cli_and_report.py
    test_v1_3_real_model_contract.py

docs/
    ANALYSIS_SCHEMA_V1_3.md
    REPORT_POLICY_V1_3.md
    RDK_X5_ONNX_Doctor_V1_3_Development_Log.md
references/
    x5_attention_resize_sources.md
    x5_attention_resize_rule_review.md
```

Use fewer files when a simpler layout fits, but keep these **responsibilities separate**:

1. ONNX decoding and graph construction.
2. Deterministic, bounded shape/value fact propagation.
3. Hardware rule extraction/validation using YAML.
4. User-facing explanations and optional drill-down.

Do not solve shape unknowns by rewriting/saving a new ONNX. It is acceptable to build a transient metadata-only representation **in memory** as an analysis aid, but it must never be persisted as a modified ONNX and must never change the original graph's semantics or weights.

### 3.1 Evidence-bearing facts

For each inferred fact record at least:

```json
{
  "tensor_name": "example/sliced_output",
  "axis": 1,
  "value": 32,
  "status": "PROVEN",
  "method": "BOUNDED_STATIC_SHAPE_EVALUATION",
  "producer_node_id": "main/node_000019",
  "source_node_ids": ["main/node_000009", "main/node_000019"],
  "source_tensor_names": ["example/shape", "example/slice_ends"],
  "derivation": "Slice axis=1, start=0, end=32, step=1 on an input whose axis 1 is 64",
  "imported_opset": 11,
  "limits_applied": {"max_elements": 64, "max_encoded_bytes": 4096}
}
```

The above is an **illustrative schema**, not evidence that any such node exists. A derived dimension may be emitted as PROVEN only when the actual ONNX tensors support every premise. Unknown, partial, incompatible, or unreviewed cases must explicitly retain their reason and dependencies.

Use stable internal node IDs and real Tensor names in the actual output. Do not rely on node order alone for data connectivity.

### 3.2 Proof precedence

Suggested metadata sources:

- `MODEL_ORIGINAL`: model input/output/initializer/value_info fact from original protobuf.
- `ONNX_INFERRED`: fact inferred by the upstream ONNX shape inference engine.
- `STATIC_DERIVED`: fact proved by this tool's constrained semantics.
- `UNKNOWN`: absent or symbolic without a proof.
- `CONFLICT`: two definite facts disagree; keep both in evidence, do not silently overwrite one.

A symbolic placeholder such as `unk__42` is not a concrete dimension and may be replaced by a strictly proved integer, preserving the previous symbol in the provenance record. If two known integer dimensions disagree, report CONFLICT/NEEDS_VERIFICATION rather than choosing the smaller, larger, or more convenient one.

Do not declare a Tensor truly dynamic merely because upstream shape inference emitted `unk__*`.

## 4. Bounded static shape/value evaluator - required first

### 4.1 Scope: small integer metadata, not feature inference

Run bounded evaluation only on the dependency cone leading to **shape-parameter consumers** (e.g. `Slice` starts/ends/axes/steps, `Reshape` target, optional `Resize` sizes) and on explicit `Shape` chains needed for diagnosis. Do not fold arbitrary feature-map operations or evaluate neural-network weight arithmetic.

Allowed shape-expression operations for a first implementation, subject to the *actual imported ONNX opset and operator schema*:

| Operator | Initially supported deterministic subset | Otherwise |
| --- | --- | --- |
| Constant / initializer | Small inline, immutable INT32/INT64 scalar or vector | UNKNOWN; never scan full weights or external weight payloads |
| Shape | Input rank and per-axis static dimensions; may produce a partially known vector | PARTIAL or UNKNOWN for symbolic dims |
| Gather | Small integer indices into a small shape vector; correct axis/negative-index handling | UNKNOWN on unknown indices, unsupported axes, or out-of-range indices |
| Add, Sub, Mul | Tiny integer-only scalar/vector elementwise values with proven ONNX broadcasting | UNKNOWN on unsupported dtype/broadcast or absent operands |
| Div | Tiny integer-only expression with verified ONNX integer semantics, nonzero divisor, safe bounds | UNKNOWN on zero divisor, mismatched types, overflow, or unreviewed rounding semantics |
| Concat | Concat of tiny known integer vectors where axis semantics are proved | PARTIAL/UNKNOWN otherwise |
| Unsqueeze / Squeeze | Tiny shape-expression tensors with proved axes and imported version semantics | UNKNOWN otherwise |
| Cast | Only provably lossless/reversible tiny integer conversion required by a shape path | UNKNOWN on narrowing/range ambiguity |
| Slice (on tiny integer vectors) | Constant small starts/ends/axes/steps and proved ONNX result | UNKNOWN for unreviewed corner cases |
| Slice (on a feature Tensor) | **Infer output shape only**, never read feature values | Keep unresolved axes unknown |
| Reshape (on a feature Tensor) | Infer target shape from a proved small target and ONNX version semantics | Keep unresolved dims unknown |

The above list adds **value/shape semantics**, NOT new X5 BPU rules for `Shape`, `Gather`, or other helper operators. These nodes remain `NOT_COVERED` in the hardware-rule registry unless independently given official X5 rule support in a later version.

#### Hard safety budgets

Use constants configurable in code but not user-provided executable YAML:

```text
MAX_TINY_ELEMENTS           = 64
MAX_ENCODED_CONSTANT_BYTES  = 4096
MAX_VALUE_BYTES             = 4096
MAX_DEPENDENCY_DEPTH        = 64
MAX_VISITED_SHAPE_NODES     = 2048
MAX_PROPAGATION_PASSES      = 4
MAX_UNKNOWN_TRACE_NODES     = 64
MAX_ABS_INTEGER_VALUE      = 2^63 - 1, unless the imported schema explicitly requires a narrower type
```

The numbers above are engineering budgets for **this bounded evaluator**, not claimed hardware constraints. Reject/skip an oversized or unsafe shape expression with a stable machine-readable reason. Never make huge Python lists based on untrusted shape metadata; never load external initializers as part of this feature.

The existing `small_constants.py` decodes only requested inline integers, currently 64 elements/4096 encoded bytes. Reuse its decoder and validation where practical. Extend the requested shape-constant set in a bounded way rather than reading every initializer. An overridable initializer (also declared as graph input) is NOT immutable.

### 4.2 Value representation

Use typed, immutable records (dataclass or validated dictionary). Proposed conceptual shape:

```python
@dataclass(frozen=True)
class TinyValueFact:
    tensor_name: str
    onnx_dtype: str
    value_shape: tuple[int, ...] | None
    values: tuple[int | None, ...] | None
    status: Literal['PROVEN', 'PARTIAL', 'UNKNOWN', 'CONFLICT']
    provenance_node_ids: tuple[str, ...]
    reason_code: str | None
    explanation: str | None
```

`None` **inside** a value vector means that an axis is unknown. It must never be silently converted into `0`, `1`, or a model batch guess.

A `TinyValueFact` describes a small integer **data value** such as the `[N,C,H,W]` vector produced by `Shape`. A Tensor's `shape` (metadata) is a different fact. Do not conflate "the value of a Shape node output" with "the Shape node output's own Tensor shape".

When `Shape(images)` is applied to a known `[1,3,640,640]` input, a complete small vector `[1,3,640,640]` can be PROVEN. The output Tensor of Shape itself then has metadata shape `[4]`. These are distinct results.

### 4.3 Evaluation plan

1. Build a name -> Tensor and name -> producer lookup using `GraphIR`, not display-name heuristics.
2. Use original and ONNX-inferred Tensor shape metadata as input facts.
3. Identify the bounded backwards dependency cones for needed shape parameters.
4. Traverse their actual producer/consumer links in a safe topological order. Detect non-DAG and nested graph cases; do not recurse indefinitely.
5. For each supported op, dispatch to a narrowly reviewed evaluator for the imported schema (e.g. opset 11). No `eval`, arbitrary Python code in YAML, or runtime loading of operator plugins.
6. Store value results with status, source edges, dependencies, ONNX version, and failure reasons.
7. Use those value facts to refine specific feature Tensor output **shape metadata**, starting with Slice/Reshape. Mark every improved axis and its provenance.
8. Propagate newly proved shape metadata only through safe, reviewed shape-transfer rules. Revisit dependent nodes in a bounded fixed-point loop; stop when no proved dimension changes or a budget is reached.
9. Feed the final proven Tensor shape facts into the existing resource analyzer and operator rule extractors, without mutating the original model file.
10. Recompute only those diagnostics whose evidence truly changed. For each such change, record `before`, `after`, and `why` in report JSON.

The simplest robust solution is a dedicated shape fact overlay on `GraphIR`. A temporary in-memory ONNX `ModelProto` with **metadata-only** annotations may be passed through upstream shape inference as an optional second pass, but do not rely on it working, and never serialize the modified representation to an ONNX file. Prove that all original known dimensions and public input/output contracts are preserved.

### 4.4 Slice output dimensions

For an ONNX Slice node:

- Distinguish data input from `starts`, `ends`, `axes`, and `steps` inputs.
- These integer parameters may come from a **computed shape subgraph**, not just a direct initializer or Constant.
- Normalize axes only when input rank is known.
- Validate parameter vector lengths, duplicate/invalid axes, dtype, nonzero step, and version-specific ONNX semantics.
- For a proved positive step=1 and an input axis with known positive length, compute exact clamped slice length according to ONNX semantics. Extend to other signs only with explicit tests matching ONNX reference behavior.
- Preserve dimensions on unaffected axes.
- If an affected axis length or an index is unknown, report that axis as unknown; do not discard already known axes or the whole Tensor shape.
- Never mark a shape-derived graph as an illegal BPU op just because the current evaluator does not understand it.

**High-priority real-model example:** recover a Slice channel boundary when `Shape` and `Gather` of a fixed-shape feature yield a provable integer. Doing so may resolve downstream Conv/Concat/Mul sizes. Do not hardcode specific `node_000019` channel counts into the product.

### 4.5 Downstream shape refinement

Implement only tested shape-transfer semantics that are needed for the actual graph, for example:

- Conv: reuse ONNX-inferred output shape if complete; any additional local derivation must respect kernel, stride, dilation, pads, groups, and ONNX `auto_pad` rules; avoid inventing a second generic Conv implementation solely to increase count.
- Sigmoid/Tanh/Relu: preserve input shape when operator semantics and rank are known.
- Add/Mul: use correct NumPy/ONNX broadcasting to derive output dimensions; unknown symbolic axes remain unknown if they cannot be proved.
- Concat: require known input ranks, same non-concat axes, and provable concat-axis dimensions before stating an exact sum.
- Split: only if split sizes and axis are fully proved by ONNX semantics.
- Reshape/Flatten/Unsqueeze: use exact imported schema and safe size arithmetic.
- Pool/Resize/MatMul: only if supported by explicit reviewed transfer logic; otherwise leave unchanged.

Do not infer layout (NCHW/NHWC) merely from rank=4, from a Tensor name, or from a plausible channel dimension. Layout evidence and shape evidence are different proof types.

### 4.6 Shape-result provenance and unknown-origin tracing

Every generated shape-analysis result should distinguish:

- `PROVEN`: all premises and the final dimension are known.
- `PARTIAL`: some axes known, some unresolved.
- `UNKNOWN`: cannot prove the relevant dimension.
- `CONFLICT`: proven facts disagree.

Recommended machine-readable `reason_code` values:

```text
SYMBOLIC_UPSTREAM_DIM
VALUE_NOT_STATIC
UNSUPPORTED_OPERATOR_VERSION
UNSUPPORTED_SHAPE_OPERATOR
UNSUPPORTED_INDEX_PATTERN
UNSUPPORTED_SLICE_STEP
MISSING_TENSOR_METADATA
OVERRIDABLE_INITIALIZER
EXTERNAL_DATA_NOT_READ
CONSTANT_BUDGET_EXCEEDED
SHAPE_PROPAGATION_BUDGET_EXCEEDED
DIVISION_BY_ZERO
INTEGER_OVERFLOW_OR_CAST_LOSS
ONNX_SEMANTIC_CONFLICT
LAYOUT_NOT_PROVEN
```

Expose the earliest blocking node(s) and Tensor(s) via a bounded, cycle-safe backwards trace. Do not print the entire graph repeatedly for each of 172 unknowns. Group unknowns by root cause in Markdown; keep per-Tensor full evidence in JSON.

### 4.7 CLI requirements for shape inspection

Add terminal-only queries consistent with the existing `nodes`, `tensor`, and `trace` commands. Suggested names:

```bash
python -m rdkx5_doctor shapes --analysis reports/model/analysis.json --summary
python -m rdkx5_doctor shapes --analysis reports/model/analysis.json --status UNKNOWN --limit 20
python -m rdkx5_doctor shape --analysis reports/model/analysis.json --tensor '/model/model.2/Slice_output_0' --json
```

If different command names better suit the existing CLI, document them explicitly in `README.md`, `--help`, and Skill. All queries must read saved JSON **without reloading the ONNX**. For a legacy 1.0/1.1/1.2 report with no V1.3 shape section, give a clear "rerun analyze to obtain shape provenance" message rather than inventing facts.

### 4.8 Regression invariants

- The 231 originally known Tensor raw-byte values in the real model must remain **identical**.
- Any newly known `raw_bytes` comes exclusively from a fully proved Tensor shape and supported dtype width.
- UNKNOWN count must never be reduced by dropping/merging Tensor records, replacing symbols with guesses, or ignoring unsupported types.
- No original model input/output shape, Tensor name, node ID, edge, or weight should change.
- Do not regress previously documented five rank-zero Mul static rule failures.
- No semantic claim about actual memory lifetime or peak deployment memory is introduced.

## 5. Versioning and compatibility plan

Suggested versions (Codex may select equivalent semantic versioning if well documented):

```text
Python package:          0.4.0
analysis schema:         1.3
multi-operator registry: 0.3.0
existing Conv subrules:  0.1.0 unchanged
V1.2 operator rule packs: retain their IDs/versions unless a separately proven rule fix is required
V1.3 new operator packs: each independently versioned with official source and review status
shape_analysis subsection schema: 1.0
```

Preserve existing top-level keys, especially `model`, `nodes`, `tensors`, `edges`, `diagnostics`, `traces`, `summary`, `resource_analysis`, and `optimization_candidates`.

Add optional `shape_analysis` and new per-op rule registration metadata; never break old `--analysis` queries. Legacy reports 1.0/1.1/1.2 must remain readable through their previously supported commands. New shape inspection returns a helpful error for old reports.

The new shape fact overlay must be the **same canonical fact source** used by Tensor resources and new MatMul/Resize/Softmax checks; avoid a report where one command says `[1,32,160,160]` but another still uses a stale `unk__*` dimension without explanation.

## 6. New BPU rule packs: source-driven and conservative

### 6.1 Authority and evidence requirements

Prior to writing ANY numerical hardware rule, Codex must read and compare the relevant RDK X5 ONNX rows in these official documents:

1. RDK X3/X5 official documentation, **specifically the RDK X5 ONNX section**:
   https://developer.d-robotics.cc/rdk_x_doc/Advanced_development/toolchain_development/intermediate/supported_op_list
2. X5 Chip User Manual 1.1.2:
   https://developer.d-robotics.cc/x5_sdk_doc/toolchain_development/intermediate/supported_op_list.html
3. X5 Chip User Manual 2.0.0 (English):
   https://developer.d-robotics.cc/x5_sdk_doc_v2.0.0/en/toolchain_development/intermediate/supported_op_list.html
4. ONNX operator schemas for the actual imported opset (semantic evidence, **NOT** hardware limits):
   https://onnx.ai/onnx/operators/onnx__MatMul.html
   https://onnx.ai/onnx/operators/onnx__Softmax.html
   https://onnx.ai/onnx/operators/onnx__Resize.html

Source facts that motivated this specification (verify again during implementation):

- **MatMul:** RDK X5 describes non-4D support subject to equal input ranks, last-two-axis limits (reported `[1,8192]`), higher-axis limits (reported `[1,4096]`), and restricted high-axis broadcast patterns. ONNX itself supports more general NumPy-like MatMul behavior.
- **Softmax:** official X5 documentation states the op defaults to CPU. For the documented `onnx::softmax` path, rank-4 input with axis=3 can be directed to BPU using `run_on_bpu`; PyTorch-specific paths have separately described conditions. Do NOT infer PyTorch converter provenance merely from an ONNX node name.
- **Resize:** official X5 documentation describes 4D NCHW feature data, H/W-only resizing, `nearest` or `linear`, version-specific coordinate modes, and specific `nearest` enlargement factor requirements. The available evidence for ROI and scaling must be checked before a deterministic verdict.

Do not blindly copy the textual summaries above into YAML. Codex must retain the exact quoted source subsection, manual version, source URL, date of review, field derivation, whether the field is directly observed, and any cross-version discrepancy or exclusion. If a source is unavailable, retain only previously verifiable constraints and mark new disputed items `review_only`/UNKNOWN. Do not invent a fallback restriction.

**Never import the CPU-support column** as a BPU rule. In particular, ONNX format validity, a CPU-only `float` requirement, an int16-capability note, and the compiler's actual quantized dtype are four different topics.

New pack metadata must be consistent with the V1.2 `Strict` Pydantic schemas, safe YAML loader, `RuleRegistry` manifest, operator field allowlists, predicate allowlists, version/domain checks, and `rules list` CLI.

### 6.2 MatMul - first priority in the new hardware layer

Actual real-model nodes: two MatMul nodes in attention. Use real `nodes`/`inspect`/Tensor queries to retrieve exact inputs before finalizing the real-model report.

**ONNX semantic facts:** `A`, `B`, their ranks, full Shapes, dtype, last-two-axis matrix dimensions, leading batch axes, actual output Shape, and whether known dimensions permit ONNX MatMul. Do not turn an ONNX semantic issue into an X5 rule violation or vice versa.

**RDK X5 BPU fact extraction / candidate static checks, after source verification:**

| Candidate check | Derive from | When a deterministic FAIL is permitted |
| --- | --- | --- |
| Input rank relationship | Actual `A_shape`, `B_shape` | Source explicitly forbids the observed rank relationship; known ranks |
| Last-two-axis limits | Concrete last-two dims of each input | Observed concrete value violates an official X5 bound |
| Higher-axis limits | Concrete high axes | Observed concrete value violates an official X5 bound |
| Leading-axis broadcast pattern | Both inputs' real high-dimensional axes | Fully known high-dimensional pattern demonstrably falls in an officially excluded case |
| ONNX matmul compatibility | MatMul schema for imported opset | Report an ONNX semantic issue separately, not as an X5 BPU FAIL |
| Final compiler layout / allocation | Not present in static ONNX | UNKNOWN / review only, never a made-up PASS |

Special care is required for the official **asymmetric broadcast patterns** for inputs A and B. Do not substitute ordinary NumPy broadcasting (which validates ONNX legality) for the more restrictive X5 BPU broadcast rules. Implement a pure, whitelist predicate with separately named evidence for each high axis. Test documented positive and counterexample patterns.

If an axis is symbolic, do not conclude the broadcast pattern is permitted or forbidden unless the remaining concrete facts already prove a contradiction. Keep a structured `UNKNOWN` reason with the exact unresolved axis and Tensor.

Suggested field and evidence names, subject to validation against source and existing field conventions:

```text
matmul_a_rank
matmul_b_rank
matmul_rank_relation_ok
matmul_a_last_two_dims
matmul_b_last_two_dims
matmul_higher_dims_within_limits
matmul_broadcast_pattern_supported
matmul_onnx_matrix_product_compatible
matmul_output_shape
```

Suggested rule ID namespace: `X5-MATMUL-*`. Do not assign a rule ID to a condition with no verified official BPU requirement.

Expected interpretation for attention: a MatMul can be ONNX-legal and still need validation of documented X5 leading-axis broadcasting. The diagnosis must mention actual Tensor Shapes and node IDs, not a generic "attention is expensive" assertion.

### 6.3 Softmax - documented CPU default versus BPU eligibility

The real model includes a Softmax node in the attention sequence. Retrieve its **actual opset, axis attribute, input rank, actual operator domain, and output path**. Do not assume axis=-1 or axis=3 without reading the node.

For standard ONNX-domain Softmax:

1. Normalize positive or negative axis against a **known input rank**, with imported ONNX schema semantics taken into account.
2. Validate the actual ONNX Softmax semantics for the imported schema: opset 11 and opset 13 do not have identical axis interpretation. Do not silently apply modern per-axis semantics to opset 11.
3. Check the official X5 conditions for the standard `onnx::softmax` BPU-eligible path (documented rank-4 + axis=3) only after source review.
4. Keep the explicit `run_on_bpu`/compiler configuration question as `REVIEW_ONLY` or a separate eligibility state because it cannot be determined from the ONNX node alone.
5. Do not infer the PyTorch-specific path from the fact that the model was originally trained using PyTorch. A standard ONNX-domain node must be analyzed under its actual ONNX contract; provenance not represented in the model stays unknown.
6. If the node does not satisfy the documented BPU-eligible path, describe that specific path as unsupported/not established, **not that the model is invalid or necessarily fails compilation**. It may run on CPU.

Record hardware eligibility separately from the existing node verdict if that improves clarity:

```json
{
  "operator": "Softmax",
  "static_bpu_path": "ELIGIBLE_WITH_COMPILER_CONFIGURATION",
  "compiler_configuration_verified": false,
  "actual_runtime_placement": "UNKNOWN",
  "reason": "Documented ONNX Softmax path has rank=4 and normalized axis=3; run_on_bpu not inspected"
}
```

This is an example, not a claim about the real node. Alternative statuses may be `NOT_ELIGIBLE_UNDER_DOCUMENTED_ONNX_PATH` and `UNKNOWN`. The existing four diagnostic statuses continue to describe the checked rule set; avoid a misleading "BPU PASS" headline.

Suggested source-backed rule namespace: `X5-SOFTMAX-*` plus one **nonblocking** `run_on_bpu` review item. No code that edits compiler configs is required.

### 6.4 Resize - rank/layout, mode, coordinate semantics, scale evidence

The real YOLO26 model contains one Resize. Obtain the exact attributes and connections from the actual ONNX node, including:

```text
input X Tensor name, dtype, shape
optional ROI input and whether a fixed constant is provable
optional scales Tensor, its type/source/value if small and immutable
optional sizes Tensor, its type/source/value if small and immutable
mode
coordinate_transformation_mode
nearest_mode (ONNX semantic metadata, not automatically a BPU restriction)
output Shape
source opset and operator schema version
```

**Bounded numeric constant decoding:** Unlike the existing integer shape constants, `scales` and ROI may be FLOAT32/FLOAT64. Add a *separate* strict decoder for **requested small immutable numeric constants only**. Enforce `MAX_TINY_ELEMENTS`, encoded-byte limits, finite values, correct element counts, external-data rejection, overridable-initializer rejection, and observed dtype. Never load feature map contents or large weights. If the requested value is unavailable, keep the field UNKNOWN.

Hardware-rule candidates, subject to current X5 source verification:

- Input feature Tensor has rank 4. NCHW layout requires reliable layout evidence, **not rank=4 alone**.
- Only the spatial H/W axes may be resized. N and C scales/sizes are unchanged when those dimensions are provable.
- Supported `mode` values as documented (`nearest`, `linear`). Distinguish `cubic` as an ONNX-legal value that may be outside the documented X5 BPU path.
- Supported coordinate transformation modes depend on imported opset and official X5 column; use verified version conditions, never apply a general latest-ONNX rule to opset 11.
- For `nearest` **enlargement**, apply the documented factor restrictions only if enlargement and both scale factors are provable. Do not apply enlargement-only constraints to a genuine downsampling case.
- The documented H/W factor relationship must use the actual proven factors, and must not be inferred from an output Shape with unknown input dimensions.
- If `tf_crop_and_resize` is selected, verify the official ROI requirements only where the ROI and transformed boundaries are statically provable. A partially known ROI is UNKNOWN, not PASS.
- Compiler-specific conversion behavior, fused Resize placement, and actual execution remain unverified.

For floating-point scales, avoid fragile strict equality against rounded values: define and test a defensible numerical comparison or exact integer-factor recognition. If the factor cannot be proved unambiguously under the input dtype, return UNKNOWN rather than PASS/FAIL. Do not invent a tolerance claimed to be the official X5 tolerance.

Rule namespace: `X5-RESIZE-*`. Source every rule in the versioned YAML. Preserve the separate ONNX semantic issue list for malformed or contradictory sizes/scales.

### 6.5 Shared rule-engine requirements

Reuse rather than replace V1.2's secure rule registry:

- Add `MatMul`, `Softmax`, `Resize` to the operator `Literal`/allowed-registry map.
- Add explicit per-operator fields to `operator_fields.py` (or its current equivalent).
- Add a whitelist of pure predicates to `rule_predicates.py` if required. A YAML rule names an allowed predicate; it must never execute Python expressions.
- Update `operator_facts.py` schema allowlists using `onnx.defs.get_schema(op, imported_opset, '')`. Import opset 11 is mandatory for the real model; add other versions only if the relevant semantics have actually been reviewed.
- Preserve strict YAML key validation, duplicate-ID rejection, cross-file/operator mismatch rejection, source URL restrictions, path/symlink escape rejection, and explicit version compatibility.
- Treat official source versions independently of rule-pack version, model opset, Python/ONNX package version, and installed Horizon toolchain version.
- Per rule: actual/expected values, `PASS` / `FAIL` / `UNKNOWN` / `NOT_APPLICABLE`, blocking status, source metadata, observed node/Tensor evidence, reason code, and unverified assumptions.
- At node level: keep exactly one diagnostic per node and preserve `VIOLATION`, `NEEDS_VERIFICATION`, `NO_VIOLATION_FOUND`, `NOT_COVERED`. An all-review/no-auto-check operator must never become a blanket verified PASS.
- `rules validate` and `rules list --operator MatMul|Softmax|Resize --json` must work from both editable install and independently installed wheel.

### 6.6 Carefully distinguish a failure from an unverified compiler decision

Examples:

| Observation | Correct interpretation |
| --- | --- |
| MatMul violates a clearly sourced, fully evaluated X5 dimensional bound | Static X5 BPU rule FAIL, node VIOLATION; still not a compiler-failure prediction |
| MatMul rank valid but unknown high-axis dimensions | UNKNOWN / NEEDS_VERIFICATION, with exact missing axis |
| Standard ONNX Softmax has documented eligible axis/rank | Static eligible path; `run_on_bpu` and actual placement still unverified |
| Standard ONNX Softmax fails the documented BPU-only path | BPU path eligibility issue, **not** ONNX invalidity or proof of compile failure |
| Resize `mode=cubic` where official X5 BPU row only lists nearest/linear | Static documented-path conflict, ONNX may remain valid |
| Resize uses constant scales but H/W values unknown | UNKNOWN; do not guess nearest power-of-two factors |
| Five rank-zero Mul nodes from V1.2 | Retain documented original-graph static conflict; compiler scalar folding unknown |

No generalized "risk score" or made-up performance/accuracy estimate is allowed.

## 7. Source-independent, architecture-level advice only

The output narrative should have two compact levels:

**Level 1 - provable model fact:**

```text
Node ID / operator / exact Tensor names / observed property
Official X5 rule and source version
What the static rule can and cannot conclude
```

**Level 2 - optional model design direction:**

```text
Possible area of investigation: decouple neural output head and downstream
postprocessing, change an operator arrangement, or use an export-friendly
formulation. Such changes belong in the original training/export project.
Re-export the ONNX, compare public interfaces and numerical/task outputs,
then rerun the ONNX Doctor. No source-code location is asserted.
```

Do not make architecture advice mandatory for every finding. A missing Shape fact usually needs only an analysis-engine improvement, not a recommendation to redesign a network.

For the lanerobot `main` versus `quant corrected` analogy, Codex can say "an export/quantization-friendly output-head separation may be considered". It must **not** claim it knows the detailed branch diff or suggest editing a specific file without the source repository in the current task.

Do NOT add features to search a training repo, edit PyTorch code, create git branches for model architecture, or automatically transform ONNX.

## 8. JSON schema, CLI, Skill and report integration

### 8.1 Schema 1.3 extensions

Keep existing schema 1.2 fields and add a dedicated `shape_analysis` section:

```json
{
  "schema_version": "1.3",
  "shape_analysis": {
    "schema_version": "1.0",
    "method": "BOUNDED_STATIC_INFERENCE",
    "summary": {
      "before_known_tensor_count": 231,
      "before_unknown_tensor_count": 172,
      "after_known_tensor_count": null,
      "after_unknown_tensor_count": null,
      "newly_proved_tensor_count": null,
      "newly_proved_axis_count": null,
      "conflict_count": null,
      "reason_counts": {}
    },
    "facts": [],
    "unknown_origins": [],
    "limits": {},
    "limitations": []
  }
}
```

`before_*` values above represent the **specific reference model**, not universal defaults. The actual implementation must compute them from the current model. No result may leave `after_*` null when a complete analysis finishes; partial analysis must state why it is partial.

Each fact must include immutable provenance sufficient to audit it. Output contract:

```text
Tensor name
original Shape and dtype
ONNX-inferred Shape and dtype
final Shape and dtype
per-axis status and value
producer and contributing source node IDs
source Tensor names and actual values (within budgets)
operator/version transfer rule
explicit proof explanation
unknown/blocker reason codes
whether Tensor payload size changed from unknown to known
```

Never include entire raw weight contents. Model source path can be omitted from Markdown (basename + hash), consistent with V1.2 report policy. Preserve source ONNX SHA256 scope limitations for external data.

If `shape_analysis` is unavailable in older saved JSON, do not alter old `tensors` commands to require it. Only new `shape`/`shapes` commands should need the new section.

### 8.2 Add diagnostic layers without concealing evidence

V1.3 checks should use the improved canonical Tensor metadata. When an existing diagnosis changes because shape facts are improved, record:

```json
{
  "node_id": "main/node_000029",
  "field": "broadcast_mergeable",
  "previous_status": "UNKNOWN",
  "new_status": "PASS_OR_UNKNOWN_AS_PROVED",
  "evidence": ["named shape fact ids"],
  "note": "Computed from statically proved input dimensions; compiler behavior remains unverified"
}
```

The illustrative `new_status` above is a placeholder: in real data it must be a valid status such as `PASS`, `FAIL`, or `UNKNOWN`. Do not duplicate old and new diagnostics in the `diagnostics` list; each node must still have exactly one current diagnostic.

Respect the existing status precedence: fully proved failures are VIOLATION; blocking unknowns produce NEEDS_VERIFICATION; no applicable rules means NOT_COVERED. Conditions that need compiler flags should remain an explicit review item, not be promoted to a hardware PASS.

### 8.3 Terminal commands and report contents

Preserve V1.2 CLI:

```text
analyze
rules validate
rules list
nodes
inspect
trace
tensors
tensor
candidates
candidate
```

Add optional new commands for shape forensics, as outlined in section 4.7. Run rules list on each newly supported operator:

```bash
python -m rdkx5_doctor rules list --operator MatMul --json
python -m rdkx5_doctor rules list --operator Softmax --json
python -m rdkx5_doctor rules list --operator Resize --json
```

Markdown report rules, extending V1.2's existing anomaly-first design:

1. Model identity, hash, standard ONNX opset, registered rule versions, reviewed manual versions, and `toolchain_verified=false`.
2. Operator coverage matrix (all 23 observed real-model types), with `AUTO_CHECKED`, `PARTIAL_OR_CONDITIONAL`, and `NOT_COVERED` clearly separated from node PASS/FAIL.
3. Static rule failures, grouped by operator and reason. Preserve full node/Tensor source evidence in JSON. Avoid restating identical benign details 100 times.
4. Blocking and nonblocking unknowns, grouped by root cause and exact node count (group counts can overlap; do not misreport them as unique node count).
5. Shape inference: counts **before/after**, number of proved axes, representative fixed cases, unresolved first blockers, and conflicts. Never promise all 172 will resolve.
6. Tensor resources: output Tensor rows, Top 10 *known* intermediates, remaining unknown counts, and a warning that these raw bytes are not peak BPU/DDR memory.
7. MatMul + Softmax + Resize: actual attributes, shaped input evidence, exact X5 source, static status, compiler-configuration caveats, output reachability if relevant.
8. Graph candidates: preserve the five V1.1 conservative structural candidates; zero remains zero when no actual candidates exist.
9. Concise, architecture-only improvement directions **when justified**. Explicitly say that model changes happen in the original training/export project, not in this repository.
10. Exact non-verifiable items: compiler transformations (including scalar normalization, Softmax run_on_bpu, Gemm conversion), quantized dtype, runtime placement, latency, accuracy, final peak memory.

Honor V1.2 report truncation strategy: render at most 50 definite violation node rows, at most 10 examples per unknown group, at most 10 candidates; full machine evidence remains in analysis.json. Escape model-origin names and control characters. Do not regenerate V1 verbose "no modifications necessary" advice for all normal nodes.

### 8.4 Update the Codex Skill entrypoint

Update `.agents/skills/rdk-x5-onnx-doctor/SKILL.md` to define a **decision workflow**, not a dump of fixed shell commands:

1. Validate rule packs and read the model.
2. Check important static violations and missing source evidence.
3. Inspect `shape_analysis` before making claims based on symbolic Shapes.
4. When a MatMul/Softmax/Resize issue matters, inspect that node and trace actual output dependencies.
5. For unresolved shape facts, query first blocking source(s); avoid investigating 172 unknowns individually without grouping.
6. Explain exact rule evidence and uncertainty.
7. Offer only model-agnostic architectural directions, optionally output-head separation, not patches to an unknown training repository.
8. End with an explicit offline-validation boundary and the path to the saved report.

Keep `SKILL.md` concise (workflow control). Place detailed operator semantics in references and the V1.3 development/schema documents.

### 8.5 Git hygiene

Current `.gitignore` correctly ignores `reports/`; do not remove that line. Keep demo fixtures small and safe; never commit the 104 MB real model, model-specific validation results, external weights, temporary wheel environments, or per-node stdout/stderr logs.

Only source code, tests, YAML/rule references, generic examples, schema docs, and this development specification are tracked deliverables. Do not re-add any historical model reports to Git. Do not automatically commit/push. Do not delete untracked user files.

## 9. Development phases for Codex

**Execute, test, and report each phase before moving to the next.** If a phase uncovers a genuine defect, fix the smallest root cause, add a regression test, then continue. Do not stop after a plan.

### P19 - Baseline and rule-source audit

- Verify the working tree and preserve uncommitted user changes.
- Read the actual V1.2 repository state and local V1.2 real-model report.
- Run `--help`, `rules validate`, and all pytest tests, preferably `env -u PYTHONPATH .venv/bin/python -m pytest -q`.
- Re-analyze the real YOLO26 into a **new** ignored baseline directory.
- Record original input/output metadata, all node IDs, graph connectivity, known/unknown Tensor counts, 5 Mul failures, and source SHA256.
- Check each official X5 MatMul/Softmax/Resize BPU row against its version and cross-manual conflicts; record citations and exclusions.

**Exit criterion:** trustworthy V1.2 baseline and written source review. No behavior changes yet.

### P20 - Evidence-aware tiny integer evaluator

- Implement typed tiny shape values and provenance.
- Support Constant, immutable initializer, Shape, Gather, and selected Add/Mul/Div/Sub for shape-parameter dependency cones.
- Enforce hard count, size, type, overflow, depth, and external-data bounds.
- Respect actual imported opset and tensor rank vs tensor values.
- Unit-test all supported and unsupported cases, including dynamic and malformed input.

**Exit criterion:** deterministic shape-derived small integers with no feature-map or weight allocation.

### P21 - Slice/Reshape shape refinement and unknown tracing

- Infer feature Slice output dimensions when starts/ends/axes/steps are proved.
- Reuse known Shape facts for Reshape targets; preserve exact -1/0/allowzero semantics by imported schema.
- Propagate downstream only through reviewed shape-transfer logic.
- Add per-axis proof chains and root-cause UNKNOWN categories.
- Create `shape_analysis` JSON and shape CLI queries.
- Test fixed-point behavior, false contradictions, and memory budgets.

**Exit criterion:** a repeatable before/after shape comparison with all original known facts preserved.

### P22 - Prove integration with existing BPU/Resource results

- Feed proved Tensor metadata into existing `tensor_resource` and Add/Mul/Concat extractors.
- Assert original known-byte results are unchanged.
- Log all status changes with exact proof IDs.
- Ensure baseline five rank-zero Mul findings remain present and identical in evidence.
- Run real-model analysis and review every newly concrete Tensor Shape by an independent oracle where possible.

**Exit criterion:** improved data without stale or conflicting facts between CLI, JSON, and Markdown.

### P23 - MatMul rule pack

- Implement standard ONNX MatMul extraction for actual opset 11 and explicitly reviewed versions.
- Add X5 MatMul YAML, secure predicates, registry entries, documented source metadata, and tests.
- Test rank relation, boundary dimensions, high-axis broadcast patterns, symbolic inputs, ONNX-legal-but-BPU-unsupported cases, and non-standard domain.
- Run `rules list --operator MatMul --json` and inspect two actual attention MatMul nodes.

**Exit criterion:** independent per-rule values, correct PASS/FAIL/UNKNOWN boundaries, no invented compiler verdicts.

### P24 - Softmax rule pack and eligibility review

- Extract actual `axis`, normalized axis, input rank, imported schema version and actual operator domain.
- Add sourced static BPU eligibility checks and **nonblocking** `run_on_bpu` configuration review.
- Distinguish standard ONNX softmax from a provenance-confirmed converter path.
- Verify opset 11 semantics and add opset 13 tests showing the relevant distinction.
- Inspect actual attention Softmax and trace its MatMul neighbors/output reachability.

**Exit criterion:** no unsupported inference of actual CPU/BPU placement; meaningful documented-path guidance.

### P25 - Resize rule pack

- Add exact attrs and input-source extraction for the real Resize node.
- Add a separate bounded FLOAT32/FLOAT64 tiny scale/ROI decoder if needed; keep the existing INT decoder safe.
- Add official X5 rule checks for rank/layout, H/W-only resizing, mode, coordinate behavior, and provable nearest factors.
- Keep unknown factor/layout/ROI and conversion cases as review/UNKNOWN.
- Test opset10/11 differences and documented positive/negative patterns, including downscaling.

**Exit criterion:** real Resize diagnostics are evidence-bound, not extrapolated from rank or names.

### P26 - CLI, report, Skill and docs

- Bump schema and package versions following project conventions.
- Update `README`, `.agents/skills/rdk-x5-onnx-doctor/SKILL.md`, `AGENTS.md`, schema docs, report policy, source review, and development log.
- Implement shape CLI commands and registered rule list support.
- Keep report compact and machine facts complete; avoid generic source-code modification advice.
- Ensure ignored `reports/` output and no incidental model or log commits.

**Exit criterion:** an independent reader can understand what was proved, what remains UNKNOWN, and how to query evidence.

### P27 - Full regression and real-model acceptance

- Run all V1-V1.3 unit tests, check CLI help, validate rules, and build wheel/sdist.
- Install final wheel into a new temporary venv. From a directory outside the source tree, run `rules validate`, three new `rules list`, real `analyze`, shape query, node inspect/trace, and Tensor queries.
- Compare original model SHA256 before/after.
- Verify original 47 Conv rule outcomes and 5 Mul static failures remain evidence-identical.
- Re-run real-model coverage and unknown-shape analysis; report actual changed numbers, not expected numbers.
- Check JSON schema integrity, deterministic order and output, and no accidental external data loading.
- Write the full `REAL_ONNX_V1_3_VALIDATION.md` into the ignored real-model report directory, not Git.
- Check Git status; do not commit or push.

**Exit criterion:** reproducible real-model results from both editable and independently installed distributions, with no regression.

## 10. Test matrix: minimum expectations

### 10.1 Tiny integer and Shape evaluator

- Known input `[1,3,640,640]` -> Shape -> correct tiny vector and Tensor metadata `[4]`.
- Shape with partially unknown `[1, 'batch', 64, 64]` -> `[1,None,64,64]`, never `[1,1,64,64]`.
- Gather positive/negative known index; scalar and vector index; out-of-range index stays unknown/semantic issue as appropriate.
- Integer Add/Mul/Sub with identical and broadcastable tiny shapes; ONNX-invalid broadcast detected separately.
- Integer Div exact positive cases; negative-rounding cases match imported ONNX spec; division by zero and overflow do not crash.
- Constant immutable vs overridable initializer; empty/raw/typed encoding; malformed/truncated payload.
- External initializer must not be read; oversized tiny tensor fails closed.
- Symbolic dims do not become fake concrete sizes; conflicts preserve both facts.
- Unsupported operator schema/domain triggers UNKNOWN, not a fallback interpretation.
- Dependency graph with repeated diamond paths cannot cause exponential work; bounded trace records truncation.

### 10.2 Slice and derived feature Shapes

- Slice with static positive steps and fully known bounds yields an exact output Shape.
- Slice ends computed by Shape -> Gather -> Div -> Mul are resolved if all inputs are proved.
- Negative/empty/reversed/out-of-bounds cases are tested against the exact reviewed ONNX semantics; unsupported cases remain UNKNOWN.
- Slicing only changes selected axes; all other concrete axes stay concrete.
- Symbolic affected axis does not invalidate known unaffected dimensions.
- Several Slice and downstream Conv/Concat/elementwise ops produce consistent shapes after bounded passes.
- Rerunning analysis is deterministic.
- All originally known Tensor byte counts remain exactly equal, not approximately equal.

### 10.3 MatMul rules

- ONNX legal 2D matrix product with in-bound dims.
- X5 documented dimension bounds: lower edge, upper edge, just below/above as sourced.
- Equal-rank and mismatched-rank inputs (legal ONNX matmul can have different ranks).
- Supported high-axis broadcast examples from the official manual.
- Explicitly unsupported broadcast counterexamples from the official manual.
- Unknown high-axis dimensions remain UNKNOWN.
- Invalid ONNX K alignment generates an ONNX semantic issue rather than being mislabeled X5 FAIL.
- Wrong standard-domain/opset or invalid attributes cannot select a permissive fallback.

### 10.4 Softmax rules

- Standard ONNX-domain rank4 with documented eligible axis, and missing `run_on_bpu` configuration evidence.
- Rank4 with other axes; rank3; rank unknown.
- Axis=-1 normalized correctly for known rank4.
- Opset11 versus opset13 semantics reviewed separately.
- Custom PyTorch-style domain does not get standard-ONNX rule verdicts unless documented schema support exists.
- Default CPU behavior is not reported as a compile failure or an actual placement measurement.

### 10.5 Resize rules

- Rank4 NCHW input with trusted layout versus rank4 with unknown layout.
- Correct `nearest`/`linear` modes; documented-unsupported `cubic` mode.
- Supported and unsupported coordinate transformation modes by reviewed opset.
- Proven H/W resize with N/C fixed versus unproven N/C behavior.
- Nearest power-of-two enlargements, non-power-of-two enlargement, H-factor versus W-factor, and valid downscaling.
- Float scales FP32/FP64 supported only within strict bounded constant budgets.
- Sizes-only input; scales-only input; missing/ambiguous inputs.
- Fixed ROI versus unknown ROI in `tf_crop_and_resize` mode.
- Unsupported nearest_mode semantics must not be invented from CPU column text.

### 10.6 Security, report, compatibility

- YAML rejects unknown fields, predicate names, duplicate keys, path escapes, symlink escapes and operator/field mismatch.
- Rules retain source/document/version/opset/column and exact node facts.
- `analysis.json` schema1.3 full facts; legacy 1.0/1.1/1.2 queries still work as previously documented.
- `shapes` on a V1.2 JSON gives an explicit upgrade message, not a stack trace.
- Proper escaping for model-origin terminal control sequences and Markdown special characters.
- Status and operator coverage counts sum to actual graph node count.
- New UNKNOWN groups have accurate unique-node counts and group overlap labels.
- Normal nodes are not repeated as verbose pseudo-recommendations.
- Original model byte digest unchanged, no ONNX file written, no Docker/PTQ/run_on_bpu configuration modified.
- Git excludes reports and large ONNX/weight files; packaged wheel contains the new YAML registry rules.

## 11. Real model validation checklist

Use a fresh report directory, e.g. `reports/yolo26-v1_3-<timestamp>`.

```bash
MODEL='/home/xhm/lianghua_ws/rdktoolchain/horizon_x5_open_explorer_v1.2.8/horizon_x5_open_explorer_v1.2.8-py310_20240926/samples/ai_toolchain/horizon_model_convert_sample/04_detection/15_tasknav/model/yolo26_lane_robot.onnx'
sha256sum "$MODEL"

.venv/bin/python -m rdkx5_doctor rules validate
.venv/bin/python -m rdkx5_doctor rules list --operator MatMul --json
.venv/bin/python -m rdkx5_doctor rules list --operator Softmax --json
.venv/bin/python -m rdkx5_doctor rules list --operator Resize --json

env -u PYTHONPATH .venv/bin/python -m pytest -q
.venv/bin/python -m build --no-isolation

.venv/bin/python -m rdkx5_doctor analyze --model "$MODEL" --out reports/yolo26-v1_3-final
.venv/bin/python -m rdkx5_doctor shapes --analysis reports/yolo26-v1_3-final/analysis.json --summary
.venv/bin/python -m rdkx5_doctor nodes --analysis reports/yolo26-v1_3-final/analysis.json --status VIOLATION
.venv/bin/python -m rdkx5_doctor nodes --analysis reports/yolo26-v1_3-final/analysis.json --search MatMul
.venv/bin/python -m rdkx5_doctor nodes --analysis reports/yolo26-v1_3-final/analysis.json --search Softmax
.venv/bin/python -m rdkx5_doctor nodes --analysis reports/yolo26-v1_3-final/analysis.json --search Resize
.venv/bin/python -m rdkx5_doctor tensors --analysis reports/yolo26-v1_3-final/analysis.json --kind intermediate --sort bytes --limit 10

sha256sum "$MODEL"
git status --short
```

The `shapes` command is the proposed V1.3 CLI; if the implementation adopts a different documented name, use that name. The sample final report path should be made unique to avoid overwriting a previous run.

For actual important nodes, use the current `nodes --search` output to obtain `main/node_...` IDs, then call `inspect --node ... --json` and `trace --node ... --json`. Do not use the example ID blindly if an updated ONNX differs from the baseline SHA256.

### Required real-model result table

Complete this table using actual tool output; do not prefill the 'after' column with a desirable answer:

| Measurement | V1.2 reference | V1.3 actual |
| --- | ---: | --- |
| Original ONNX SHA256 | a8ea6ccf474c77614f33512c4a118390f3b14c28e4871615670888a6ca292de4 | To measure |
| Total nodes | 281 | To measure |
| Total Tensor records | 403 | To measure |
| Known-size Tensors | 231 | To measure |
| Unknown-size Tensors | 172 | To measure |
| Static Mul violations | 5 | To measure and compare |
| Rule-scope nodes | 181 | To measure; up to 185 with 3 new op types |
| MatMul/Softmax/Resize new checked nodes | 0 | To measure |
| Actual BPU placement | Not measured | Not measured |
| Real-model optimization execution | Not performed | Not performed |

### Success does NOT require a fabricated result

An eligible MatMul with UNKNOWN BPU broadcast due to a symbolic axis is acceptable if the exact unknown is shown. A Softmax with a documented static eligible path but unknown `run_on_bpu` config is acceptable. Resize with unknown layout is acceptable. Unknown Shape facts that cannot be proved must remain unknown.

The objective is **accuracy, explainability, and explicit coverage**, not maximizing the number of PASS labels.

## 12. Final handoff that Codex must provide

When implementation is done, provide a compact terminal summary:

```text
1. Actual HEAD and local Git status; identify modified, new and deleted source files.
2. V1.3 package/schema/rule-registry versions.
3. Tested shape evaluator operations, budgets and limitations.
4. Tensor sizes known/unknown before and after, with 3-5 concrete proof examples.
5. MatMul/Softmax/Resize per-rule findings on real nodes, with actual source URLs.
6. Whether all 5 original Mul findings remain; explain any evidence-driven change.
7. Count of tests passed; actual build and independent-wheel validation outcomes.
8. Locations of analysis.json, report.md and ignored REAL_ONNX_V1_3_VALIDATION.md.
9. Explicit list of unresolved conditions and model-only architecture directions.
10. Statement that no source ONNX, training project, Docker/quantization, or remote Git branch was modified.
```

If a milestone remains incomplete, explain precisely what was implemented, what failed, and what follow-up is needed. Do not claim success for tests not run.

## 13. Copy/paste prompt to initiate this specification in Codex

> Read `RDK_X5_ONNX_Doctor_V1_3_Shape_Inference_And_Attention_BPU_Development_Spec.md` in this repository and carry out phases P19 through P27. First verify the repository baseline and review official RDK X5 ONNX BPU constraints. Then implement bounded, evidence-bearing static Shape inference and the MatMul/Softmax/Resize rule packs; integrate CLI, report, Skill and tests. Revalidate my real `yolo26_lane_robot.onnx` at the absolute path in this specification. Preserve all existing V1.2 behavior and the five rank-zero Mul findings. Do not modify ONNX or guess edits to a training repository. Engineering advice may be architecture-level only, such as decoupling an output head in a separate source project. Do not use Docker or hb_mapper, do not generate GUIs, and do not commit or push Git changes. Complete actual code development, all tests, wheel validation, and a local ignored Markdown/JSON real-model report, then summarize evidence and limitations.

---

## Appendix A. Source map (recheck dates/versions during implementation)

**Hardware sources:**

- RDK X5 ONNX operators: https://developer.d-robotics.cc/rdk_x_doc/Advanced_development/toolchain_development/intermediate/supported_op_list
- X5 chip manual 1.1.2: https://developer.d-robotics.cc/x5_sdk_doc/toolchain_development/intermediate/supported_op_list.html
- X5 chip manual 2.0.0: https://developer.d-robotics.cc/x5_sdk_doc_v2.0.0/en/toolchain_development/intermediate/supported_op_list.html

**ONNX semantics sources (not X5 hardware rules):**

- MatMul: https://onnx.ai/onnx/operators/onnx__MatMul.html
- Softmax: https://onnx.ai/onnx/operators/onnx__Softmax.html
- Resize: https://onnx.ai/onnx/operators/onnx__Resize.html
- Shape: https://onnx.ai/onnx/operators/onnx__Shape.html
- Gather: https://onnx.ai/onnx/operators/onnx__Gather.html
- Slice: https://onnx.ai/onnx/operators/onnx__Slice.html
- ONNX shape inference documentation: https://onnx.ai/onnx/repo-docs/ShapeInference.html

**Repository baseline documentation:**

- `docs/ANALYSIS_SCHEMA_V1_2.md`
- `docs/REPORT_POLICY_V1_2.md`
- `docs/DEVELOPMENT_LOG.md`
- `references/x5_multiop_sources.md`
- `.agents/skills/rdk-x5-onnx-doctor/SKILL.md`

## Appendix B. Strict scope statement

This V1.3 is an **offline ONNX diagnostic skill plus deterministic Python analyzer**. It does not gain access to or understanding of the training/export project merely by observing the ONNX computation graph. Any architecture change (including output-head decoupling) must be performed separately by the owner of that model project, followed by re-export and independent validation.
