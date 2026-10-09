# RDK X5 ONNX Doctor - V1.1 Development Specification

**Scope:** Static Tensor Resource Analysis + Graph-Based Redundancy and Optimization Candidate Discovery<br>
**Target repository:** `Stradlin1/skillzuoye`<br>
**Target environment:** Ubuntu, VS Code, Codex, Python 3.10+<br>
**Document date:** 2026-10-09<br>
**Implementation mode:** Incremental modification of the existing repository, not a new project

> **Instruction to Codex:** Implement the complete V1.1 scope specified here in the existing repository. Do not merely summarize this document. Read the current code before editing, preserve all V1 features, implement in stages, run relevant tests after each stage, and record actual test evidence. If an unknown value cannot be established from ONNX metadata, preserve it as unknown. Do not fabricate platform conclusions.

---

## 0. Required operating constraints

1. **Terminal only.** No GUI, interactive webpage, `graph.html`, frontend libraries, browser automation, or Netron integration. The authoritative requirements are `AGENTS.md` and `docs/TERMINAL_V1_SPEC.md`.
2. **Read-only ONNX diagnostics.** Do not modify, simplify, delete nodes from, re-export, or overwrite the source ONNX model. Optimization **detection** is in scope; actual graph rewriting is not.
3. **No Docker, OpenExplorer, `hb_mapper`, quantization, board execution, CPU/BPU partition measurement, or additional hardware rules.** The RDK X5 Conv2D rules remain as in V1.
4. **No scores or rankings disguised as risk scores.** Sorting by objectively computed Tensor bytes is allowed; assigning arbitrary compatibility, risk, or optimization scores is not.
5. **Do not convert theoretical Tensor bytes into actual BPU SRAM, DDR footprint, memory allocation, latency, FPS, model deployability, or accuracy claims.** These require subsequent toolchain or runtime validation.
6. **No automatic deletion or model rewrites.** Every optimization candidate is a recommendation backed by a node pattern, evidence, preconditions, and verification steps.
7. **Do not replace the working V1 package or CLI.** Extend existing code. Existing `analyze`, `rules validate`, `nodes`, `inspect`, and `trace` commands must continue to work, including their `--json` behavior and exit codes.
8. **Never treat unsupported operators as BPU compatible.** Keep the V1 `NOT_COVERED` treatment of non-Conv operators. New graph patterns identify semantic or structural candidates, not new BPU operator compatibility decisions.
9. **Keep tests small, deterministic, and offline.** Do not commit large ONNX weights, user models, API keys, or environment-specific absolute paths.
10. **Preserve source/evidence traceability and uncertainty.** Each displayed candidate must refer to actual node IDs and Tensor names in `analysis.json`.

### What V1 already provides - reuse it

Verified from the repository's current `main` content at the time this document was prepared:

| Existing file | Existing capability | What V1.1 should do |
|---|---|---|
| `src/rdkx5_doctor/onnx_reader.py` | Loads ONNX, checks structure, infers shapes, builds node/Tensor dictionaries, avoids external-weight materialization | Extend metadata carefully; preserve reader safeguards |
| `src/rdkx5_doctor/graph_ir.py` | `GraphIR` dataclass and Tensor-labelled NetworkX directed topology | Use as the single graph representation; do not create a second conflicting graph model |
| `src/rdkx5_doctor/conv_extractor.py` | Derives Conv parameters for X5 checks | Leave behavior unchanged |
| `src/rdkx5_doctor/rules.py` | Validated YAML rules + Conv2D checks and conservative unknown handling | Do not break or widen rule semantics |
| `src/rdkx5_doctor/trace.py` | Predecessors, successors, reachable outputs, BFS representative paths | Reuse for node/candidate evidence when needed |
| `src/rdkx5_doctor/report.py` | `analysis.json` facts, Markdown report | Append resource and candidate sections |
| `src/rdkx5_doctor/cli.py` | Terminal analysis/search/inspect/trace; reads saved JSON for queries | Add terminal subcommands without breaking old arguments |
| `.agents/skills/rdk-x5-onnx-doctor/SKILL.md` | Repository-scoped Codex instructions | Extend the workflow to consume new facts, not invent new measurements |
| `docs/DEVELOPMENT_LOG.md` | V1 build/test history | Add a new V1.1 validation section with commands and observed results |

The V1 development log records **47 passing tests**. This is a *historical baseline reported by the repository*, not a result from this document. Run and record the current baseline independently before changing code.

---

## 1. V1.1 deliverables and acceptance definition

### 1.1 Module A: Tensor resource analysis

For **model outputs** and **intermediate Tensors**, calculate where possible:

- Tensor name, dtype, original shape (including symbolic or unknown dimensions).
- A non-duplicated classification: model output, intermediate activation, model input, initializer, Constant node output, or unclassified.
- Exact number of elements for entirely static shapes.
- Exact raw logical payload bytes from element count and supported dtype width.
- Human-readable MiB (`1 MiB = 1,048,576 bytes`) and optionally decimal MB (`1 MB = 1,000,000 bytes`) with explicit units.
- A separately labeled **hypothetical INT8 dense payload** for eligible numeric tensors with known shape; never present it as a real quantization result.
- Producer node ID, consumer node IDs, consumer count, whether the Tensor is also a graph output.
- Largest known intermediate activations and largest known model outputs, sorted by **raw logical bytes**.
- Unknown values and precise reasons; do not infer missing symbolic dimensions.
- Separate initializer/weight theoretical sizes from intermediate activation sizes.

**Output:** Structured resource data embedded in `analysis.json`, new Markdown report section, terminal resource queries.

### 1.2 Module B: graph optimization candidate detection

Detect and explain these **five** patterns, without changing the ONNX:

1. **Identity node** whose output directly forwards its input.
2. **Two consecutive inverse Transpose nodes** whose permutation composition is identity.
3. **Same-dtype Cast** (input dtype is demonstrably identical to Cast destination dtype).
4. **No-op Reshape** whose fully resolved target shape is identical to the fully known input shape.
5. **Conv followed by inference BatchNormalization** as a **fusion-review candidate**, subject to structural and semantic checks; never claim that it has already been fused by a compiler.

Additional patterns (for example, constant-only arithmetic folding, redundant Squeeze/Unsqueeze, algebraic Add/Mul rewrites) are **out of scope** until this set passes tests.

Every candidate needs:

- Stable candidate ID and pattern type.
- Original node IDs + original names, involved Tensor names, and adjacency evidence.
- Explanation of why it was recognized.
- Explicit **conditions for applying a rewrite** and whether a graph boundary, shared output, unknown dtype/shape, or unsupported operator semantics prevents safe automation.
- Classification as locally redundant, review-required fusion, or insufficient information; **not a score**.
- Likely structural benefit stated conservatively (for example, potentially fewer graph operations); **no unmeasured performance gain**.
- A verification procedure that would be required before any future rewrite (ONNX checker, shape checks, output interface, ONNX Runtime numerical comparison where applicable).
- Optional reused reachability context: which model outputs are downstream; this is a data-dependency fact, not proof of performance impact.

**Output:** Candidate list and evidence embedded in `analysis.json`, new Markdown report section, terminal candidate queries.

### 1.3 V1.1 acceptance in one sentence

Given a valid ONNX file, the existing terminal `analyze` command must produce the original Conv2D diagnostics **plus** trustworthy theoretical Tensor size details and identifiable, non-destructive graph optimization candidates; all results must be inspectable from the saved JSON using terminal commands, without Docker or model mutation.

---

## 2. Implementation plan: stages P7 through P11

| Stage | Objective | Files | Exit condition |
|---|---|---|---|
| P7 | Baseline + interface planning | existing files, tests, short notes | V1 baseline recorded and backward compatibility contract fixed |
| P8 | Tensor raw resource analyzer | `tensor_resource.py`, optional reader helper, tests | Exact bytes, categories, unknowns, dedup, sorting work |
| P9 | Candidate pattern analyzer | `optimization_candidates.py`, optional bounded constant resolver, tests | Five patterns detected conservatively; branch/boundary checks work |
| P10 | Reports, CLI, Skill integration | `report.py`, `cli.py`, `SKILL.md`, README | Saved-JSON terminal queries and readable reports work |
| P11 | Regression, packaging, demo, developer log | tests, examples, docs | All old/new tests and independent wheel install pass |

Do not implement every stage in one uncontrolled edit. Complete, test, and briefly summarize each stage before continuing to the next.

---

## 3. P7: baseline inspection and safety checks

From the repository root:

```bash
pwd
git status --short
git branch --show-current
.venv/bin/python -m rdkx5_doctor --help
.venv/bin/python -m rdkx5_doctor rules validate
.venv/bin/pytest -q
```

If `.venv` is unavailable, follow the existing README/AGENTS instructions instead of installing packages globally:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

Record the branch, uncommitted modifications, test count, failures (if any), and exact environment before editing. **Do not reset or overwrite pre-existing user modifications.** If baseline tests fail, determine whether it is an environmental failure or an existing code regression; do not silently claim an initially clean baseline.

Read the whole of `onnx_reader.py`, `graph_ir.py`, `trace.py`, `report.py`, `cli.py`, and test fixtures. Check `AGENTS.md`, `docs/TERMINAL_V1_SPEC.md`, and the current `SKILL.md` for conflicts. This document explicitly applies only to the additional V1.1 scope and does not revoke earlier terminal-only constraints.

**Interface decision:** V1 uses `analysis.json` with `schema_version="1.0"`. V1.1 shall emit `"1.1"` and add new top-level sections, without renaming/removing existing keys. The JSON reader must accept both `"1.0"` and `"1.1"` for existing `nodes`, `inspect`, and `trace` commands; new resource/candidate commands must fail clearly with an instruction to rerun `analyze` when given an old `1.0` file that lacks the corresponding sections. Tests must enforce this.

Existing CLI exit semantics must remain unchanged: `0` for successful analysis/query (even when violations or no matches exist); `2` for invalid input, invalid rules, invalid reports, or invalid query arguments.

---

## 4. P8: Tensor resource analysis - data extraction

### 4.1 Add a self-contained module

Create:

```text
src/rdkx5_doctor/tensor_resource.py
```

Suggested entry point:

```python
def analyze_tensor_resources(ir: GraphIR) -> dict:
    """Return JSON-serializable resource facts derived from GraphIR."""
```

Do not open the model weights a second time. Use the existing `GraphIR.tensors`, `GraphIR.nodes`, and `GraphIR.model` objects. Do not compute size from Python object overhead, serialized ONNX file bytes, or NumPy array allocation.

### 4.2 Tensor inclusion and unique classification

The existing reader provides Tensor fields similar to:

```json
{
  "name": "features",
  "shape": [1, 64, 320, 320],
  "dtype": "float32",
  "producer": "main/node_000017",
  "consumers": ["main/node_000018", "main/node_000031"],
  "kind": "intermediate",
  "is_graph_input": false,
  "is_graph_output": false,
  "is_initializer": false
}
```

Keep all original flags. Add a **derived** `resource_category` using an unambiguous precedence so a Tensor appears once in aggregate statistics:

1. `is_initializer == true` -> `initializer` (unless explicitly designated as an exported model output, in which case use one primary category and list both flags; explain the unusual overlap).
2. `is_graph_output == true` -> `model_output`.
3. `is_graph_input == true` -> `model_input`.
4. `kind == "constant"` -> `constant`.
5. `producer != null` -> `intermediate_activation`.
6. Otherwise -> `other_unknown`.

Do not classify all `graph.output` values as intermediates just because they have a producer. Conversely, a graph-output Tensor can also have consumers elsewhere; keep its `is_graph_output` flag and those consumer IDs without double counting its payload in categories.

Use names as keys to deduplicate Tensors, but preserve stable producer/consumer **node IDs** for traceability. Retain the original Tensor name; never derive graph connections by matching node display names.

### 4.3 Exact element-count calculation

For a `shape` list of length `N`, calculate:

```python
num_elements = math.prod(shape)
```

**Only** if every dimension is a Python integer >= 0 (bool is not a valid dimension). Cases:

| Shape | `num_elements` | Status |
|---|---:|---|
| `[1, 64, 320, 320]` | `6553600` | Known |
| `[1, 84, 8400]` | `705600` | Known |
| `[]` | `1` | Known scalar Tensor |
| `[1, 0, 320]` | `0` | Known empty Tensor |
| `["batch", 64, 80, 80]` | `null` | Unknown: symbolic dimension |
| `[null, 64, 80, 80]` | `null` | Unknown: unresolved dimension |
| `null` | `null` | Unknown rank and/or shape |
| `[-1, 64]` | `null` | Malformed or unresolved: do not multiply |

**Important:** A zero dimension does not justify returning zero bytes if another dimension is unresolved. Treat such a shape as unknown unless the overall shape can be proven; avoid silently converting uncertainty into a definite result.

Use Python integers (not bounded `int32` or `numpy.int64`) to avoid integer overflow in element counts. Never allocate an array of the computed size.

### 4.4 Element-size map and unsupported types

Use a transparent and explicitly bounded dtype map. At minimum support:

| Normalized dtype | Logical dense bytes / element |
|---|---:|
| `float16`, `bfloat16` | 2 |
| `float32` | 4 |
| `float64` | 8 |
| `int8`, `uint8` | 1 |
| `int16`, `uint16` | 2 |
| `int32`, `uint32` | 4 |
| `int64`, `uint64` | 8 |
| `bool` | 1 **assumed logical byte representation** |

Optionally support `complex64` (8) and `complex128` (16) when verified against the ONNX dtype representation. For `string`, packed 2/4-bit types, sparse/sequence/map/optional, and other unfamiliar types, return `raw_bytes = null` with `UNKNOWN_OR_UNSUPPORTED_DTYPE`; do not assume a full byte or fabricate packing rules.

The ONNX reader currently normalizes most types using the enum name and special-cases floats. Reuse or extend that normalization in **one place**; do not create different dtype conventions in the resource analyzer and Cast detector.

Calculate:

```python
raw_bytes = num_elements * bytes_per_element
mib = raw_bytes / (1024 ** 2)
mb = raw_bytes / (1000 ** 2)
```

Prefer `raw_bytes` as the authoritative value in JSON; derived decimal values can be rounded **for display only**. Do not round before aggregation or sorting.

Worked test oracle:

```text
Shape: [1, 64, 320, 320]
Elements: 6,553,600
FP32 raw payload: 26,214,400 bytes = 25.00 MiB
Hypothetical dense INT8 payload: 6,553,600 bytes = 6.25 MiB
```

### 4.5 Hypothetical INT8 scenario

For eligible, numeric, densely represented Tensor shapes with known element counts, compute:

```python
hypothetical_int8_bytes = num_elements * 1
```

Display the heading exactly as a **hypothetical INT8 raw-payload scenario** or a clearly equivalent phrase. It is **not**:

- Evidence that this Tensor is quantized to INT8.
- Evidence that the compiler assigns it to BPU.
- Evidence that the runtime consumes fewer bytes.
- A forecast of quantization error, DDR use, FPS, or peak memory.

Do not present a 4x compression prediction for the *whole model* based on FP32 to INT8 Tensor arithmetic. Mixed precision, channel padding, tiling, fusion, internal formats, and aliasing are unknown.

### 4.6 Producer, consumers, fanout

For each Tensor, copy:

- `producer_node_id` (nullable).
- `consumer_node_ids` (deduplicated, stable ordering).
- `consumer_count`.
- `is_graph_output`, `is_initializer`, and other original classification flags.

Optionally annotate `shared_by_multiple_consumers` when consumer_count >= 2. Treat fanout as a **graph fact**, not automatically as poor performance or additional memory allocations.

### 4.7 Aggregates, unknowns and important limitations

Produce separate, descriptive aggregates:

- Sum of known raw logical payload bytes for **model outputs**.
- Sum of known raw logical payload bytes for **intermediate activations**.
- Sum of known raw logical payload bytes for **initializers**.
- Number of members with unknown sizes per category.
- Largest 10 known intermediates by raw bytes; largest 10 known outputs by raw bytes.
- Number of multi-consumer intermediates; optionally list the largest.

For any aggregate with unknown members, record `completeness="PARTIAL"`; otherwise `"COMPLETE"`. Use a field name like `known_bytes_sum`, **not** `total_memory` or `peak_memory`.

Never add the initializer sum and activation sum and call the result real peak RAM. Activation lifetimes may overlap or be reused; compiler optimization and aliasing are unknown.

**Not in V1.1:** true live-range allocation, static peak SRAM/DDR, on-chip scratchpad, layout/padding-aware memory, Tensor reuse scheduling, `hb_perf`, latency, and real CPU/BPU placement.

### 4.8 JSON resource schema

The exact shape may be adjusted when integrating with the codebase, but preserve the following semantics and field names wherever feasible:

```json
{
  "resource_analysis": {
    "schema_version": "1.0",
    "units": {"bytes": "B", "mib_divisor": 1048576},
    "tensor_records": [
      {
        "name": "features",
        "resource_category": "intermediate_activation",
        "shape": [1, 64, 320, 320],
        "dtype": "float32",
        "element_count": 6553600,
        "element_size_bytes": 4,
        "raw_bytes": 26214400,
        "mib": 25.0,
        "hypothetical_int8_bytes": 6553600,
        "size_status": "KNOWN",
        "unknown_reason": null,
        "producer_node_id": "main/node_000017",
        "consumer_node_ids": ["main/node_000018", "main/node_000031"],
        "consumer_count": 2,
        "is_graph_output": false
      }
    ],
    "summary": {
      "model_outputs": {
        "known_bytes_sum": 0,
        "unknown_tensor_count": 0,
        "completeness": "COMPLETE"
      },
      "intermediate_activations": {
        "known_bytes_sum": 26214400,
        "unknown_tensor_count": 0,
        "completeness": "COMPLETE"
      },
      "initializers": {
        "known_bytes_sum": 0,
        "unknown_tensor_count": 0,
        "completeness": "COMPLETE"
      },
      "largest_intermediates": ["features"],
      "largest_outputs": [],
      "limitations": [
        "Logical Tensor payload only; not measured runtime/BPU/DDR memory."
      ]
    }
  }
}
```

Use `null`, not magic values such as `-1` or zero, for unknown `raw_bytes`/counts. `size_status` must identify unknown shape and unsupported dtype separately. Validate generated JSON with a test and ensure `allow_nan=False` still succeeds.

---

## 5. P9: graph-based optimization candidate discovery

### 5.1 New analyzer

Create:

```text
src/rdkx5_doctor/optimization_candidates.py
```

Suggested entry point:

```python
def detect_optimization_candidates(ir: GraphIR, *, constants=None) -> list[dict]:
    """Detect and explain non-destructive graph pattern candidates."""
```

Data source:

- `ir.nodes`: stable IDs, operator names, domains, input/output Tensors, attributes.
- `ir.tensors`: producer, consumer lists, output flags, dtype, shape.
- `ir.topology()`: directed node adjacency with Tensor names on edges.
- Small shape constants from an explicitly bounded resolver (see section 5.6).

Use Tensor connectivity rather than node-list adjacency. Do not infer relationships from substring matches in node names such as `/model.3/conv`.

### 5.2 Common detector rules

- Only apply standard ONNX operator semantics when `domain` is `""` or a verified standard ONNX alias. Custom-domain operators are not automatically equivalent.
- Ensure the applicable operator version and its attributes have understood semantics. Unsupported or ambiguous opset behavior -> review/unknown, not proof.
- Never claim rewrite safety solely from matching `op_type` strings.
- When examining a chain `A -> B`, explicitly confirm a real Tensor output of A is the corresponding input of B.
- Distinguish ordinary intermediate edges from graph-input/output interface contracts.
- Detect fanout and multiple outputs. A local transform may be valid on one path but **not equivalent to deleting the whole upstream node**.
- Deduplicate by `(pattern_type, ordered stable node IDs, relevant Tensor names)`.
- Use deterministic ordering (ONNX node order, then pattern type) so candidate IDs and tests are reproducible for a given unchanged model.
- Keep model output visibility optional/lazy: no exponential path enumeration; reuse `trace_node` only for nominated candidates or provide a bounded on-demand query.
- It is fine for multiple candidate patterns to overlap. Mark the overlap and avoid adding up supposed savings as if all candidates could be applied at once.
- Do not import or call ONNX graph-rewrite libraries for V1.1.

### 5.3 Pattern OPT-IDENTITY

Recognize: standard `Identity` with one actual data input and one output.

Evidence:

```text
input Tensor -> Identity(node ID) -> output Tensor
```

Its **value transformation** is identity. Still record whether removing the node would impact output names, graph interface contracts, fanout consumers, or downstream metadata.

Classification:

- Ordinary internal Identity: `SEMANTICALLY_REDUNDANT` (candidate only, not actually deleted).
- Identity directly responsible for a public graph output: still a candidate but `REVIEW_REQUIRED` because output naming/ABI must be preserved.

Do not require dtype or shape inference to explain Identity semantics, but report missing metadata where relevant to future interface-preserving rewrites.

### 5.4 Pattern OPT-TRANSPOSE-INVERSE

Recognize `Transpose(A) -> Transpose(B)` connected by A's output Tensor into B's data input.

First determine each permutation:

- Explicit `perm`: ensure it is a valid permutation of `[0..rank-1]`.
- Missing `perm`: ONNX defaults to reverse-axis order, but only use that default when the input rank is known.

The effective composite permutation is:

```python
composite = [perm_a[perm_b[i]] for i in range(rank)]
is_identity = composite == list(range(rank))
```

Example:

```text
perm_a = [0, 2, 3, 1]
perm_b = [0, 3, 1, 2]
composite = [0, 1, 2, 3]
```

Only if the composite is identity should a candidate be reported as an inverse pair. If rank/perm cannot be determined, do not incorrectly report a proven pair. Optionally record a `REVIEW_REQUIRED` near-match with an explicit missing-fact explanation.

For a simple fully redundant pair, additionally check:

- Intermediate Tensor is **not** a graph output.
- Intermediate Tensor has exactly one consumer: B.
- A has no other required consumers/outputs that would prevent removing the whole A/B pair.

If fanout exists or A's output is externally visible, report only a **conditional/local candidate**; never claim that both nodes can be safely deleted globally. If B's output is a model output, preserve its public name in any future rewrite.

### 5.5 Pattern OPT-CAST-SAME-DTYPE

Recognize standard `Cast` whose required `to` attribute maps to the *same* normalized dtype as its input Tensor. Check supported ONNX opset semantics before treating the node as a no-op.

For example:

```text
input dtype = float32
Cast(to = TensorProto.FLOAT)
output dtype = float32
```

Use one dtype-normalization function shared with `onnx_reader.py`. Do not compare raw enum integer and string without normalization. For unfamiliar/packed/string/custom types, leave as unknown/review rather than claiming equivalence. Preserve graph output names in any hypothetical rewrite plan.

**Negative example:** float32 -> int8 is not redundant, even if a downstream Cast converts it back.

### 5.6 Pattern OPT-RESHAPE-NOOP: bounded constant resolution

Do not equate `Reshape` with a no-op merely because input and output Tensor element counts match: **all valid Reshape operations preserve element count**.

Only recognize a definite no-op when:

1. The input Tensor shape is fully known (all non-negative dimensions).
2. The Reshape target is a **known constant** shape vector.
3. Target semantics are resolved correctly for the model's ONNX opset (`0` copying semantics, optional `allowzero`, a single `-1`, and valid element-count conservation).
4. The resolved target shape is **exactly equal** to the input shape and compatible output metadata.
5. No ambiguity remains about scalar/empty dimensions, invalid `allowzero`/`-1` combinations, or dynamic values.

The current reader stores Constant Tensor **metadata**, not their integer values. Implement a narrow, safe helper only for tiny shape vectors:

```text
src/rdkx5_doctor/small_constants.py   # optional, if needed
```

Read only inline standard ONNX integer initializer values and Constant-node Tensor values whose element count is <= 64 and whose encoded content is within a small fixed byte budget. Never materialize arbitrary full model weights or external data. Reject external-data paths, oversized tensors, unsupported types, and uncertain encodings with a specific reason. You may use `onnx.numpy_helper.to_array` **only after** enforcing a bounded value count and storage check; do not use it on large floating-point model weights.

Rules for resolving target shapes should follow the ONNX `Reshape` operator version imported by the model. If insufficiently verified for that version, leave a review-required observation rather than asserting a no-op.

Examples:

| Input | Target data | Context | Candidate? |
|---|---|---|---|
| `[1, 64, 32, 32]` | `[1, 64, 32, 32]` | Const target | Yes |
| `[1, 64, 32, 32]` | `[0, 64, 32, 32]` | `allowzero=0`, correctly resolved | Yes |
| `[1, 64, 32, 32]` | `[-1, 64, 32, 32]` | Valid known shape, resolves to 1 | Yes |
| `[1, 64, 32, 32]` | `[1, 65536]` | Different shape | No |
| `["N", 64, 32, 32]` | `[-1, 64, 32, 32]` | Symbolic first dimension | Unknown, no proven candidate |
| `[1, 64, 32, 32]` | output of dynamic `Shape` computation | Not a small constant | Unknown, no proven candidate |

Do not implement a general ONNX constant-propagation engine in V1.1.

### 5.7 Pattern OPT-CONV-BN-FUSION-REVIEW

Recognize a standard `Conv` immediately followed along one Tensor edge by a standard `BatchNormalization` with inference behavior.

Report only a **fusion-review opportunity**, not a declaration that the two ONNX nodes may always be safely deleted or that an X5 compiler has not already fused them.

Required considerations:

- Confirm BN's data input comes from the Conv's output.
- Verify no intervening operator.
- Check BN inference mode against its **version-specific** attributes (`training_mode` for newer versions; older versions may use different conventions). Training mode -> do **not** classify as ordinary inference fusion.
- Prefer one consumer of Conv output and no public graph-output boundary in the middle; otherwise record a conditional/fanout blocker.
- Verify shape/channel metadata and the presence of scale, bias, mean, variance parameters where possible. If actual parameter values are unavailable, record that limitation.
- Do not assume group convolution or mixed-precision folding is safe without validation.
- No actual BN parameter math or new model export is required for V1.1.

Optional explanation for a later version (not to be executed now): inference BN parameters can sometimes be folded into Conv weights and bias with formulas using `gamma`, `beta`, `mean`, `variance`, and `epsilon`. This is a future rewrite/verification topic, not a measured performance benefit.

### 5.8 Candidate JSON schema

Append a top-level `optimization_candidates` section to analysis data. Example:

```json
{
  "optimization_candidates": {
    "schema_version": "1.0",
    "candidates": [
      {
        "candidate_id": "OPT-0001",
        "pattern": "TRANSPOSE_INVERSE_PAIR",
        "node_ids": ["main/node_000010", "main/node_000011"],
        "node_names": ["transpose_nchw_to_nhwc", "transpose_back"],
        "tensor_names": ["feature_nhwc", "feature_nchw"],
        "classification": "SEMANTICALLY_REDUNDANT",
        "evidence": {
          "perm_a": [0, 2, 3, 1],
          "perm_b": [0, 3, 1, 2],
          "composite_perm": [0, 1, 2, 3],
          "intermediate_consumer_count": 1,
          "intermediate_is_graph_output": false
        },
        "conditions": [
          "Preserve graph output names and all external Tensor interfaces"
        ],
        "blockers": [],
        "benefit_hint": "May reduce redundant data-layout transformations if retained in the compiled graph",
        "verification_needed": [
          "Validate the proposed graph with ONNX checker",
          "Compare outputs on representative inputs before and after a future rewrite"
        ],
        "reachable_outputs": ["prediction"],
        "limitations": [
          "No rewrite performed",
          "No compiler fusion or performance data available"
        ]
      }
    ],
    "summary": {
      "candidate_count": 1,
      "counts_by_pattern": {"TRANSPOSE_INVERSE_PAIR": 1}
    }
  }
}
```

Use candidate classifications such as:

- `SEMANTICALLY_REDUNDANT` for local operations whose no-op semantics have been established in the supported scenario (still **not automatically rewritable**).
- `REVIEW_REQUIRED` for pattern-based fusion or interface/fanout complications.
- `INSUFFICIENT_INFORMATION` for a near-match requiring missing facts, **only if** intentionally emitted and clearly separated from confirmed candidates.

No free-form risk scores, numerical confidence, pretend SRAM savings, or fabricated throughput improvements.

### 5.9 Evidence and output tracing

For confirmed/review candidates, include actual node/tensor evidence from GraphIR and optionally which model outputs are reachable from the last node. Prefer a bounded on-demand `candidate` CLI detail lookup that reuses existing `trace_node`. Avoid computing potentially expensive full shortest-path traces for every node in a large YOLO graph during basic analysis.

When showing reachable outputs:

```text
Candidate: OPT-0004 (IDENTITY)
Nodes: main/node_000027
Related Tensor: skip_features
Reachable outputs: prediction, auxiliary
```

These output relationships are **data dependence**, not evidence that removing the node will improve accuracy, memory, or performance.

---

## 6. P10: report, CLI and Skill integration

### 6.1 Keep the existing report interface

In `src/rdkx5_doctor/report.py`:

1. Continue producing every V1 top-level key: `model`, `ruleset`, `nodes`, `tensors`, `edges`, `diagnostics`, `traces`, `summary`, `limitations`, `unverified_assumptions`.
2. Set the new overall `schema_version` to `"1.1"`.
3. Add `resource_analysis` and `optimization_candidates` as additional top-level sections.
4. Keep V1 Conv2D diagnostic results and node IDs unchanged for the same input model.
5. Append Markdown sections after the existing diagnostic/trace content (or reorganize headings carefully without dropping facts):
   - `Tensor Resource Analysis`: output sizes, top intermediates, unknowns, shared-consumer facts, initializer totals, units.
   - `Graph Optimization Candidates`: pattern, node IDs/names, involved tensors, evidence, constraints, blockers, validation requirements.
6. **Avoid drowning the report in hundreds of normal Tensor rows.** Default report: summaries and top 10; full facts belong to JSON and the `tensors` CLI query.
7. Continue HTML-escaping/terminal-escaping untrusted strings where appropriate so adversarial model names do not inject markup/terminal control sequences.

### 6.2 New CLI commands

Add to the existing CLI without replacing old commands:

```bash
# Analyze ONNX (existing command; now includes both new modules)
.venv/bin/python -m rdkx5_doctor analyze \
  --model examples/demo.onnx \
  --out reports/demo-v1_1

# List Tensor resources from saved JSON
.venv/bin/python -m rdkx5_doctor tensors \
  --analysis reports/demo-v1_1/analysis.json \
  --kind intermediate --sort bytes --limit 10

# List model outputs only
.venv/bin/python -m rdkx5_doctor tensors \
  --analysis reports/demo-v1_1/analysis.json \
  --kind output --json

# Search for a specific Tensor and show full details
.venv/bin/python -m rdkx5_doctor tensor \
  --analysis reports/demo-v1_1/analysis.json \
  --name prediction --json

# List optimization candidates
.venv/bin/python -m rdkx5_doctor candidates \
  --analysis reports/demo-v1_1/analysis.json

# Filter candidate patterns
.venv/bin/python -m rdkx5_doctor candidates \
  --analysis reports/demo-v1_1/analysis.json \
  --pattern IDENTITY --json

# Candidate details and optional reachable outputs
.venv/bin/python -m rdkx5_doctor candidate \
  --analysis reports/demo-v1_1/analysis.json \
  --id OPT-0001
```

Command names can follow the above exactly; do not invent an incompatible parallel CLI. Recommended `tensors --kind`: `all`, `output`, `intermediate`, `input`, `initializer`, `constant`, `unknown`; default `all`. Recommended `--sort`: `bytes` (unknown last), `name` (stable alphabetic). Support `--json` on **every new read command**, so the Skill can programmatically consume facts via pipes. Do not silently suppress unknown members in human output: show a final unknown-size count and a command to inspect them.

If `tensors` or `candidates` receives an old V1 `analysis.json` without the required section, print a clear error instructing the user to rerun `analyze` with V1.1; exit code `2`. Old V1 `nodes`, `inspect`, and `trace` must still accept V1 JSON.

No new command should need to load ONNX weights to answer a query; all post-analysis queries read **saved `analysis.json`**.

### 6.3 Terminal format expectations

For a Tensor listing, show objective fields:

```text
Tensor name           category       shape                dtype    raw MiB  consumers
features              intermediate   [1,64,320,320]       float32  25.00    2
prediction            output         [1,84,8400]         float32   2.69    0
...
Unknown-size Tensors: 3 (symbolic dimensions or unsupported dtype)
Note: logical raw Tensor payload, not actual RDK X5 BPU/DDR allocation.
```

The table above is a **format illustration**, not a single real model's factual output. The full saved JSON must contain **exact bytes** even if the table displays rounded MiB.

For candidate details, show: ID, pattern, node IDs, Tensor names, recognition evidence, blockers, downstream outputs (when calculated), what future verification requires, and the fact that **no model rewrite was performed**.

### 6.4 Upgrade the Codex Skill

Edit the **existing** `.agents/skills/rdk-x5-onnx-doctor/SKILL.md`; do not introduce a second Skill with overlapping triggers.

Extend the frontmatter description to cover new requests such as:

- "Analyze ONNX output Tensor size and large feature maps."
- "Find redundant nodes, inverse Transpose, no-op Reshape."
- "Identify optimization candidates before quantizing for RDK X5."

Required Skill workflow:

1. Resolve the user's model path and selected ruleset; confirm read-only mode.
2. Validate ruleset and run a new `analyze` into a fresh report directory.
3. Read the saved JSON and report; explicitly distinguish computed resource facts from unknowns.
4. If the user asks about outputs/memory, run `tensors` and `tensor`, identify the largest **known** output/intermediate Tensors, trace selected producers/consumers when useful.
5. If the user asks about optimization, run `candidates` and `candidate`, then use `nodes`/`inspect`/`trace` to verify source nodes, fanout, and path-to-output evidence.
6. Explain suggested changes conditionally: which ONNX nodes, why they are candidates, interface concerns, verification requirements. Do not claim that the toolchain has or has not already fused nodes.
7. Provide `analysis.json`, `report.md`, and reproducible terminal commands. Refuse to invent performance/accuracy data.

**What distinguishes the Skill from the Python tool:** Codex decides which of the existing deterministic inspections to perform next based on findings, cross-checks evidence, and explains the engineering tradeoffs. It must not just paste generic advice about every operator or fabricate a rule not in the checked corpus.

### 6.5 Documentation updates

Update:

- `README.md` with examples and limitations for the new commands.
- `docs/DEVELOPMENT_LOG.md` with stage-by-stage observed results and known limitations.
- If a schema becomes nontrivial, add `docs/ANALYSIS_SCHEMA_V1_1.md` describing required/optional fields and V1 compatibility.
- Update `AGENTS.md` only if necessary; do **not** weaken terminal-only, read-only, and no-Docker requirements.
- Keep `docs/TERMINAL_V1_SPEC.md` as historical V1 specification; optionally link this new V1.1 document instead of rewriting history.

---

## 7. P11: test design and expected oracles

### 7.1 General testing instructions

Use `pytest` and small generated ONNX models built with `onnx.helper`, saved under temporary pytest paths. No production YOLO weights need to be checked into Git.

Add at minimum:

```text
tests/test_tensor_resource.py
tests/test_optimization_candidates.py
tests/test_cli_v1_1.py
```

Prefer parametrized tests for dtype widths, static/dynamic shapes, and graph pattern permutations. Use an external data/shape fixture when testing the small-constant resolver. Existing V1 tests must all continue to pass.

### 7.2 Tensor resource cases (required)

| Test ID | Input condition | Expected behavior |
|---|---|---|
| TR-01 | `float32 [1,64,320,320]` | `6553600` elements, `26214400` B, `25.00` MiB |
| TR-02 | Same shape, hypothetical INT8 | `6553600` B clearly labeled hypothetical |
| TR-03 | `float16`, `bfloat16`, `int8`, `int32`, `int64` | Correct per-element widths and exact byte counts |
| TR-04 | Shape `[]` | Scalar has one element |
| TR-05 | Shape contains `0` and all dims static | Zero payload bytes |
| TR-06 | Symbolic dimension or `None` | `raw_bytes=null` and useful reason |
| TR-07 | Missing shape | Unknown, not zero |
| TR-08 | Unsupported/packed/string dtype | Unknown width, not guessed |
| TR-09 | Initializer `weight` and intermediate feature | Report distinct categories; no double-counted activation total |
| TR-10 | Output Tensor is also consumed elsewhere | Single aggregate inclusion, retained consumer list |
| TR-11 | Multi-consumer intermediate | Correct fanout count, no memory/performance claim |
| TR-12 | Unknown-size member in a category | `completeness=PARTIAL` and known-only bytes sum |
| TR-13 | Large dimension product | No integer overflow or memory allocation proportional to count |
| TR-14 | ONNX with external data missing | Retain existing warning and do not read external weights |

### 7.3 Candidate detection cases (required)

| Test ID | Graph fixture | Expected result |
|---|---|---|
| OP-01 | `X -> Identity -> Y` internal | `IDENTITY` candidate with node/Tensor evidence |
| OP-02 | `Identity` produces public model output | Candidate requires output interface review |
| OP-03 | Inverse `Transpose[0,2,3,1] -> Transpose[0,3,1,2]` | Confirmed inverse pair |
| OP-04 | Non-inverse Transpose pair | **No confirmed inverse pair** |
| OP-05 | Omitted perm and known rank | Apply ONNX reverse-dims default and check composition |
| OP-06 | Omitted perm and unknown rank | Unknown/review, not confirmed |
| OP-07 | Inverse pair whose intermediate has two consumers | Must flag fanout; no claim both nodes globally removable |
| OP-08 | `Cast float32 -> float32` | Same-dtype Cast candidate |
| OP-09 | `Cast float32 -> int8` | Not a redundant Cast |
| OP-10 | No-op Reshape with known constant target | Confirmed no-op candidate |
| OP-11 | Reshape with changed shape but equal element count | Not a no-op |
| OP-12 | Reshape `0` and `-1` semantics | Correct version-aware resolution, no false positives |
| OP-13 | Reshape dynamic target or unresolved symbolic input | No confirmed no-op candidate |
| OP-14 | Conv -> inference BN, one consumer | Fusion-review candidate, not 'proven already fused' |
| OP-15 | Conv -> BN training mode | Do not label ordinary inference fusion |
| OP-16 | Conv output fanout to BN and another node | Record structural blocker/review |
| OP-17 | Custom-domain same operator name | No assumed standard semantics |
| OP-18 | Duplicate original node names | Stable internal IDs, no ambiguity |
| OP-19 | Two branches merging then two model outputs | Reachability/candidate output context correct |
| OP-20 | Overlapping patterns | Deterministic candidate IDs and explicit overlap; no summed savings |

### 7.4 CLI and report compatibility cases

- `analyze` still produces `analysis.json` and `report.md` in a new output directory.
- V1 Conv2D rule IDs and diagnostics unchanged for the existing demo fixture.
- New V1.1 JSON has both additional sections and every V1 top-level key.
- New `tensors`, `tensor`, `candidates`, `candidate` commands print sane text and valid `--json` without loading ONNX again.
- Existing `nodes`, `inspect`, `trace` still process a saved `schema_version=1.0` report.
- New queries on a saved V1 report fail clearly with exit code 2 and 'rerun analyze' guidance.
- Unknown/missing Tensor and candidate IDs result in a clear error, not a traceback.
- Terminal output escapes untrusted model, node, and Tensor names; JSON output retains correct escaped strings.
- A model with no optimizable pattern outputs an empty list and a normal success result; it is not a program error.
- Original ONNX SHA256 unchanged before and after analysis; output collision and symlink safety tests still pass.
- No HTML, frontend asset, model rewrite, or Docker invocation appears in the code path.

### 7.5 Integration demo

Reuse `examples/demo.onnx`, then add a second generated mini-model designed to contain:

```text
Input -> Identity -> Transpose(A) -> Transpose(A^-1) -> Cast(same dtype)
                                      |
                                      v
                            Reshape(no-op) -> Conv -> BatchNormalization -> Output
```

Arrange valid Tensor shapes and actual edges to match the intended patterns; the diagram is conceptual, not an instruction to link unrelated operators artificially. Include at least one branch/fanout case in a separate generated fixture.

Suggested example artifacts:

```text
examples/generate_v1_1_demo.py
examples/v1_1_demo.onnx
examples/v1_1_demo-report/analysis.json
examples/v1_1_demo-report/report.md
```

If repository policy prefers generated reports ignored by Git, document regeneration commands rather than adding bulky files. Use tiny weights only.

After local verification, if a real YOLO ONNX is available **on the machine**, test parsing and report size/performance and record results. If not available, explicitly write **"real YOLO ONNX not tested"**; do not fabricate its node count, candidate count, or tensor sizes.

---

## 8. Full acceptance commands

Run from a clean or well-understood local working tree:

```bash
.venv/bin/python -m rdkx5_doctor --help
.venv/bin/python -m rdkx5_doctor rules validate
.venv/bin/pytest -q
.venv/bin/python -m build

.venv/bin/python examples/generate_demo.py
.venv/bin/python -m rdkx5_doctor analyze --model examples/demo.onnx --out reports/v1_1-demo
.venv/bin/python -m rdkx5_doctor tensors --analysis reports/v1_1-demo/analysis.json --kind output
.venv/bin/python -m rdkx5_doctor candidates --analysis reports/v1_1-demo/analysis.json
.venv/bin/python -m rdkx5_doctor nodes --analysis reports/v1_1-demo/analysis.json --search Conv
.venv/bin/python -m rdkx5_doctor inspect --analysis reports/v1_1-demo/analysis.json --node oversized_kernel
.venv/bin/python -m rdkx5_doctor trace --analysis reports/v1_1-demo/analysis.json --node oversized_kernel
```

Also install the built wheel into a fresh temporary virtual environment, change working directory outside the repository, and rerun `--help`, resource/candidate saved-JSON queries, and one `analyze`. This catches accidental reliance on editable install and uncopied package assets.

Verify:

- The **number of old passing tests** has not decreased due to regressions.
- New tests are actually executed and documented by name/count.
- All deterministic outputs are reproducible for the same unchanged ONNX input and ruleset.
- Input model bytes are unchanged.
- Report headings and JSON agree on candidate IDs, Tensor sizes, limitations and unknowns.
- No hardcoded model-specific names, paths, or fake Tensor sizes have been used.

---

## 9. Coding quality, safety and performance requirements

**Code organization:** Keep pure computations in new modules; use `report.py` to compose facts, `cli.py` for argument handling and text rendering, and `SKILL.md` for the Agent workflow. Do not move domain reasoning into a prompt when it can be computed deterministically.

**Memory handling:** ONNX `load_external_data=False` remains the default. Resource analysis should perform integer arithmetic over metadata, not Tensor materialization. Tiny integer constant decoding is the only permitted narrow exception for no-op Reshape detection, bounded before allocation.

**Complexity:** Avoid enumerating every graph path. A linear node/Tensor scan and bounded neighborhood lookups should dominate. Candidate output paths should be traced on demand or with explicit cost limits for very large graphs.

**Robustness:** Unknown dimensions, unsupported dtype, suspicious large dims, empty graph, custom domains, duplicate original node names, and missing external weights must not cause silent wrong conclusions. Fail with clear messages or propagate typed unknown facts.

**Security:** No `eval`/`exec` of untrusted ONNX strings, no arbitrary shell execution derived from model names, no unsafe external-data path traversal, no terminal ANSI injection in text views, no unbounded constant materialization.

**Documentation precision:** Use 'theoretical raw Tensor bytes', 'potential redundancy', and 'fusion-review candidate'. Do not use 'real peak memory', 'guaranteed deletable', 'BPU exceeds limit', '4x end-to-end speedup', or 'quantization accuracy improves' unless measured in future versions.

**Backward compatibility:** The existing Conv2D YAML remains canonical in `src/rdkx5_doctor/resources/rulesets/x5-bayes-e/`. Root `rulesets/` is a symlink. Avoid duplicate rule copies or introducing rules for Add/Mul/Reshape/Transpose in this version; structural optimization patterns are **not** X5 BPU compatibility rules.

---

## 10. Execution checklist for Codex

Mark each item complete **only after executing tests**:

- [ ] P7: Read repo instructions and source; capture baseline Git state, CLI help, `pytest` result.
- [ ] P7: Decide and document V1.0 -> V1.1 JSON read compatibility.
- [ ] P8: Implement exact static Tensor element/byte calculations and unknown handling.
- [ ] P8: Implement Tensor categories, output/intermediate summaries, fanout and top-10 known sizes.
- [ ] P8: Implement hypothetical INT8 raw-payload scenario with explicit caveats.
- [ ] P8: Pass all Tensor resource tests.
- [ ] P9: Implement Identity, inverse Transpose and same-dtype Cast detectors.
- [ ] P9: Implement safely bounded small-constant resolution and no-op Reshape detector.
- [ ] P9: Implement inference Conv->BatchNormalization fusion-review detection.
- [ ] P9: Implement stable evidence-rich candidate objects, deterministic IDs, fanout/interface blockers.
- [ ] P9: Pass candidate positive and negative tests.
- [ ] P10: Extend `analysis.json` and `report.md` without dropping old diagnostics.
- [ ] P10: Implement and test `tensors`, `tensor`, `candidates`, `candidate` terminal commands.
- [ ] P10: Update existing `SKILL.md` to orchestrate the new deterministic tools.
- [ ] P10: Update README and schema/reference docs.
- [ ] P11: Generate mini-model demo, run full `pytest` and `python -m build`.
- [ ] P11: Fresh venv wheel install and CLI smoke test outside repo.
- [ ] P11: Confirm read-only operation and unchanged source model hash.
- [ ] P11: Record what is tested vs not tested in `docs/DEVELOPMENT_LOG.md`.

### Required final handoff message from Codex

When development is done, report:

1. Files added/modified and what each does.
2. Example terminal commands to reproduce resource and optimization-candidate results.
3. Real outputs **from commands actually run**, including at least one exact Tensor byte count and one detected graph candidate from a test model.
4. Test summary and packaging result; separate historical V1 test numbers from actual new test results.
5. Known limitations and unsupported/unknown cases, explicitly including lack of Docker/hb_mapper validation.
6. Any departures from this specification, with their rationale.

Do **not** claim success if the tests were not actually executed.

---

## 11. Future work - explicitly not part of V1.1

- New X5 rules for Concat, Add, Mul, Reshape, Transpose, Resize, etc.
- AI-driven automatic ONNX rewrite, simplification, or operator substitution.
- ONNX Runtime equivalence test runner and before/after accuracy verification.
- Accurate compiler-aware memory planning or peak BPU/DDR estimates.
- OpenExplorer Docker adapter, `hb_mapper checker`, PTQ, actual precision/performance diagnostics.
- GUI, graph.html, browser display, interactive Netron/Cytoscape view.

The completed V1.1 should be a **trustworthy terminal-first, non-destructive static graph diagnostic tool**, with Codex Skill orchestration. Future versions can add more operator rules and toolchain validation without replacing its Tensor/candidate data model.

---

## 12. References for implementing ONNX semantics

These references explain ONNX operator behavior; they are **not** evidence of RDK X5 hardware constraints. Use the operator version imported by the actual model, not blindly the newest webpage version.

- ONNX model validation / shape inference: https://onnx.ai/onnx/api/checker.html and https://onnx.ai/onnx/api/shape_inference.html
- Transpose axes and default permutation: https://onnx.ai/onnx/operators/onnx__Transpose.html
- Reshape `0`, `-1`, `allowzero`: https://onnx.ai/onnx/operators/onnx__Reshape.html
- Cast `to` attribute: https://onnx.ai/onnx/operators/onnx__Cast.html
- Identity: https://onnx.ai/onnx/operators/onnx__Identity.html
- BatchNormalization inference/training modes: https://onnx.ai/onnx/operators/onnx__BatchNormalization.html
- Existing repository Conv2D source references: `references/official_sources.md` and `references/conv_rule_notes.md`.

**End of V1.1 development specification.**
