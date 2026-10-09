# Report policy 1.3

Extends 1.2 compact anomaly-first rendering. The JSON is complete; Markdown never conceals failed/unknown totals. Includes model basename/hash, source/opset/registry/subpack versions, coverage matrix, failures, blocking and nonblocking unknown groups, Shape before/after counts, proof examples, first blockers/conflicts, resources, MatMul/Softmax/Resize facts, candidates and optional architecture-only directions.

Limits: first50 violation nodes,10 representative nodes per unknown group,10 candidates,5 example recovered Tensors,10 attention/Resize nodes. Omitted counts are explicit. Group counts are unique within each reason; overlap is explicitly stated and is not a unique-node sum. Full proof and graph queries use saved JSON. Model strings are escaped for terminal controls, HTML and Markdown.

Original scalar Mul evidence is preserved; compiler normalization remains unknown. Softmax static eligibility is not run_on_bpu/placement confirmation. Fixed shape does not measure BPU/DDR allocation or peak memory. No large weights, feature contents, source-code locations or specific patches appear in advice. Missing shape facts usually require analyzer investigation, not architecture changes. Architecture directions, including output-head/postprocessing separation, are optional and belong in a separate owner-provided model project.

All model-specific Markdown/JSON/logs stay under ignored reports/. Generic development/schema/source documents may be tracked; no automatic commit/push.
