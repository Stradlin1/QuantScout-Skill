# V1.3 implementation review

Source links and exclusions: x5_attention_resize_sources.md. New packs use manual1.1.2 and retain complete node/Tensor evidence. No cross-manual disagreement was found in the three X5 ONNX rows; Caffe Softmax axis1/2/3 is deliberately excluded.

MatMul limits apply to the original input metadata, including ONNX-legal mismatched rank and zero-size counterexamples. ONNX K mismatch or illegal broadcasting is an independent semantic issue. The asymmetric leading-axis predicate enforces A all-broadcast or all-nonbroadcast, then B nonbroadcast prefix; symbolic cases are unknown unless a concrete contradiction is independently established. Rank-one input has no two-dimensional matrix lowering proof and remains review/unknown.

Softmax eligibility is separate from deployment: static rank/axis checks can pass while run_on_bpu remains nonblocking UNKNOWN. Opset11 flattens the suffix starting at axis; opset13 uses the specified axis directly. No PyTorch provenance is guessed.

Resize coordinate-mode hardware evidence explicitly names opset11; other versions do not borrow those conditions. NCHW requires a Conv/explicit layout proof, not rank alone. Nearest factors use exact finite binary floating-point values: no invented hardware tolerance. Enlargement-only restrictions do not prohibit shrinking. ROI crop boundaries use (dimension-1)*ROI coordinates; missing/dynamic/unsupported ROI remains unknown.

Shape and tiny-value semantics are analysis-only: helper operators remain NOT_COVERED. Values are bounded and only decoded in requested dependency cones; mutable/external data is never used. A known dimension conflict keeps both premises and does not overwrite the original fact. Recommendations remain optional architectural directions; no source location, training-code patch or ONNX rewrite is generated.
