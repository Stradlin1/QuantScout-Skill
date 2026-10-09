# V1.3 official source audit

Reviewed 2026-10-09. Only X5 **ONNX BPU** rows are authoritative. Primary pack source: [X5 manual 1.1.2](https://developer.d-robotics.cc/x5_sdk_doc/toolchain_development/intermediate/supported_op_list.html), section 6.3.3.3. Crosschecked [English 2.0.0](https://developer.d-robotics.cc/x5_sdk_doc_v2.0.0/en/toolchain_development/intermediate/supported_op_list.html) and [RDK X5 ONNX section](https://developer.d-robotics.cc/rdk_x_doc/Advanced_development/toolchain_development/intermediate/supported_op_list). Three ONNX rows agree; earlier Caffe Softmax and X3 MatMul are excluded.

MatMul: equal input ranks; trailing axes 1–8192, leading axes 1–4096. A cannot mix broadcasting and nonbroadcasting leading axes; B's nonbroadcasting axes must form the leading prefix. Documented examples: HDMK/H1KN, 11MK/HDKN, BHDMK/B11KN accepted; H1MK/1DKN, H1MK/HDKN, BHDMK/B1DKN excluded. Ambiguous symbolic patterns remain unknown.

Softmax: standard ONNX path rank4/axis3 eligible with explicit run_on_bpu. PyTorch converter path has different axes; standard domain/name is insufficient converter provenance. Default CPU statement is documentation, not a runtime measurement.

Resize: rank4 NCHW, spatial-only; nearest/linear. Enlargement nearest factors are powers of two with H≤W. Opset11 coordinate modes: half_pixel, pytorch_half_pixel, asymmetric, align_corners, tf_crop_and_resize. Crop ROI must be constant/spatial with integral transformed boundaries. Downsampling does not inherit enlargement restrictions.

ONNX semantics: [Div](https://onnx.ai/onnx/operators/onnx__Div.html), [Softmax](https://onnx.ai/onnx/operators/onnx__Softmax.html), [Resize](https://onnx.ai/onnx/operators/onnx__Resize.html), [MatMul](https://onnx.ai/onnx/operators/onnx__MatMul.html), plus installed onnx.defs schemas. These are semantic evidence, not hardware rules. Version and dtype differences are checked locally. No numerical hardware restriction comes from CPU float-only/maximum rank columns. Ruleset version and source version are independent of the unverified toolchain version.
