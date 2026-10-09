# V1.1 ONNX 语义依据

检索日期 2026-10-09。这些来源只说明 ONNX 语义，不新增 RDK X5 硬件规则。实现同时通过安装的 ONNX `defs.get_schema(op, imported_version, '')` 核对版本；未审查版本留为信息不足。

| 模式 | 官方来源 | 使用的语义 |
|---|---|---|
| Identity | https://onnx.ai/onnx/operators/onnx__Identity.html | 原值转发；公开输入/输出接口需保留 |
| inverse Transpose | https://onnx.ai/onnx/operators/onnx__Transpose.html | 输出各轴来自 perm 指定输入轴，缺 perm 时逆序；组合 a[b[i]] |
| same dtype Cast | https://onnx.ai/onnx/operators/onnx__Cast.html | `to` 为目标类型 enum；仅相同且受支持的 dtype 才识别局部 no-op |
| no-op Reshape | https://onnx.ai/onnx/operators/onnx__Reshape.html | 常量目标、逐轴解析、0/-1 和 schema>=14 allowzero；resolved target 必须逐维相同 |
| Conv-BN review | https://onnx.ai/onnx/operators/onnx__BatchNormalization.html | inference/training 由版本相关模式属性与输出数量区分；仅直接 Tensor 连边考虑融合审查 |

不因算子名称相同就套用标准域语义。不执行 rewrite、BN 权重数学、ONNX Runtime 对比或工具链融合判断；verification_needed 只是未来变更所需的验证步骤。
