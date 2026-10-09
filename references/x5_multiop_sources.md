# V1.2 六算子来源审查

核对日期：2026-10-09。只采纳 X5 ONNX 表的 BPU 栏；以下三个官方网页均实际在线读取，主站标示最后更新 2026-08-04。source_version、ruleset_version、模型 opset、toolchain_version 是不同概念。工具链始终 unverified。

- [RDK X3/X5 DOC](https://developer.d-robotics.cc/rdk_x_doc/Advanced_development/toolchain_development/intermediate/supported_op_list)：RDK X5 支持的 ONNX 算子列表，不使用前面的 X3/Caffe 表。
- [X5 芯片用户手册 1.1.2](https://developer.d-robotics.cc/x5_sdk_doc/toolchain_development/intermediate/supported_op_list.html)：6.3.3.3 X5 支持的 ONNX 算子列表。作为新增 YAML 的主来源。
- [X5 Chip User Manual 2.0.0](https://developer.d-robotics.cc/x5_sdk_doc_v2.0.0/en/toolchain_development/intermediate/supported_op_list.html)：X5 ONNX operator list，交叉核对同名行。

| 算子行 | BPU 栏语义摘要 | 跨版本核对 |
|---|---|---|
| Sigmoid | 输入/输出 rank 1–10；具备 int16 能力 | 三份一致 |
| Concat | 禁止沿 N 轴拼接；具备 int16 能力 | 三份一致；非明确 batch 布局保留未知 |
| Slice | 可处理非四维，无额外数值限制；具备 int16 能力 | 三份一致；动态索引能否转换没有明确承诺 |
| Add / Mul | rank 1–10、最多一个常量、支持双边广播；高维需按相邻轴规则归并到四维 | 三份一致；Add shortcut 属编译器审查 |
| Gemm | 编译器转成 Conv；参考转换后的 Conv 条款 | 三份一致；二维原始形状不是转换布局证据 |

ONNX 参数含义使用安装的 `onnx.defs.get_schema(op, imported_opset, '')` 核对，并参考 [ONNX 官方算子文档](https://onnx.ai/onnx/operators/)；这些不是硬件约束来源。新增提取器只接受明确审查的 schema 版本与标准域。

本轮不搬用 CPU 栏的 float-only 或 max-rank=8；不引入 X3 的 Mul 4D/C≤2048；int16 能力不排除原始 FP32。一般尺寸/对齐条款没有被转换成基于原始 Tensor B 的限制。
