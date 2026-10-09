# 官方来源与版本记录

## 官网不可访问时的离线原文

仓库保存了官方开源手册的固定源码快照、Apache-2.0 许可和 SHA256：见 [离线调用说明](offline_manual/README.md)。在根目录运行 `python3 references/offline_manual/query.py --operator Transpose --json` 即可读取 X5 ONNX 表的原始 BPU/CPU 列、行号和使用限制；`--list` 列出 159 个条目。此查询标记 CACHED，不自动判断 BPU 支持，不替换下面经过版本核对的规则依据。源码快照没有 SDK 发布版本号，不能冒充实时网页或 SDK 1.1.2/2.0.0。

检索日期：2026-10-09。目标：RDK X5 / Bayes-e / 标准 ONNX Conv2D。工具链版本 **unverified**；本版不执行工具链，也不把已知其他项目版本当作本项目已验证版本。

| 用途 | 标题 / 版本 | URL | 精确章节 |
|---|---|---|---|
| 主规则依据 | RDK X3/X5 DOC，页面标示最后更新 2026-08-04 | https://developer.d-robotics.cc/rdk_x_doc/Advanced_development/toolchain_development/intermediate/supported_op_list | RDK X5 支持的 ONNX 算子列表 → Conv → 四维输入（conv2d）→ X5 BPU 支持约束 |
| 交叉核对 | X5 芯片用户手册 1.1.2 | https://developer.d-robotics.cc/x5_sdk_doc/toolchain_development/intermediate/supported_op_list.html | ONNX 算子列表 → Conv |
| 版本对照 | X5 Chip User Manual 2.0.0 | https://developer.d-robotics.cc/x5_sdk_doc_v2.0.0/en/toolchain_development/intermediate/supported_op_list.html | ONNX operator list → Conv |
| 属性语义 | ONNX Conv 规范，检索时网页版本 1.24.0；示例模型 opset 11 | https://onnx.ai/onnx/operators/onnx__Conv.html | Conv attributes / inputs；不是 BPU 数值依据 |
| 可选可视化 | Netron 官方仓库 | https://github.com/lutzroeder/netron | Python start / wait；不使用私有 JS 接口 |

## 提炼结果（语义归纳，不复制整个官方表格）

三份 X5 资料的 Conv2D 基础约束一致：kernel H/W 为 1–31；每组 kernel C×H×W 最大 32767；stride H/W 为 1–256；dilation H/W 为 1–16；padding 各侧为 0–256。膨胀卷积要求 stride=1，输入 H/W 被相应 dilation 整除，并涉及实际量化输出 int8 条件。

通道限制存在常规范围和量化子图末端放宽条件，工具链未接入时不能确定例外的适用性。Kernel N/C 范围与放宽条款的交互保守处理为条件提示，见规则备注。

本版规则包 0.1.0 的确定性检查保留上述数值及来源。需要融合/量化事实的判断返回 UNKNOWN。Conv→Add shortcut 的规则存在版本措辞差异，明确排除自动检查。

注意表格列：auto_pad、对称 pads、某些原始 dtype 描述属于 **CPU 支持约束**，不作为 BPU 违规判据。网页会变化，规则更新必须升级规则包版本并同步边界测试。
