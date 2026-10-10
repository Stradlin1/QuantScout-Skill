# RDK X5 ONNX Doctor V1.3 分析报告

**静态规则检查不是工具链 CPU/BPU 分配，也不意味着模型不能运行。未经过 OpenExplorer/hb_mapper 实测。**

## 0. 执行范围与核心结论

模型 `demo.onnx`；SHA256 `9f1da4be1ddb5e3b3873daffc4d5734014bf593120e7812d7a4736b316b2b116`（仅 ONNX protobuf）。
checker=checker_passed；opset=&#91;{&#x27;domain&#x27;: &#x27;&#x27;, &#x27;version&#x27;: 11}&#93;；主图节点 5；Tensor 7；ONNX graph inputs 1 / outputs 2。
逐节点统计仅限已解析主图；嵌套子图未展开、不计入诊断统计。ONNX checker 不等于子图诊断。
规则入口覆盖 3；未覆盖 2；确定违规 1；待验证 0。
Registry 0.4.0；子版本 {&#x27;Add&#x27;: &#x27;0.2.0&#x27;, &#x27;AveragePool&#x27;: &#x27;0.1.0&#x27;, &#x27;Concat&#x27;: &#x27;0.2.0&#x27;, &#x27;Conv&#x27;: &#x27;0.1.0&#x27;, &#x27;Gemm&#x27;: &#x27;0.2.0&#x27;, &#x27;MatMul&#x27;: &#x27;0.1.0&#x27;, &#x27;MaxPool&#x27;: &#x27;0.1.0&#x27;, &#x27;Mul&#x27;: &#x27;0.2.0&#x27;, &#x27;Reshape&#x27;: &#x27;0.1.0&#x27;, &#x27;Resize&#x27;: &#x27;0.1.0&#x27;, &#x27;Sigmoid&#x27;: &#x27;0.2.0&#x27;, &#x27;Slice&#x27;: &#x27;0.2.0&#x27;, &#x27;Softmax&#x27;: &#x27;0.1.0&#x27;, &#x27;Split&#x27;: &#x27;0.1.0&#x27;}；工具链 unverified，verified=false。
覆盖表示静态规则范围，不是 BPU 运行比例。

## 1. 算子覆盖与结论矩阵

| 算子 | 数量 | 已检查无违规 | 违反硬约束 | 待验证 | 未覆盖 | 规则 / 来源版本 | AUTO / PARTIAL / NOT |
|---|---|---|---|---|---|---|---|
| Conv | 1 | 0 | 1 | 0 | 0 | 0.1.0 / 网页最后更新 2026-08-04；检索快照 2026-10-09 | {&#x27;AUTO_CHECKED&#x27;: 0, &#x27;NOT_COVERED&#x27;: 0, &#x27;PARTIAL_OR_CONDITIONAL&#x27;: 1} |
| Relu | 1 | 0 | 0 | 0 | 1 | None / None | {&#x27;AUTO_CHECKED&#x27;: 0, &#x27;NOT_COVERED&#x27;: 1, &#x27;PARTIAL_OR_CONDITIONAL&#x27;: 0} |
| Identity | 1 | 0 | 0 | 0 | 1 | None / None | {&#x27;AUTO_CHECKED&#x27;: 0, &#x27;NOT_COVERED&#x27;: 1, &#x27;PARTIAL_OR_CONDITIONAL&#x27;: 0} |
| Add | 1 | 1 | 0 | 0 | 0 | 0.2.0 / 1.1.2 | {&#x27;AUTO_CHECKED&#x27;: 1, &#x27;NOT_COVERED&#x27;: 0, &#x27;PARTIAL_OR_CONDITIONAL&#x27;: 0} |
| Concat | 1 | 1 | 0 | 0 | 0 | 0.2.0 / 1.1.2 | {&#x27;AUTO_CHECKED&#x27;: 1, &#x27;NOT_COVERED&#x27;: 0, &#x27;PARTIAL_OR_CONDITIONAL&#x27;: 0} |

四状态按节点守恒；NO_VIOLATION_FOUND 只说明已执行条件未发现违规。

## 2. 确定违反 BPU 约束的节点

### main/node_000000 / oversized_kernel / Conv

输入 &#91;&#x27;image&#x27;, &#x27;weight&#x27;&#93;；输出 &#91;&#x27;features&#x27;&#93;。
- `X5-CONV2D-KERNEL-H`：实际 32；允许 {&#x27;max&#x27;: 31, &#x27;min&#x27;: 1}；卷积核空间维度超出官方范围。证据 {&#x27;attributes&#x27;: {}, &#x27;derived_field&#x27;: &#x27;kernel_h&#x27;, &#x27;inputs&#x27;: &#91;{&#x27;consumers&#x27;: &#91;&#x27;main/node_000000&#x27;&#93;, &#x27;dtype&#x27;: &#x27;float32&#x27;, &#x27;name&#x27;: &#x27;image&#x27;, &#x27;producer&#x27;: None, &#x27;shape&#x27;: &#91;1, 2, 40, 8&#93;}, {&#x27;consumers&#x27;: &#91;&#x27;main/node_000000&#x27;&#93;, &#x27;dtype&#x27;: &#x27;float32&#x27;, &#x27;name&#x27;: &#x27;weight&#x27;, &#x27;producer&#x27;: None, &#x27;shape&#x27;: &#91;2, 2, 32, 1&#93;}&#93;}；[RDK X3/X5 DOC / 模型算子支持列表](https://developer.d-robotics.cc/rdk_x_doc/Advanced_development/toolchain_development/intermediate/supported_op_list) / 网页最后更新 2026-08-04；检索快照 2026-10-09 / RDK X5 支持的 ONNX 算子列表 / Conv / 四维输入（conv2d）/ X5 BPU 支持约束。
可达输出 &#91;&#x27;prediction&#x27;, &#x27;auxiliary&#x27;&#93;；前驱 &#91;&#93; / 后继 &#91;&#x27;main/node_000001&#x27;, &#x27;main/node_000002&#x27;&#93;。完整 Tensor 代表路径在 JSON/trace；依赖不表示下游同样违规。
仅可提出架构/导出方向；修改须在独立原工程实施、重新导出并对比，本项目不定位训练源码或修改 ONNX。


## 3. 需要更多元信息或工具链确认

| 算子 / 原因码 | 节点数 | 代表节点 | 缺失条件 |
|---|---|---|---|
| — | 0 | — | 无阻碍性未知项 |

### Shape 推断与未知来源

载荷已知 7 → 7；未知 0 → 0；新证明 Tensor 0 / 轴 0；冲突 0；预算截断 False。
符号占位不等于动态外部输入；仅有界元信息证明，无特征图/大权重求值。

本次 Shape 未留未解决轴；不代表部署内存或量化已验证。
诊断字段变化均附 proof IDs；原始静态失败证据保存在 shape_analysis.original_static_failures。

同一节点可在多个原因组出现，不能相加当作待验证节点总数。非阻碍性融合提示不计入关键 UNKNOWN。

## 4. 重要 Tensor 资源 / Tensor Resource Analysis

**理论原始载荷，不是实际 BPU/DDR/SRAM 或峰值内存。Hypothetical INT8 raw-payload scenario 仅是假设元素数×1 B。**

| 边界 Tensor | Shape / dtype | 原始 B | 假设 INT8 B |
|---|---|---|---|
| image | &#91;1, 2, 40, 8&#93; / float32 | 2560 | 640 |
| prediction | &#91;1, 2, 9, 8&#93; / float32 | 576 | 144 |
| auxiliary | &#91;1, 4, 9, 8&#93; / float32 | 1152 | 288 |

### 前 10 个已知大小的中间 Tensor

| Tensor | Shape / dtype | 原始 B | MiB | producer | consumers | 假设 INT8 B |
|---|---|---|---|---|---|---|
| activated | &#91;1, 2, 9, 8&#93; / float32 | 576 | 0.00054931640625 | main/node_000001 | 2 | 144 |
| features | &#91;1, 2, 9, 8&#93; / float32 | 576 | 0.00054931640625 | main/node_000000 | 2 | 144 |
| skip | &#91;1, 2, 9, 8&#93; / float32 | 576 | 0.00054931640625 | main/node_000002 | 1 | 144 |

资源未知 0 个；按原因 {}；仅已知集合的 Top10，不保证全图最大。
中间已知载荷之和 1728 B（COMPLETE）；多消费者中间 Tensor 2 个。
1 MiB=1,048,576 B；1 MB=1,000,000 B。载荷之和不是峰值，fanout 不代表额外分配。


### MatMul / Softmax / Resize 专项

- 非阻塞 review Add/COMPILER_SHORTCUT_UNVERIFIED：1 个唯一节点，样例 &#91;&#x27;main/node_000003&#x27;&#93;。组可重叠，不是额外待验证节点。

## 5. Graph Optimization Candidates

候选/观察 1；按模式 {&#x27;IDENTITY&#x27;: 1}；按分类 {&#x27;SEMANTICALLY_REDUNDANT&#x27;: 1}。
SEMANTICALLY_REDUNDANT 为局部语义冗余；REVIEW_REQUIRED 需接口/分支/融合审查；INSUFFICIENT_INFORMATION 未证明冗余。没有改写模型。
- OPT-0001 / IDENTITY / SEMANTICALLY_REDUNDANT：节点 &#91;&#x27;main/node_000002&#x27;&#93;，原名 &#91;&#x27;residual_skip&#x27;&#93;；条件/阻碍 &#91;&#93;；完整 Tensor 证据、重叠与验证步骤见 candidate/JSON。

## 6. 基于证据的回源建议

- main/node_000000 / X5-CONV2D-KERNEL-H / FAIL：卷积核空间维度超出官方范围。可选架构方向：审查算子排列、导出表达或输出头与后处理的分离；没有原工程时不提供代码位置或补丁。如需模型变更，在独立训练/导出工程实施并重新导出，对比接口/数值及任务指标；不修补原 ONNX。

## 7. 实际使用规则来源、未覆盖与限制

- [RDK X3/X5 DOC / 模型算子支持列表](https://developer.d-robotics.cc/rdk_x_doc/Advanced_development/toolchain_development/intermediate/supported_op_list)；文档版本 网页最后更新 2026-08-04；检索快照 2026-10-09；抓取 2026-10-09；仅 X5 ONNX BPU 栏。
- [X5 芯片用户手册](https://developer.d-robotics.cc/x5_sdk_doc/toolchain_development/intermediate/supported_op_list.html)；文档版本 1.1.2；抓取 2026-10-09；仅 X5 ONNX BPU 栏。
未覆盖算子/范围：&#91;&#x27;Relu&#x27;, &#x27;Identity&#x27;&#93;。
- Conv→Add shortcut 的 stride 措辞存在版本差异：RDK 为 2，手册为 {1,2}；不自动套用。
- 通道与 Kernel N/C 限制及量化末端放宽的交互：仅条件提示，不判 FAIL。
- CPU 支持列中的 auto_pad、对称 pads 及 dtype 条款不作为 BPU 条款。
- 通用 shape、字节大小、真实内存、CPU/BPU 分配、延迟、量化精度未覆盖。
- 新增六类来源为 X5 ONNX BPU 栏；一般大小/对齐、实际量化精度、编译器融合不做确定性判断。
- Concat 非明确 NCHW rank4 的 batch/N 布局未知；Slice 动态参数和 Gemm→Conv 实际转换需验证。
- V1.4 新包仅审查 Opset 11；Pool 数值条款仅用于 2D rank4 上下文，auto_pad 有效填充未推导；不套用 CPU 列的 5D/存储限制。
- Relu/Transpose 只登记官方知识，不以无专属限制制造自动 PASS；Split 的 N 维需要独立布局依据。
- Split 的 split数应可以整除 未明确指输出数还是分块长度；COUNT-DIVISIBILITY 仅非阻塞审查，不按 L % output_count 产生 FAIL。
- 尚未用实际工具链验证：未经过 OpenExplorer/hb_mapper 实测。
- 只检查实际注册的标准 ONNX 算子和已审查版本；其它算子不代表兼容。
- 静态检查不能确定真实 CPU/BPU 分配、量化误差、延迟、峰值 DDR 或任务精度。
- 路径只表示数据依赖，不表示下游节点违规。
- 模型 SHA256 仅覆盖 .onnx 文件；external data 内容不包含在该摘要中。

## 8. 查询方法和复现元数据

- 完整证据：[analysis.json](analysis.json)；机器路径和模型边界保存在 JSON。
- `python -m rdkx5_doctor shapes --analysis analysis.json --summary`；`shape --analysis analysis.json --tensor <tensor>`。
- `python -m rdkx5_doctor rules validate`；`rules list --operator Mul`。
- `python -m rdkx5_doctor nodes --analysis analysis.json --status VIOLATION`（或 NEEDS_VERIFICATION）。
- `python -m rdkx5_doctor inspect --analysis analysis.json --node <node_id>`。
- `python -m rdkx5_doctor trace --analysis analysis.json --node <node_id>`。
- `python -m rdkx5_doctor tensors --analysis analysis.json --kind output`；`tensor --analysis analysis.json --name <tensor>`。
- `python -m rdkx5_doctor candidates --analysis analysis.json`；`candidate --analysis analysis.json --id <candidate_id>`。
- 在训练/导出工程生成新模型后：`analyze --model <new_model.onnx> --out reports/new-run`。
