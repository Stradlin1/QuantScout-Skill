# RDK X5 ONNX Doctor 分析报告

**尚未用实际工具链验证：未经过 OpenExplorer/hb_mapper 实测。**

## 1. 模型概况

- 文件：examples/v1_1_demo.onnx
- SHA256：`fe9d960d5264b8167a8e8b20cb309dfd160eb5bf302ed975de024fbda5215cba`（仅 ONNX 文件）
- 节点：7；initializer：6；结构校验：checker_passed
- Opset：[{&#x27;domain&#x27;: &#x27;&#x27;, &#x27;version&#x27;: 15}]
- 算子计数：{&#x27;Identity&#x27;: 1, &#x27;Transpose&#x27;: 2, &#x27;Cast&#x27;: 1, &#x27;Reshape&#x27;: 1, &#x27;Conv&#x27;: 1, &#x27;BatchNormalization&#x27;: 1}

| 边界 | Tensor | Shape | dtype |
|---|---|---|---|
| inputs | image | [1, 2, 4, 4] | float32 |
| outputs | prediction | [1, 2, 4, 4] | float32 |

## 2. 本次规则来源

规则包 `x5-bayes-e-onnx-conv2d` / `0.1.0`；工具链版本：`unverified`。

- [RDK X3/X5 DOC / 模型算子支持列表](https://developer.d-robotics.cc/rdk_x_doc/Advanced_development/toolchain_development/intermediate/supported_op_list)；版本：网页最后更新 2026-08-04；检索快照 2026-10-09；章节：RDK X5 支持的 ONNX 算子列表 / Conv / 四维输入（conv2d）/ X5 BPU 支持约束
- [X5 芯片用户手册](https://developer.d-robotics.cc/x5_sdk_doc/toolchain_development/intermediate/supported_op_list.html)；版本：1.1.2；章节：ONNX / Conv
- [X5 Chip User Manual](https://developer.d-robotics.cc/x5_sdk_doc_v2.0.0/en/toolchain_development/intermediate/supported_op_list.html)；版本：2.0.0；章节：ONNX / Conv

已收录条款：

| 规则 ID | 字段 | 判定方式 |
|---|---|---|
| X5-CONV2D-KERNEL-H | kernel_h | auto_check / range |
| X5-CONV2D-KERNEL-W | kernel_w | auto_check / range |
| X5-CONV2D-KERNEL-VOLUME | kernel_elements_per_group | auto_check / max_value |
| X5-CONV2D-STRIDE-H | strides_h | auto_check / range |
| X5-CONV2D-DILATION-H | dilation_h | auto_check / range |
| X5-CONV2D-STRIDE-W | strides_w | auto_check / range |
| X5-CONV2D-DILATION-W | dilation_w | auto_check / range |
| X5-CONV2D-PAD-TOP | pads_top | auto_check / range |
| X5-CONV2D-PAD-LEFT | pads_left | auto_check / range |
| X5-CONV2D-PAD-BOTTOM | pads_bottom | auto_check / range |
| X5-CONV2D-PAD-RIGHT | pads_right | auto_check / range |
| X5-CONV2D-DILATED-STRIDE-H | strides_h | auto_check / equals |
| X5-CONV2D-DILATED-DIVISIBLE-H | dilation_h_divisible | auto_check / equals |
| X5-CONV2D-DILATED-STRIDE-W | strides_w | auto_check / equals |
| X5-CONV2D-DILATED-DIVISIBLE-W | dilation_w_divisible | auto_check / equals |
| X5-CONV2D-DILATED-OUTPUT | quantized_output_dtype | conditional / equals |
| X5-CONV2D-CHANNEL-IN_CHANNELS_PER_GROUP | in_channels_per_group | conditional / max_value |
| X5-CONV2D-CHANNEL-OUT_CHANNELS_PER_GROUP | out_channels_per_group | conditional / max_value |
| X5-CONV2D-CHANNEL-OUT_CHANNELS | out_channels | conditional / max_value |

未覆盖或需审查的条款：

- Conv→Add shortcut 的 stride 措辞存在版本差异：RDK 为 2，手册为 {1,2}；不自动套用。
- 通道与 Kernel N/C 限制及量化末端放宽的交互：仅条件提示，不判 FAIL。
- CPU 支持列中的 auto_pad、对称 pads 及 dtype 条款不作为 BPU 条款。
- 通用 shape、字节大小、真实内存、CPU/BPU 分配、延迟、量化精度未覆盖。

## 3. Conv2D 检查汇总

Conv 节点 1；确认 Conv2D 并执行检查 1。

| 状态 | 个数 |
|---|---|
| VIOLATION | 0 |
| NEEDS_VERIFICATION | 0 |
| NO_VIOLATION_FOUND | 1 |
| NOT_COVERED | 0 |

`NO_VIOLATION_FOUND` 仅表示已检查规则未发现违规；非 Conv 节点不按通过统计。

## 4. 异常节点清单

| 节点 ID / 名称 | 规则 | 实际值 | 允许值 | 理由 |
|---|---|---|---|---|

未发现已检查规则违规。

## 5. 节点风险溯源

无确定违规源节点，无需生成违规溯源。

## 6. 优化建议

- **main/node_000005 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker

## 7. 尚不能下结论的事项

- 尚未用实际工具链验证：未经过 OpenExplorer/hb_mapper 实测。
- V1 仅检查已收录的标准 ONNX Conv2D 规则，其他算子未覆盖。
- 静态检查不能确定真实 CPU/BPU 分配、量化误差、延迟、峰值 DDR 或任务精度。
- 路径只表示数据依赖，不表示下游节点违规。
- 模型 SHA256 仅覆盖 .onnx 文件；external data 内容不包含在该摘要中。

## 8. 文件索引

- [机器事实 analysis.json](analysis.json)
- 终端节点列表：`python -m rdkx5_doctor nodes --analysis analysis.json`
- 参数与证据：`python -m rdkx5_doctor inspect --analysis analysis.json --node <节点 ID>`
- 依赖路径：`python -m rdkx5_doctor trace --analysis analysis.json --node <节点 ID>`


## 9. Tensor Resource Analysis

**理论原始 Tensor 载荷，不是实际 BPU/DDR 分配或峰值内存。**
单位：B；1 MiB = 1,048,576 B；1 MB = 1,000,000 B。bool 假定一个逻辑字节。

| 类别 | Tensor 数 | 已知 B 之和 | 未知数 | 完整性 |
|---|---|---|---|---|
| model_outputs | 1 | 128 | 0 | COMPLETE |
| intermediate_activations | 6 | 768 | 0 | COMPLETE |
| initializers | 6 | 80 | 0 | COMPLETE |
| model_inputs | 1 | 128 | 0 | COMPLETE |
| constants | 0 | 0 | 0 | COMPLETE |
| other_unknown | 0 | 0 | 0 | COMPLETE |

### 最大已知输出（前 10）

| Tensor | Shape | dtype | 精确 B | MiB | Consumers | 假设 INT8 B |
|---|---|---|---|---|---|---|
| prediction | [1, 2, 4, 4] | float32 | 128 | 0.00 | 0 | 32 |

### 最大已知中间激活（前 10）

| Tensor | Shape | dtype | 精确 B | MiB | Consumers | 假设 INT8 B |
|---|---|---|---|---|---|---|
| copied | [1, 2, 4, 4] | float32 | 128 | 0.00 | 1 | 32 |
| features | [1, 2, 4, 4] | float32 | 128 | 0.00 | 1 | 32 |
| nchw | [1, 2, 4, 4] | float32 | 128 | 0.00 | 1 | 32 |
| nhwc | [1, 4, 4, 2] | float32 | 128 | 0.00 | 1 | 32 |
| same_dtype | [1, 2, 4, 4] | float32 | 128 | 0.00 | 1 | 32 |
| same_shape | [1, 2, 4, 4] | float32 | 128 | 0.00 | 1 | 32 |

**Hypothetical INT8 raw-payload scenario（假设 INT8 原始载荷场景）**：仅元素数乘 1 B，不是实际量化结果或整模型压缩预测。
多消费者中间 Tensor：0；fanout 只表示图连接，不代表额外分配。

未知尺寸（最多列 10 个；完整内容查询 tensors）：

- 无。

完整资源查询：`python -m rdkx5_doctor tensors --analysis analysis.json --limit 0`。

- Theoretical logical dense Tensor payload only; not measured runtime/BPU/DDR memory.
- Known sums are not peak memory: lifetimes, aliasing, reuse, layout, padding and fusion are unknown.
- Hypothetical INT8 raw-payload scenario only; not actual quantization, placement, accuracy or performance.
- bool uses an assumed one logical byte representation; packed/string/sparse/container types unsupported.
- Each Tensor is counted once; initializer/output overlaps use initializer as primary category.

## 10. Graph Optimization Candidates

**没有改写、删除或融合任何模型节点。结构候选不是新增的 X5 BPU 支持规则。**
候选/待验证观察共 5；分类：{&#x27;SEMANTICALLY_REDUNDANT&#x27;: 4, &#x27;REVIEW_REQUIRED&#x27;: 1}；模式：{&#x27;IDENTITY&#x27;: 1, &#x27;TRANSPOSE_INVERSE_PAIR&#x27;: 1, &#x27;CAST_SAME_DTYPE&#x27;: 1, &#x27;RESHAPE_NOOP&#x27;: 1, &#x27;CONV_BN_FUSION_REVIEW&#x27;: 1}。

INSUFFICIENT_INFORMATION 是未确认观察，与已建立局部语义冗余明确分开。

### OPT-0001 / IDENTITY / SEMANTICALLY_REDUNDANT

节点 ID：[&#x27;main/node_000000&#x27;]；原始名称：[&#x27;input_identity&#x27;]
Tensor：[&#x27;image&#x27;, &#x27;copied&#x27;]
识别理由：Supported local operator semantics establish an unchanged value/shape transformation.
证据：{&#x27;value_transformation&#x27;: &#x27;identity&#x27;, &#x27;input_tensor&#x27;: &#x27;image&#x27;, &#x27;output_tensor&#x27;: &#x27;copied&#x27;, &#x27;output_consumer_count&#x27;: 1, &#x27;schema_version&#x27;: 14, &#x27;input_metadata_missing&#x27;: False, &#x27;connections&#x27;: [], &#x27;tensor_interfaces&#x27;: [{&#x27;name&#x27;: &#x27;image&#x27;, &#x27;is_graph_input&#x27;: True, &#x27;is_graph_output&#x27;: False, &#x27;producer&#x27;: None, &#x27;consumers&#x27;: [&#x27;main/node_000000&#x27;]}, {&#x27;name&#x27;: &#x27;copied&#x27;, &#x27;is_graph_input&#x27;: False, &#x27;is_graph_output&#x27;: False, &#x27;producer&#x27;: &#x27;main/node_000000&#x27;, &#x27;consumers&#x27;: [&#x27;main/node_000001&#x27;]}]}
条件：[&#x27;Preserve public input/output names, shapes, dtypes and all consumer connections&#x27;, &#x27;Apply to a separate model copy only after validating all blockers and overlapping candidates&#x27;]
阻碍/需审查：[]
重叠候选：[]；不能累加收益。
保守结构收益：Potentially fewer graph operations if retained by the compiler; no measured speed or memory improvement
未来验证：[&#x27;ONNX checker on a future rewritten copy&#x27;, &#x27;Compare input/output shapes, dtypes and public output names&#x27;, &#x27;ONNX Runtime numerical comparison on representative inputs before any future rewrite; check task metrics where relevant&#x27;]
下游输出按需查询：`candidate --analysis analysis.json --id OPT-0001`；只表示数据依赖。
- ONNX 语义来源：https://onnx.ai/onnx/operators/onnx__Identity.html

### OPT-0002 / TRANSPOSE_INVERSE_PAIR / SEMANTICALLY_REDUNDANT

节点 ID：[&#x27;main/node_000001&#x27;, &#x27;main/node_000002&#x27;]；原始名称：[&#x27;to_nhwc&#x27;, &#x27;back_nchw&#x27;]
Tensor：[&#x27;copied&#x27;, &#x27;nhwc&#x27;, &#x27;nchw&#x27;]
识别理由：Supported local operator semantics establish an unchanged value/shape transformation.
证据：{&#x27;perm_a&#x27;: [0, 2, 3, 1], &#x27;perm_b&#x27;: [0, 3, 1, 2], &#x27;composite_perm&#x27;: [0, 1, 2, 3], &#x27;intermediate_consumer_count&#x27;: 1, &#x27;intermediate_is_graph_output&#x27;: False, &#x27;connections&#x27;: [{&#x27;source&#x27;: &#x27;main/node_000001&#x27;, &#x27;target&#x27;: &#x27;main/node_000002&#x27;, &#x27;tensor&#x27;: &#x27;nhwc&#x27;}], &#x27;tensor_interfaces&#x27;: [{&#x27;name&#x27;: &#x27;copied&#x27;, &#x27;is_graph_input&#x27;: False, &#x27;is_graph_output&#x27;: False, &#x27;producer&#x27;: &#x27;main/node_000000&#x27;, &#x27;consumers&#x27;: [&#x27;main/node_000001&#x27;]}, {&#x27;name&#x27;: &#x27;nhwc&#x27;, &#x27;is_graph_input&#x27;: False, &#x27;is_graph_output&#x27;: False, &#x27;producer&#x27;: &#x27;main/node_000001&#x27;, &#x27;consumers&#x27;: [&#x27;main/node_000002&#x27;]}, {&#x27;name&#x27;: &#x27;nchw&#x27;, &#x27;is_graph_input&#x27;: False, &#x27;is_graph_output&#x27;: False, &#x27;producer&#x27;: &#x27;main/node_000002&#x27;, &#x27;consumers&#x27;: [&#x27;main/node_000003&#x27;]}]}
条件：[&#x27;Preserve public input/output names, shapes, dtypes and all consumer connections&#x27;, &#x27;Apply to a separate model copy only after validating all blockers and overlapping candidates&#x27;]
阻碍/需审查：[]
重叠候选：[]；不能累加收益。
保守结构收益：Potentially fewer graph operations if retained by the compiler; no measured speed or memory improvement
未来验证：[&#x27;ONNX checker on a future rewritten copy&#x27;, &#x27;Compare input/output shapes, dtypes and public output names&#x27;, &#x27;ONNX Runtime numerical comparison on representative inputs before any future rewrite; check task metrics where relevant&#x27;]
下游输出按需查询：`candidate --analysis analysis.json --id OPT-0002`；只表示数据依赖。
- ONNX 语义来源：https://onnx.ai/onnx/operators/onnx__Transpose.html

### OPT-0003 / CAST_SAME_DTYPE / SEMANTICALLY_REDUNDANT

节点 ID：[&#x27;main/node_000003&#x27;]；原始名称：[&#x27;same_float_cast&#x27;]
Tensor：[&#x27;nchw&#x27;, &#x27;same_dtype&#x27;]
识别理由：Supported local operator semantics establish an unchanged value/shape transformation.
证据：{&#x27;input_dtype&#x27;: &#x27;float32&#x27;, &#x27;destination_dtype&#x27;: &#x27;float32&#x27;, &#x27;to&#x27;: 1, &#x27;schema_version&#x27;: 13, &#x27;connections&#x27;: [], &#x27;tensor_interfaces&#x27;: [{&#x27;name&#x27;: &#x27;nchw&#x27;, &#x27;is_graph_input&#x27;: False, &#x27;is_graph_output&#x27;: False, &#x27;producer&#x27;: &#x27;main/node_000002&#x27;, &#x27;consumers&#x27;: [&#x27;main/node_000003&#x27;]}, {&#x27;name&#x27;: &#x27;same_dtype&#x27;, &#x27;is_graph_input&#x27;: False, &#x27;is_graph_output&#x27;: False, &#x27;producer&#x27;: &#x27;main/node_000003&#x27;, &#x27;consumers&#x27;: [&#x27;main/node_000004&#x27;]}]}
条件：[&#x27;Preserve public input/output names, shapes, dtypes and all consumer connections&#x27;, &#x27;Apply to a separate model copy only after validating all blockers and overlapping candidates&#x27;]
阻碍/需审查：[]
重叠候选：[]；不能累加收益。
保守结构收益：Potentially fewer graph operations if retained by the compiler; no measured speed or memory improvement
未来验证：[&#x27;ONNX checker on a future rewritten copy&#x27;, &#x27;Compare input/output shapes, dtypes and public output names&#x27;, &#x27;ONNX Runtime numerical comparison on representative inputs before any future rewrite; check task metrics where relevant&#x27;]
下游输出按需查询：`candidate --analysis analysis.json --id OPT-0003`；只表示数据依赖。
- ONNX 语义来源：https://onnx.ai/onnx/operators/onnx__Cast.html

### OPT-0004 / RESHAPE_NOOP / SEMANTICALLY_REDUNDANT

节点 ID：[&#x27;main/node_000004&#x27;]；原始名称：[&#x27;same_shape_reshape&#x27;]
Tensor：[&#x27;same_dtype&#x27;, &#x27;target_shape&#x27;, &#x27;same_shape&#x27;]
识别理由：Supported local operator semantics establish an unchanged value/shape transformation.
证据：{&#x27;input_shape&#x27;: [1, 2, 4, 4], &#x27;target_values&#x27;: [0, 2, 4, 4], &#x27;resolved_target&#x27;: [1, 2, 4, 4], &#x27;allowzero&#x27;: 0, &#x27;schema_version&#x27;: 14, &#x27;connections&#x27;: [], &#x27;tensor_interfaces&#x27;: [{&#x27;name&#x27;: &#x27;same_dtype&#x27;, &#x27;is_graph_input&#x27;: False, &#x27;is_graph_output&#x27;: False, &#x27;producer&#x27;: &#x27;main/node_000003&#x27;, &#x27;consumers&#x27;: [&#x27;main/node_000004&#x27;]}, {&#x27;name&#x27;: &#x27;target_shape&#x27;, &#x27;is_graph_input&#x27;: False, &#x27;is_graph_output&#x27;: False, &#x27;producer&#x27;: None, &#x27;consumers&#x27;: [&#x27;main/node_000004&#x27;]}, {&#x27;name&#x27;: &#x27;same_shape&#x27;, &#x27;is_graph_input&#x27;: False, &#x27;is_graph_output&#x27;: False, &#x27;producer&#x27;: &#x27;main/node_000004&#x27;, &#x27;consumers&#x27;: [&#x27;main/node_000005&#x27;]}]}
条件：[&#x27;Preserve public input/output names, shapes, dtypes and all consumer connections&#x27;, &#x27;Apply to a separate model copy only after validating all blockers and overlapping candidates&#x27;]
阻碍/需审查：[]
重叠候选：[]；不能累加收益。
保守结构收益：Potentially fewer graph operations if retained by the compiler; no measured speed or memory improvement
未来验证：[&#x27;ONNX checker on a future rewritten copy&#x27;, &#x27;Compare input/output shapes, dtypes and public output names&#x27;, &#x27;ONNX Runtime numerical comparison on representative inputs before any future rewrite; check task metrics where relevant&#x27;]
下游输出按需查询：`candidate --analysis analysis.json --id OPT-0004`；只表示数据依赖。
- ONNX 语义来源：https://onnx.ai/onnx/operators/onnx__Reshape.html

### OPT-0005 / CONV_BN_FUSION_REVIEW / REVIEW_REQUIRED

节点 ID：[&#x27;main/node_000005&#x27;, &#x27;main/node_000006&#x27;]；原始名称：[&#x27;normal_conv&#x27;, &#x27;inference_bn&#x27;]
Tensor：[&#x27;same_shape&#x27;, &#x27;weight&#x27;, &#x27;features&#x27;, &#x27;scale&#x27;, &#x27;bias&#x27;, &#x27;mean&#x27;, &#x27;variance&#x27;, &#x27;prediction&#x27;]
识别理由：An actual Conv output directly enters inference BN; fixed-parameter fusion is a review opportunity, not evidence of compiler fusion.
证据：{&#x27;bn_schema_version&#x27;: 15, &#x27;inference_mode&#x27;: True, &#x27;conv_output_tensor&#x27;: &#x27;features&#x27;, &#x27;conv_output_consumer_count&#x27;: 1, &#x27;channels&#x27;: 2, &#x27;bn_parameters&#x27;: [{&#x27;name&#x27;: &#x27;scale&#x27;, &#x27;shape&#x27;: [2], &#x27;dtype&#x27;: &#x27;float32&#x27;, &#x27;is_constant&#x27;: True}, {&#x27;name&#x27;: &#x27;bias&#x27;, &#x27;shape&#x27;: [2], &#x27;dtype&#x27;: &#x27;float32&#x27;, &#x27;is_constant&#x27;: True}, {&#x27;name&#x27;: &#x27;mean&#x27;, &#x27;shape&#x27;: [2], &#x27;dtype&#x27;: &#x27;float32&#x27;, &#x27;is_constant&#x27;: True}, {&#x27;name&#x27;: &#x27;variance&#x27;, &#x27;shape&#x27;: [2], &#x27;dtype&#x27;: &#x27;float32&#x27;, &#x27;is_constant&#x27;: True}], &#x27;epsilon&#x27;: 1e-05, &#x27;parameter_values_read&#x27;: False, &#x27;connections&#x27;: [{&#x27;source&#x27;: &#x27;main/node_000005&#x27;, &#x27;target&#x27;: &#x27;main/node_000006&#x27;, &#x27;tensor&#x27;: &#x27;features&#x27;}], &#x27;tensor_interfaces&#x27;: [{&#x27;name&#x27;: &#x27;same_shape&#x27;, &#x27;is_graph_input&#x27;: False, &#x27;is_graph_output&#x27;: False, &#x27;producer&#x27;: &#x27;main/node_000004&#x27;, &#x27;consumers&#x27;: [&#x27;main/node_000005&#x27;]}, {&#x27;name&#x27;: &#x27;weight&#x27;, &#x27;is_graph_input&#x27;: False, &#x27;is_graph_output&#x27;: False, &#x27;producer&#x27;: None, &#x27;consumers&#x27;: [&#x27;main/node_000005&#x27;]}, {&#x27;name&#x27;: &#x27;features&#x27;, &#x27;is_graph_input&#x27;: False, &#x27;is_graph_output&#x27;: False, &#x27;producer&#x27;: &#x27;main/node_000005&#x27;, &#x27;consumers&#x27;: [&#x27;main/node_000006&#x27;]}, {&#x27;name&#x27;: &#x27;scale&#x27;, &#x27;is_graph_input&#x27;: False, &#x27;is_graph_output&#x27;: False, &#x27;producer&#x27;: None, &#x27;consumers&#x27;: [&#x27;main/node_000006&#x27;]}, {&#x27;name&#x27;: &#x27;bias&#x27;, &#x27;is_graph_input&#x27;: False, &#x27;is_graph_output&#x27;: False, &#x27;producer&#x27;: None, &#x27;consumers&#x27;: [&#x27;main/node_000006&#x27;]}, {&#x27;name&#x27;: &#x27;mean&#x27;, &#x27;is_graph_input&#x27;: False, &#x27;is_graph_output&#x27;: False, &#x27;producer&#x27;: None, &#x27;consumers&#x27;: [&#x27;main/node_000006&#x27;]}, {&#x27;name&#x27;: &#x27;variance&#x27;, &#x27;is_graph_input&#x27;: False, &#x27;is_graph_output&#x27;: False, &#x27;producer&#x27;: None, &#x27;consumers&#x27;: [&#x27;main/node_000006&#x27;]}, {&#x27;name&#x27;: &#x27;prediction&#x27;, &#x27;is_graph_input&#x27;: False, &#x27;is_graph_output&#x27;: True, &#x27;producer&#x27;: &#x27;main/node_000006&#x27;, &#x27;consumers&#x27;: []}]}
条件：[&#x27;Preserve public input/output names, shapes, dtypes and all consumer connections&#x27;, &#x27;Apply to a separate model copy only after validating all blockers and overlapping candidates&#x27;]
阻碍/需审查：[&#x27;Public graph output must retain name/interface: prediction&#x27;]
重叠候选：[]；不能累加收益。
保守结构收益：Potentially fewer graph operations if retained by the compiler; no measured speed or memory improvement
未来验证：[&#x27;ONNX checker on a future rewritten copy&#x27;, &#x27;Compare input/output shapes, dtypes and public output names&#x27;, &#x27;ONNX Runtime numerical comparison on representative inputs before any future rewrite; check task metrics where relevant&#x27;]
下游输出按需查询：`candidate --analysis analysis.json --id OPT-0005`；只表示数据依赖。
- ONNX 语义来源：https://onnx.ai/onnx/operators/onnx__BatchNormalization.html


- Structural/semantic candidates are not X5 operator compatibility rules.
- No rewrite, compiler fusion verification or measured performance gain.
- Reachability is computed on demand from saved Tensor edges. Imported opset &gt;23 is unreviewed.