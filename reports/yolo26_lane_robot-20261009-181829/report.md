# RDK X5 ONNX Doctor 分析报告

**尚未用实际工具链验证：未经过 OpenExplorer/hb_mapper 实测。**

## 1. 模型概况

- 文件：/home/xhm/lianghua_ws/rdktoolchain/horizon_x5_open_explorer_v1.2.8/horizon_x5_open_explorer_v1.2.8-py310_20240926/samples/ai_toolchain/horizon_model_convert_sample/04_detection/15_tasknav/model/yolo26_lane_robot.onnx
- SHA256：`a8ea6ccf474c77614f33512c4a118390f3b14c28e4871615670888a6ca292de4`（仅 ONNX 文件）
- 节点：281；initializer：118；结构校验：checker_passed
- Opset：[{&#x27;domain&#x27;: &#x27;&#x27;, &#x27;version&#x27;: 11}]
- 算子计数：{&#x27;Conv&#x27;: 47, &#x27;Sigmoid&#x27;: 38, &#x27;Mul&#x27;: 51, &#x27;Shape&#x27;: 4, &#x27;Constant&#x27;: 42, &#x27;Gather&#x27;: 4, &#x27;Add&#x27;: 14, &#x27;Div&#x27;: 4, &#x27;Slice&#x27;: 8, &#x27;Concat&#x27;: 11, &#x27;MaxPool&#x27;: 3, &#x27;Split&#x27;: 2, &#x27;Reshape&#x27;: 11, &#x27;Transpose&#x27;: 2, &#x27;MatMul&#x27;: 2, &#x27;Softmax&#x27;: 1, &#x27;Resize&#x27;: 1, &#x27;AveragePool&#x27;: 4, &#x27;Flatten&#x27;: 4, &#x27;Gemm&#x27;: 12, &#x27;Relu&#x27;: 4, &#x27;Tanh&#x27;: 4, &#x27;Unsqueeze&#x27;: 8}

| 边界 | Tensor | Shape | dtype |
|---|---|---|---|
| inputs | images | [1, 3, 640, 640] | float32 |
| outputs | cls_logits | [1, 161, 56, 4] | float32 |
| outputs | offset | [1, 1, 56, 4] | float32 |

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

Conv 节点 47；确认 Conv2D 并执行检查 47。

| 状态 | 个数 |
|---|---|
| VIOLATION | 0 |
| NEEDS_VERIFICATION | 0 |
| NO_VIOLATION_FOUND | 47 |
| NOT_COVERED | 0 |

`NO_VIOLATION_FOUND` 仅表示已检查规则未发现违规；非 Conv 节点不按通过统计。

## 4. 异常节点清单

| 节点 ID / 名称 | 规则 | 实际值 | 允许值 | 理由 |
|---|---|---|---|---|

未发现已检查规则违规。

## 5. 节点风险溯源

无确定违规源节点，无需生成违规溯源。

## 6. 优化建议

- **main/node_000000 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000003 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000006 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000023 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000026 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000031 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000034 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000037 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000054 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000057 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000062 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000065 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000068 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000085 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000088 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000091 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000095 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000098 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000102 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000106 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000110 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000113 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000116 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000133 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000136 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000139 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000143 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000146 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000150 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000154 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000158 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000161 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000166 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000170 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000174 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000189 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000191 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000193 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000196 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000199 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000202 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000208 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000212 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000215 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000229 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000243 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker
- **main/node_000257 / 无需修改**：目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。 证据：{&#x27;status&#x27;: &#x27;NO_VIOLATION_FOUND&#x27;}；规则：None；验证：实际工具链 checker

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
| model_outputs | 2 | 145152 | 0 | COMPLETE |
| intermediate_activations | 240 | 93475488 | 172 | PARTIAL |
| initializers | 118 | 104235968 | 0 | COMPLETE |
| model_inputs | 1 | 4915200 | 0 | COMPLETE |
| constants | 42 | 516 | 0 | COMPLETE |
| other_unknown | 0 | 0 | 0 | COMPLETE |

### 最大已知输出（前 10）

| Tensor | Shape | dtype | 精确 B | MiB | Consumers | 假设 INT8 B |
|---|---|---|---|---|---|---|
| cls_logits | [1, 161, 56, 4] | float32 | 144256 | 0.14 | 0 | 36064 |
| offset | [1, 1, 56, 4] | float32 | 896 | 0.00 | 0 | 224 |

### 最大已知中间激活（前 10）

| Tensor | Shape | dtype | 精确 B | MiB | Consumers | 假设 INT8 B |
|---|---|---|---|---|---|---|
| /model/model.0/act/Mul_output_0 | [1, 32, 320, 320] | float32 | 13107200 | 12.50 | 1 | 3276800 |
| /model/model.0/act/Sigmoid_output_0 | [1, 32, 320, 320] | float32 | 13107200 | 12.50 | 1 | 3276800 |
| /model/model.0/conv/Conv_output_0 | [1, 32, 320, 320] | float32 | 13107200 | 12.50 | 2 | 3276800 |
| /model/model.1/act/Mul_output_0 | [1, 64, 160, 160] | float32 | 6553600 | 6.25 | 1 | 1638400 |
| /model/model.1/act/Sigmoid_output_0 | [1, 64, 160, 160] | float32 | 6553600 | 6.25 | 1 | 1638400 |
| /model/model.1/conv/Conv_output_0 | [1, 64, 160, 160] | float32 | 6553600 | 6.25 | 2 | 1638400 |
| /model/model.2/cv1/act/Mul_output_0 | [1, 64, 160, 160] | float32 | 6553600 | 6.25 | 3 | 1638400 |
| /model/model.2/cv1/act/Sigmoid_output_0 | [1, 64, 160, 160] | float32 | 6553600 | 6.25 | 1 | 1638400 |
| /model/model.2/cv1/conv/Conv_output_0 | [1, 64, 160, 160] | float32 | 6553600 | 6.25 | 2 | 1638400 |
| /model/model.10/m/m.0/attn/MatMul_output_0 | [1, 4, 400, 400] | float32 | 2560000 | 2.44 | 1 | 640000 |

**Hypothetical INT8 raw-payload scenario（假设 INT8 原始载荷场景）**：仅元素数乘 1 B，不是实际量化结果或整模型压缩预测。
多消费者中间 Tensor：71；fanout 只表示图连接，不代表额外分配。

未知尺寸（最多列 10 个；完整内容查询 tensors）：

- /model/model.2/Slice_output_0：dim[0] symbolic: unk__0; dim[1] symbolic: unk__1; dim[2] symbolic: unk__2; dim[3] symbolic: unk__3
- /model/model.2/Slice_1_output_0：dim[0] symbolic: unk__4; dim[1] symbolic: unk__5; dim[2] symbolic: unk__6; dim[3] symbolic: unk__7
- /model/model.2/m.0/cv1/conv/Conv_output_0：dim[0] symbolic: unk__4; dim[2] symbolic: unk__8; dim[3] symbolic: unk__9
- /model/model.2/m.0/cv1/act/Sigmoid_output_0：dim[0] symbolic: unk__4; dim[2] symbolic: unk__8; dim[3] symbolic: unk__9
- /model/model.2/m.0/cv1/act/Mul_output_0：dim[0] symbolic: unk__4; dim[2] symbolic: unk__8; dim[3] symbolic: unk__9
- /model/model.2/m.0/cv2/conv/Conv_output_0：dim[0] symbolic: unk__4; dim[2] symbolic: unk__10; dim[3] symbolic: unk__11
- /model/model.2/m.0/cv2/act/Sigmoid_output_0：dim[0] symbolic: unk__4; dim[2] symbolic: unk__10; dim[3] symbolic: unk__11
- /model/model.2/m.0/cv2/act/Mul_output_0：dim[0] symbolic: unk__4; dim[2] symbolic: unk__10; dim[3] symbolic: unk__11
- /model/model.2/m.0/Add_output_0：dim[0] symbolic: unk__4; dim[2] symbolic: unk__12; dim[3] symbolic: unk__13
- /model/model.2/Concat_output_0：dim[0] symbolic: unk__0; dim[1] symbolic: unk__14; dim[2] symbolic: unk__2; dim[3] symbolic: unk__3

完整资源查询：`python -m rdkx5_doctor tensors --analysis analysis.json --limit 0`。

- Theoretical logical dense Tensor payload only; not measured runtime/BPU/DDR memory.
- Known sums are not peak memory: lifetimes, aliasing, reuse, layout, padding and fusion are unknown.
- Hypothetical INT8 raw-payload scenario only; not actual quantization, placement, accuracy or performance.
- bool uses an assumed one logical byte representation; packed/string/sparse/container types unsupported.
- Each Tensor is counted once; initializer/output overlaps use initializer as primary category.

## 10. Graph Optimization Candidates

**没有改写、删除或融合任何模型节点。结构候选不是新增的 X5 BPU 支持规则。**
候选/待验证观察共 0；分类：{}；模式：{}。

INSUFFICIENT_INFORMATION 是未确认观察，与已建立局部语义冗余明确分开。

未发现这五种模式的候选或信息不足观察。


- Structural/semantic candidates are not X5 operator compatibility rules.
- No rewrite, compiler fusion verification or measured performance gain.
- Reachability is computed on demand from saved Tensor edges. Imported opset &gt;23 is unreviewed.