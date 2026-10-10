# RDK X5 ONNX Doctor 分析报告

**尚未用实际工具链验证：未经过 OpenExplorer/hb_mapper 实测。**

## 1. 模型概况

- 文件：examples/demo.onnx
- SHA256：`9f1da4be1ddb5e3b3873daffc4d5734014bf593120e7812d7a4736b316b2b116`（仅 ONNX 文件）
- 节点：5；initializer：1；结构校验：checker_passed
- Opset：[{&#x27;domain&#x27;: &#x27;&#x27;, &#x27;version&#x27;: 11}]
- 算子计数：{&#x27;Conv&#x27;: 1, &#x27;Relu&#x27;: 1, &#x27;Identity&#x27;: 1, &#x27;Add&#x27;: 1, &#x27;Concat&#x27;: 1}

| 边界 | Tensor | Shape | dtype |
|---|---|---|---|
| inputs | image | [1, 2, 40, 8] | float32 |
| outputs | prediction | [1, 2, 9, 8] | float32 |
| outputs | auxiliary | [1, 4, 9, 8] | float32 |

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
| VIOLATION | 1 |
| NEEDS_VERIFICATION | 0 |
| NO_VIOLATION_FOUND | 0 |
| NOT_COVERED | 0 |

`NO_VIOLATION_FOUND` 仅表示已检查规则未发现违规；非 Conv 节点不按通过统计。

## 4. 异常节点清单

| 节点 ID / 名称 | 规则 | 实际值 | 允许值 | 理由 |
|---|---|---|---|---|
| main/node_000000 / oversized_kernel | X5-CONV2D-KERNEL-H (FAIL) | 32 | {&#x27;min&#x27;: 1, &#x27;max&#x27;: 31} | 卷积核空间维度超出官方范围 |

## 5. 节点风险溯源

### main/node_000000

依赖关系不等于下游违规、CPU 回退或精度下降

直接前驱：[]；直接后继：[&#x27;main/node_000001&#x27;, &#x27;main/node_000002&#x27;]
可达输出：[&#x27;prediction&#x27;, &#x27;auxiliary&#x27;]；其他路径省略：True

- 输出 `prediction`：main/node_000000 → main/node_000001 → main/node_000003
  - main/node_000000 → main/node_000001，Tensor：[&#x27;features&#x27;]
  - main/node_000001 → main/node_000003，Tensor：[&#x27;activated&#x27;]
- 输出 `auxiliary`：main/node_000000 → main/node_000001 → main/node_000004
  - main/node_000000 → main/node_000001，Tensor：[&#x27;features&#x27;]
  - main/node_000001 → main/node_000004，Tensor：[&#x27;activated&#x27;]

## 6. 优化建议

- **main/node_000000 / 可能需要改变网络并重训**：检查该节点导出属性 kernel_h 与训练结构是否一致；如为实际结构约束，评估替代卷积结构并重新训练或微调。评估将大卷积核改成多层较小卷积核；此方案通常改变计算与感受野组合，需要重新训练及任务指标对比。 不得直接删除节点。 证据：{&#x27;field&#x27;: &#x27;kernel_h&#x27;, &#x27;actual&#x27;: 32, &#x27;expected&#x27;: {&#x27;min&#x27;: 1, &#x27;max&#x27;: 31}}；规则：X5-CONV2D-KERNEL-H；验证：候选模型数值等价/任务指标验证，以及实际工具链 checker；无性能或精度承诺

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
