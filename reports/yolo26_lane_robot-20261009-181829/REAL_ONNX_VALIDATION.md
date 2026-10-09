# YOLO26 真实 ONNX 静态诊断验收报告

完成时间：2026-10-09T18:29:38+08:00。本报告是实际终端执行及节点级复核结果。

真实模型分析成功。初始 138 项测试通过；修复 Reshape 候选筛选缺陷并新增 3 项回归后，141 项通过。最终 281 个节点全部解析、47 个 Conv2D 已收录规则未发现违规、五种模式均无实际优化候选。172 个中间 Tensor 的尺寸未知，已保留原因；不能据此声称模型完全兼容 BPU或已掌握全图最大的 Tensor。

没有执行 Docker、hb_mapper、量化、结构优化、节点删除、板端推理、评分、GUI 或 Git commit/push。实际部署内存、数值等价、精度、CPU/BPU 划分和性能未测量。

## 1. 测试环境

| 项目 | 实际值 |
| --- | --- |
| 系统 | Ubuntu 22.04.5 LTS / WSL2 / x86_64 |
| Python | 3.10.12 |
| 解释器 | /home/xhm/lianghua_ws/skillzuoye/.venv/bin/python |
| 包版本 | rdkx5-onnx-doctor 0.2.0 / analysis schema 1.1 / ruleset 0.1.0 |
| Git | main / c10ac2ac7d672716e95c4b8b6e264048b5d6b4ec |
| 初始工作区 | 干净；无预先存在用户修改 |
| 模型 | `/home/xhm/lianghua_ws/rdktoolchain/horizon_x5_open_explorer_v1.2.8/horizon_x5_open_explorer_v1.2.8-py310_20240926/samples/ai_toolchain/horizon_model_convert_sample/04_detection/15_tasknav/model/yolo26_lane_robot.onnx` |
| 文件大小 | 104290060 B |
| SHA256（分析前） | `a8ea6ccf474c77614f33512c4a118390f3b14c28e4871615670888a6ca292de4` |
| SHA256（分析后） | `a8ea6ccf474c77614f33512c4a118390f3b14c28e4871615670888a6ca292de4` |
| 原始文件完整性 | 相同；仅只读加载，未移动、覆盖、删除或另存原始模型 |

依赖精确快照见 [environment.stdout](environment.stdout)。核心版本：ONNX 1.23.2、NumPy 2.2.6、NetworkX 3.4.2、Pydantic 2.14.0、PyYAML 6.0.3、protobuf 7.36.2、pytest 8.4.2、build 1.6.1。`pip check` 成功。

环境配置过程：初始 `.venv` 不存在。`python3 -m venv .venv` 因系统缺少 ensurepip 失败；sudo 安装 python3.10-venv 因需要密码未执行成功。随后使用 `venv --without-pip`、官方 https://bootstrap.pypa.io/get-pip.py 引导项目 pip，并完成 `.venv/bin/python -m pip install -e '.[dev]'`。下载依赖需要获准的沙箱外网络访问；系统 Python 未安装项目依赖。

首次 pytest 在收集前被系统 `PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages` 引入的 ROS launch_testing 插件阻塞，异常为 `ModuleNotFoundError: lark`。清除该外部 PYTHONPATH 后完整运行全部项目测试，未禁用项目测试或删减条件。首次失败的 stderr 也已保留。

## 2. 测试结果

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| CLI --help | 退出 0，九个子命令 | help.stdout |
| rules validate | 退出 0，19 条原规则 | rules_validate.stdout |
| 首次 pytest | 收集前失败：ROS 插件环境污染 | pytest_baseline.stderr |
| 隔离后基线 | 138 passed in 0.96s | pytest_isolated.stdout |
| 最小案例，修复前 | 2 failed / 1 passed（预期复现） | regression_before_fix.stdout |
| 最小案例，修复后 | 3 passed / 41 deselected | regression_after_fix.stdout |
| 最终完整回归 | 141 passed in 0.93s | pytest_final.stdout |
| 初始真实 analyze | 退出 0，0.935 秒 | before_fix/analyze.stdout |
| 修复后真实 analyze | 退出 0，0.922 秒 | analyze.stdout |
| 节点 / Tensor / 连边逐项比对 | PASS；与源 ONNX 一致，无遗漏节点 | independent_audit.json |
| 尺寸整数运算核对 | 231 个已知记录全部一致；172 个未知保留原因 | independent_audit.json |
| 修复影响范围核对 | 除 optimization_candidates 外全部 JSON 顶层段与修复前相同 | graph_evidence.json |
| 原始模型 SHA256 | 前后相同 | validation_metadata.json / independent_audit.json |

模型 checker 成功、shape inference 无异常、CLI 分析无异常终止。全部保存的查询（含修复前 9 次 candidate 查询）均退出 0；首次环境失败及预期失败的回归案例明确区分。未重新做 wheel 打包或实际运行推理；这些不属于本次静态模型验收范围。

复现命令（仓库根目录）：

```bash
env -u PYTHONPATH .venv/bin/pytest -q
.venv/bin/python -m rdkx5_doctor --help
.venv/bin/python -m rdkx5_doctor rules validate
.venv/bin/python -m rdkx5_doctor analyze --model '/home/xhm/lianghua_ws/rdktoolchain/horizon_x5_open_explorer_v1.2.8/horizon_x5_open_explorer_v1.2.8-py310_20240926/samples/ai_toolchain/horizon_model_convert_sample/04_detection/15_tasknav/model/yolo26_lane_robot.onnx' --out reports/yolo26-new-run
```

更多实际参数、耗时和退出码见 [command_log.jsonl](command_log.jsonl)，其中 `unset_env` 记录隔离 PYTHONPATH。验证脚本已随报告保存，不依赖模型名称硬编码到产品源码。

## 3. 真实模型结构

标准域 opset 11；1 个输入、2 个输出、281 个节点、23 种算子、118 个 initializer。节点 ID 为原始图顺序 `main/node_000000` 至 `main/node_000280`。

| 边界 | Tensor | Shape | dtype | 元素 | 原始 B |
| --- | --- | --- | --- | --- | --- |
| input | images | [1, 3, 640, 640] | float32 | 1228800 | 4915200 |
| output | cls_logits | [1, 161, 56, 4] | float32 | 36064 | 144256 |
| output | offset | [1, 1, 56, 4] | float32 | 224 | 896 |

| 算子 | 节点数 | 全部节点占比 | BPU 规则覆盖 |
| --- | --- | --- | --- |
| Mul | 51 | 18.15% | NOT_COVERED |
| Conv | 47 | 16.73% | Conv2D 已检查 |
| Constant | 42 | 14.95% | NOT_COVERED |
| Sigmoid | 38 | 13.52% | NOT_COVERED |
| Add | 14 | 4.98% | NOT_COVERED |
| Gemm | 12 | 4.27% | NOT_COVERED |
| Concat | 11 | 3.91% | NOT_COVERED |
| Reshape | 11 | 3.91% | NOT_COVERED |
| Slice | 8 | 2.85% | NOT_COVERED |
| Unsqueeze | 8 | 2.85% | NOT_COVERED |
| AveragePool | 4 | 1.42% | NOT_COVERED |
| Div | 4 | 1.42% | NOT_COVERED |
| Flatten | 4 | 1.42% | NOT_COVERED |
| Gather | 4 | 1.42% | NOT_COVERED |
| Relu | 4 | 1.42% | NOT_COVERED |
| Shape | 4 | 1.42% | NOT_COVERED |
| Tanh | 4 | 1.42% | NOT_COVERED |
| MaxPool | 3 | 1.07% | NOT_COVERED |
| MatMul | 2 | 0.71% | NOT_COVERED |
| Split | 2 | 0.71% | NOT_COVERED |
| Transpose | 2 | 0.71% | NOT_COVERED |
| Resize | 1 | 0.36% | NOT_COVERED |
| Softmax | 1 | 0.36% | NOT_COVERED |

非 Conv 共 22 种 / 234 个节点，全部为 NOT_COVERED，不能报告为兼容或通过。实际输出为 lane/task 分支的 cls_logits 与 offset；不是根据 YOLO 文件名套用常见检测输出格式。

原模型无 `value_info`（0 条），仅输入/输出提供静态 shape；边界不存在动态维度，自定义域节点 0、外部 initializer 0、缺失外部权重 0、嵌套子图警告 0。默认 shape inference 增补 282 条 value_info，其中合并边界后 113 条元信息静态、172 条含符号维度。它们是推断未求解的 `unk__*`，不能据此断言原模型实际存在动态输入。

只读、内存中的 ONNX `strict_mode=True,data_prop=True` 对照实验仍产生 172 条含符号维度的元信息，与默认推断相同；未保存或优化模型。缺失中间元信息 / 推断不完备是资源分析限制，不是 shape inference 抛出异常。详见 [independent_audit.json](independent_audit.json)。

## 4. Tensor 资源分析

全图去重 403 条 Tensor 记录：344 条 float32、59 条 int64；无 FP16 实际记录。已知 231 条、未知 172 条，均由独立整数 oracle 核对。字节数 = 所有静态维度乘积 × dtype 字节宽度；FP32 为 4 B、INT64 为 8 B，假设 INT8 为元素数 × 1 B。FP16 若相同 shape，则逻辑载荷为元素数 × 2 B，仅为类型算术说明。

| 类别 | Tensor 数 | 已知原始 B 之和 | 未知数 | 完整性 |
| --- | --- | --- | --- | --- |
| initializers | 118 | 104235968 | 0 | COMPLETE |
| model_outputs | 2 | 145152 | 0 | COMPLETE |
| model_inputs | 1 | 4915200 | 0 | COMPLETE |
| constants | 42 | 516 | 0 | COMPLETE |
| intermediate_activations | 240 | 93475488 | 172 | PARTIAL |
| other_unknown | 0 | 0 | 0 | COMPLETE |

以上是按类别去重后的逻辑载荷之和；中间已知之和 93,475,488 B 是 PARTIAL。它不是峰值内存，也不能与权重求和推断实际 BPU、DDR 或 SRAM 占用。fanout 不表示复制次数。

### 全部模型输出

| Tensor | Shape / dtype | 元素 | 原始 B / MiB | 假设 INT8 B | 生产节点 / 消费者 |
| --- | --- | --- | --- | --- | --- |
| cls_logits | [1, 161, 56, 4] / float32 | 36064 | 144256 / 0.1375732421875 | 36064 | `main/node_000275` / `/model/model.16/Concat` / [] |
| offset | [1, 1, 56, 4] / float32 | 224 | 896 / 0.0008544921875 | 224 | `main/node_000280` / `/model/model.16/Concat_1` / [] |

两个输出都无消费者，分别由末端 Concat 275 / 280 产生。输出总原始载荷 145,152 B，假设 INT8 总载荷 36,288 B；不是实际量化结果。

### 已知原始载荷最大的 10 个中间 Tensor

因 172 个中间项未知，该表只保证在已知集合中排序；不能保证是全图实际最大的十个。相同字节数按名称排序，注意力的其它等大 Tensor 因 limit=10 未入选。

| Tensor | Shape / dtype | 元素 | 原始 B / MiB | 假设 INT8 B | 生产节点 | 消费节点 |
| --- | --- | --- | --- | --- | --- | --- |
| `/model/model.0/act/Mul_output_0` | [1, 32, 320, 320] / float32 | 3276800 | 13107200 / 12.5 | 3276800 | `main/node_000002` / `/model/model.0/act/Mul` | `main/node_000003` / `/model/model.1/conv/Conv` |
| `/model/model.0/act/Sigmoid_output_0` | [1, 32, 320, 320] / float32 | 3276800 | 13107200 / 12.5 | 3276800 | `main/node_000001` / `/model/model.0/act/Sigmoid` | `main/node_000002` / `/model/model.0/act/Mul` |
| `/model/model.0/conv/Conv_output_0` | [1, 32, 320, 320] / float32 | 3276800 | 13107200 / 12.5 | 3276800 | `main/node_000000` / `/model/model.0/conv/Conv` | `main/node_000001` / `/model/model.0/act/Sigmoid`, `main/node_000002` / `/model/model.0/act/Mul` |
| `/model/model.1/act/Mul_output_0` | [1, 64, 160, 160] / float32 | 1638400 | 6553600 / 6.25 | 1638400 | `main/node_000005` / `/model/model.1/act/Mul` | `main/node_000006` / `/model/model.2/cv1/conv/Conv` |
| `/model/model.1/act/Sigmoid_output_0` | [1, 64, 160, 160] / float32 | 1638400 | 6553600 / 6.25 | 1638400 | `main/node_000004` / `/model/model.1/act/Sigmoid` | `main/node_000005` / `/model/model.1/act/Mul` |
| `/model/model.1/conv/Conv_output_0` | [1, 64, 160, 160] / float32 | 1638400 | 6553600 / 6.25 | 1638400 | `main/node_000003` / `/model/model.1/conv/Conv` | `main/node_000004` / `/model/model.1/act/Sigmoid`, `main/node_000005` / `/model/model.1/act/Mul` |
| `/model/model.2/cv1/act/Mul_output_0` | [1, 64, 160, 160] / float32 | 1638400 | 6553600 / 6.25 | 1638400 | `main/node_000008` / `/model/model.2/cv1/act/Mul` | `main/node_000009` / `/model/model.2/Shape`, `main/node_000019` / `/model/model.2/Slice`, `main/node_000022` / `/model/model.2/Slice_1` |
| `/model/model.2/cv1/act/Sigmoid_output_0` | [1, 64, 160, 160] / float32 | 1638400 | 6553600 / 6.25 | 1638400 | `main/node_000007` / `/model/model.2/cv1/act/Sigmoid` | `main/node_000008` / `/model/model.2/cv1/act/Mul` |
| `/model/model.2/cv1/conv/Conv_output_0` | [1, 64, 160, 160] / float32 | 1638400 | 6553600 / 6.25 | 1638400 | `main/node_000006` / `/model/model.2/cv1/conv/Conv` | `main/node_000007` / `/model/model.2/cv1/act/Sigmoid`, `main/node_000008` / `/model/model.2/cv1/act/Mul` |
| `/model/model.10/m/m.0/attn/MatMul_output_0` | [1, 4, 400, 400] / float32 | 640000 | 2560000 / 2.44140625 | 640000 | `main/node_000179` / `/model/model.10/m/m.0/attn/MatMul` | `main/node_000181` / `/model/model.10/m/m.0/attn/Mul` |

首层三个中间项均为 [1,32,320,320] FP32：3,276,800 元素、13,107,200 B = 12.5 MiB，假设 INT8 为 3,276,800 B = 3.125 MiB。它们分别为 Conv、Sigmoid、Mul 的逻辑输出；不能断言运行时同时分配三个独立同大小缓冲区。

### 多消费者与未知资源

71 个多消费者中间 Tensor：13 个大小已知，58 个未知。完整列表及全部 172 个未知记录、原因、producer/consumer 在 [graph_evidence.json](graph_evidence.json)；CLI 原始列表见 [tensors_unknown.stdout](tensors_unknown.stdout)。

| 例子 | 生产者 | 消费者 | 解释 |
| --- | --- | --- | --- |
| `/model/model.0/conv/Conv_output_0` | `main/node_000000` / `/model/model.0/conv/Conv` | `main/node_000001` / `/model/model.0/act/Sigmoid`, `main/node_000002` / `/model/model.0/act/Mul` | Sigmoid 与 Mul 共享 Conv 值，属于 SiLU 结构 |
| `/model/model.2/cv1/act/Mul_output_0` | `main/node_000008` / `/model/model.2/cv1/act/Mul` | `main/node_000009` / `/model/model.2/Shape`, `main/node_000019` / `/model/model.2/Slice`, `main/node_000022` / `/model/model.2/Slice_1` | Shape 与两个 Slice 共用特征图；未知 shape 的起点位于下游 Slice |

172 条未知全为 float32 的符号维度，非 dtype 宽度未知；0 条缺 rank、0 条 dtype 缺失、0 条 unsupported dtype。按符号数量分组：110 条 rank4/3 个符号轴、26 条 rank4/1 个符号轴、12 条 rank4/4 个符号轴、24 条 rank2/1 个符号轴。

| 未知项生产算子 | 数量 |
| --- | --- |
| Conv | 42 |
| Mul | 35 |
| Sigmoid | 35 |
| Gemm | 12 |
| Add | 9 |
| Concat | 9 |
| Slice | 8 |
| AveragePool | 4 |
| Flatten | 4 |
| Relu | 4 |
| Tanh | 4 |
| MaxPool | 3 |
| Split | 2 |
| Resize | 1 |

首个未知项为 node 19 的 `/model/model.2/Slice_output_0`，shape 为 [unk__0,unk__1,unk__2,unk__3]；node 22 的另一个 Slice 类似。node 8 的输入特征已知 [1,64,160,160]，node 9 Shape、node 11 Gather(axis=0) 读取通道轴，node 14 Add / 16 Div / 18 Mul 计算 Slice 终点。小型内联整数证据为 axis=1、start=0、Add=1、Div=2、Mul=1；另一路 Mul=2。因此这一局部数学链可算出终点 32 / 64，两个通道切片局部应为 [1,32,160,160]。这是手工证据解释，不写回工具 JSON，不将后续未逐项证明的尺寸当作已知。

V1.1 未做通用形状算术传播，ONNX 默认/增强 data_prop 推断都在该链未恢复静态空间维度。不通过权重或模型输出反向猜测未知输入 shape；例如 node 23 权重 [16,32,3,3] 能独立检查 kernel，却不足以让工具编造其 batch/H/W。

## 5. RDK X5 BPU 约束检查

47 个 Conv 全部确认 rank4 Conv2D；结果为 NO_VIOLATION_FOUND=47、VIOLATION=0、NEEDS_VERIFICATION=0、Conv NOT_COVERED=0。逐规则 517 PASS、376 NOT_APPLICABLE、0 FAIL、0 UNKNOWN。NOT_APPLICABLE 是触发条件不成立，不能把它报告为数值 PASS。

| 规则 ID | 状态计数 |
| --- | --- |
| X5-CONV2D-KERNEL-H | {'PASS': 47} |
| X5-CONV2D-KERNEL-W | {'PASS': 47} |
| X5-CONV2D-KERNEL-VOLUME | {'PASS': 47} |
| X5-CONV2D-STRIDE-H | {'PASS': 47} |
| X5-CONV2D-DILATION-H | {'PASS': 47} |
| X5-CONV2D-STRIDE-W | {'PASS': 47} |
| X5-CONV2D-DILATION-W | {'PASS': 47} |
| X5-CONV2D-PAD-TOP | {'PASS': 47} |
| X5-CONV2D-PAD-LEFT | {'PASS': 47} |
| X5-CONV2D-PAD-BOTTOM | {'PASS': 47} |
| X5-CONV2D-PAD-RIGHT | {'PASS': 47} |
| X5-CONV2D-DILATED-STRIDE-H | {'NOT_APPLICABLE': 47} |
| X5-CONV2D-DILATED-DIVISIBLE-H | {'NOT_APPLICABLE': 47} |
| X5-CONV2D-DILATED-STRIDE-W | {'NOT_APPLICABLE': 47} |
| X5-CONV2D-DILATED-DIVISIBLE-W | {'NOT_APPLICABLE': 47} |
| X5-CONV2D-DILATED-OUTPUT | {'NOT_APPLICABLE': 47} |
| X5-CONV2D-CHANNEL-IN_CHANNELS_PER_GROUP | {'NOT_APPLICABLE': 47} |
| X5-CONV2D-CHANNEL-OUT_CHANNELS_PER_GROUP | {'NOT_APPLICABLE': 47} |
| X5-CONV2D-CHANNEL-OUT_CHANNELS | {'NOT_APPLICABLE': 47} |

实测字段范围：kernel H/W 1–3；每组 C×H×W 9–2304；stride H/W 1–2；dilation H/W 均 1；所有 padding 每侧 0–1；每组输入通道 1–1024、每组输出通道 1–512。所有 dilation 条件和超常规通道条件都未触发，故量化输出 dtype 条件没有在本模型产生 UNKNOWN。

来源证据按安装的规则包引用，未擅自更改、扩展或声称重新审核官方网页：
- [RDK X3/X5 DOC / 模型算子支持列表](https://developer.d-robotics.cc/rdk_x_doc/Advanced_development/toolchain_development/intermediate/supported_op_list)；网页最后更新 2026-08-04；检索快照 2026-10-09；RDK X5 支持的 ONNX 算子列表 / Conv / 四维输入（conv2d）/ X5 BPU 支持约束
- [X5 芯片用户手册](https://developer.d-robotics.cc/x5_sdk_doc/toolchain_development/intermediate/supported_op_list.html)；1.1.2；ONNX / Conv
- [X5 Chip User Manual](https://developer.d-robotics.cc/x5_sdk_doc_v2.0.0/en/toolchain_development/intermediate/supported_op_list.html)；2.0.0；ONNX / Conv

仓库审查记录见 `references/official_sources.md`、`references/conv_rule_notes.md`；规则 package 0.1.0 / toolchain_version=unverified。基础约束分别为 kernel 1–31、每组体积 ≤32767、stride 1–256、dilation 1–16、padding 0–256。node 0 的原始名称 `/model/model.0/conv/Conv`、权重 [32,3,3,3]、输入 [1,3,640,640]、stride [2,2]、pads [1,1,1,1]，逐规则证据在 [inspect_node_000000.stdout](inspect_node_000000.stdout)。

42 个 Conv 的输入存在推断符号轴，但本次触发的 kernel/stride/dilation/pad 规则可由静态权重或属性判定；它们的资源大小仍未知。没有确定违规源，故违规溯源不适用，`nodes --status VIOLATION` 返回空列表。检查到输出的依赖路径，不赋予下游违规或 CPU 回退含义。

未覆盖：全部非 Conv 的硬件支持；shape 总大小/字节上限；通道量化子图末端例外；Conv→Add shortcut 版本冲突；编译器融合；实际量化精度与布局。未覆盖不等于失败，已检查未违规不等于完整 BPU 兼容保证。

## 6. 五种优化候选检测

| 模式 | 真实结构 | 最终候选数量 / 分类 | 结论 |
| --- | --- | --- | --- |
| IDENTITY | Identity 节点 0 | 0 | 无 |
| TRANSPOSE_INVERSE_PAIR | Transpose 节点 2 | 0 | 中间隔着 MatMul、Mul、Softmax，不是直接逆序对 |
| CAST_SAME_DTYPE | Cast 节点 0 | 0 | 无 |
| RESHAPE_NOOP | Reshape 节点 11 | 0 | 11 个均有已知 rank 或轴变化证据 |
| CONV_BN_FUSION_REVIEW | BatchNormalization 节点 0 | 0 | 无可检测的 Conv→BN；不能据此声称编译器已融合 |

最终 SEMANTICALLY_REDUNDANT=0、REVIEW_REQUIRED=0、INSUFFICIENT_INFORMATION=0；没有值得删除或融合的已确认候选。不是为了凑齐五类制造候选。

### 修复前的 9 个信息不足观察及逐节点排除证据

以下 ID 属于 [before_fix/analysis.json](before_fix/analysis.json)，最终报告中已排除。初始工具没有把它们误标成已确认冗余，但忽略可用反证导致无意义观察。每条均实际调用了 candidate、inspect 和 trace；candidate 输出在 before_fix/candidate_OPT-*.stdout，接口、消费者、输出路径均已核对。

#### OPT-0001 / main/node_000176

- 原名：`/model/model.10/m/m.0/attn/Reshape`；原分类 INSUFFICIENT_INFORMATION。
- 输入值：`/model/model.10/m/m.0/attn/qkv/conv/Conv_output_0`，shape `['unk__75', 512, 'unk__118', 'unk__119']`；producer `main/node_000174` / `/model/model.10/m/m.0/attn/qkv/conv/Conv`。
- shape 输入：`/model/model.10/m/m.0/attn/Constant_output_0`，inline INT64 常量 `[1, 4, 128, 400]`。
- 输出：`/model/model.10/m/m.0/attn/Reshape_output_0`，shape `[1, 4, 128, 400]`；consumer `main/node_000177` / `/model/model.10/m/m.0/attn/Split`。
- 识别/排除：初始因符号轴无法计算完整输入元素数而列观察；人工反证为 已知输入第二轴 512 与目标第二轴 4 不同，确定改变 shape；修复后排除。
- 结构限制：输入值 fanout=1，输出 fanout=1，该 Reshape 输出非公开模型输出；最终可达 `['cls_logits', 'offset']`。输入值没有其它消费支路。
- 语义与研究价值：不能当作无变化转发删除。当前 no-op 优化不值得继续研究；如研究其它等价图改写，必须保留目标 rank/shape、所有消费者、公开输出名称及 dtype，在独立副本上做 checker、shape/interface 与 ONNX Runtime 代表输入数值及任务指标验证。本次未进行这些运行时验证。

#### OPT-0002 / main/node_000222

- 原名：`/model/model.16/task_branches.0/Reshape`；原分类 INSUFFICIENT_INFORMATION。
- 输入值：`/model/model.16/task_branches.0/cls_fc2/Gemm_output_0`，shape `['unk__42', 9016]`；producer `main/node_000220` / `/model/model.16/task_branches.0/cls_fc2/Gemm`。
- shape 输入：`/model/model.16/task_branches.0/Constant_output_0`，inline INT64 常量 `[1, 161, 56]`。
- 输出：`/model/model.16/task_branches.0/Reshape_output_0`，shape `[1, 161, 56]`；consumer `main/node_000271` / `/model/model.16/Unsqueeze`。
- 识别/排除：初始因符号轴无法计算完整输入元素数而列观察；人工反证为 输入 rank2、目标 rank3，确定改变接口 shape；修复后排除。
- 结构限制：输入值 fanout=1，输出 fanout=1，该 Reshape 输出非公开模型输出；最终可达 `['cls_logits']`。输入值没有其它消费支路。
- 语义与研究价值：不能当作无变化转发删除。当前 no-op 优化不值得继续研究；如研究其它等价图改写，必须保留目标 rank/shape、所有消费者、公开输出名称及 dtype，在独立副本上做 checker、shape/interface 与 ONNX Runtime 代表输入数值及任务指标验证。本次未进行这些运行时验证。

#### OPT-0003 / main/node_000226

- 原名：`/model/model.16/task_branches.0/Reshape_1`；原分类 INSUFFICIENT_INFORMATION。
- 输入值：`/model/model.16/task_branches.0/Tanh_output_0`，shape `['unk__42', 56]`；producer `main/node_000224` / `/model/model.16/task_branches.0/Tanh`。
- shape 输入：`/model/model.16/task_branches.0/Constant_1_output_0`，inline INT64 常量 `[1, 1, 56]`。
- 输出：`/model/model.16/task_branches.0/Reshape_1_output_0`，shape `[1, 1, 56]`；consumer `main/node_000228` / `/model/model.16/task_branches.0/Mul`。
- 识别/排除：初始因符号轴无法计算完整输入元素数而列观察；人工反证为 输入 rank2、目标 rank3，确定改变接口 shape；修复后排除。
- 结构限制：输入值 fanout=1，输出 fanout=1，该 Reshape 输出非公开模型输出；最终可达 `['offset']`。输入值没有其它消费支路。
- 语义与研究价值：不能当作无变化转发删除。当前 no-op 优化不值得继续研究；如研究其它等价图改写，必须保留目标 rank/shape、所有消费者、公开输出名称及 dtype，在独立副本上做 checker、shape/interface 与 ONNX Runtime 代表输入数值及任务指标验证。本次未进行这些运行时验证。

#### OPT-0004 / main/node_000236

- 原名：`/model/model.16/task_branches.1/Reshape`；原分类 INSUFFICIENT_INFORMATION。
- 输入值：`/model/model.16/task_branches.1/cls_fc2/Gemm_output_0`，shape `['unk__42', 9016]`；producer `main/node_000234` / `/model/model.16/task_branches.1/cls_fc2/Gemm`。
- shape 输入：`/model/model.16/task_branches.1/Constant_output_0`，inline INT64 常量 `[1, 161, 56]`。
- 输出：`/model/model.16/task_branches.1/Reshape_output_0`，shape `[1, 161, 56]`；consumer `main/node_000272` / `/model/model.16/Unsqueeze_1`。
- 识别/排除：初始因符号轴无法计算完整输入元素数而列观察；人工反证为 输入 rank2、目标 rank3，确定改变接口 shape；修复后排除。
- 结构限制：输入值 fanout=1，输出 fanout=1，该 Reshape 输出非公开模型输出；最终可达 `['cls_logits']`。输入值没有其它消费支路。
- 语义与研究价值：不能当作无变化转发删除。当前 no-op 优化不值得继续研究；如研究其它等价图改写，必须保留目标 rank/shape、所有消费者、公开输出名称及 dtype，在独立副本上做 checker、shape/interface 与 ONNX Runtime 代表输入数值及任务指标验证。本次未进行这些运行时验证。

#### OPT-0005 / main/node_000240

- 原名：`/model/model.16/task_branches.1/Reshape_1`；原分类 INSUFFICIENT_INFORMATION。
- 输入值：`/model/model.16/task_branches.1/Tanh_output_0`，shape `['unk__42', 56]`；producer `main/node_000238` / `/model/model.16/task_branches.1/Tanh`。
- shape 输入：`/model/model.16/task_branches.1/Constant_1_output_0`，inline INT64 常量 `[1, 1, 56]`。
- 输出：`/model/model.16/task_branches.1/Reshape_1_output_0`，shape `[1, 1, 56]`；consumer `main/node_000242` / `/model/model.16/task_branches.1/Mul`。
- 识别/排除：初始因符号轴无法计算完整输入元素数而列观察；人工反证为 输入 rank2、目标 rank3，确定改变接口 shape；修复后排除。
- 结构限制：输入值 fanout=1，输出 fanout=1，该 Reshape 输出非公开模型输出；最终可达 `['offset']`。输入值没有其它消费支路。
- 语义与研究价值：不能当作无变化转发删除。当前 no-op 优化不值得继续研究；如研究其它等价图改写，必须保留目标 rank/shape、所有消费者、公开输出名称及 dtype，在独立副本上做 checker、shape/interface 与 ONNX Runtime 代表输入数值及任务指标验证。本次未进行这些运行时验证。

#### OPT-0006 / main/node_000250

- 原名：`/model/model.16/task_branches.2/Reshape`；原分类 INSUFFICIENT_INFORMATION。
- 输入值：`/model/model.16/task_branches.2/cls_fc2/Gemm_output_0`，shape `['unk__42', 9016]`；producer `main/node_000248` / `/model/model.16/task_branches.2/cls_fc2/Gemm`。
- shape 输入：`/model/model.16/task_branches.2/Constant_output_0`，inline INT64 常量 `[1, 161, 56]`。
- 输出：`/model/model.16/task_branches.2/Reshape_output_0`，shape `[1, 161, 56]`；consumer `main/node_000273` / `/model/model.16/Unsqueeze_2`。
- 识别/排除：初始因符号轴无法计算完整输入元素数而列观察；人工反证为 输入 rank2、目标 rank3，确定改变接口 shape；修复后排除。
- 结构限制：输入值 fanout=1，输出 fanout=1，该 Reshape 输出非公开模型输出；最终可达 `['cls_logits']`。输入值没有其它消费支路。
- 语义与研究价值：不能当作无变化转发删除。当前 no-op 优化不值得继续研究；如研究其它等价图改写，必须保留目标 rank/shape、所有消费者、公开输出名称及 dtype，在独立副本上做 checker、shape/interface 与 ONNX Runtime 代表输入数值及任务指标验证。本次未进行这些运行时验证。

#### OPT-0007 / main/node_000254

- 原名：`/model/model.16/task_branches.2/Reshape_1`；原分类 INSUFFICIENT_INFORMATION。
- 输入值：`/model/model.16/task_branches.2/Tanh_output_0`，shape `['unk__42', 56]`；producer `main/node_000252` / `/model/model.16/task_branches.2/Tanh`。
- shape 输入：`/model/model.16/task_branches.2/Constant_1_output_0`，inline INT64 常量 `[1, 1, 56]`。
- 输出：`/model/model.16/task_branches.2/Reshape_1_output_0`，shape `[1, 1, 56]`；consumer `main/node_000256` / `/model/model.16/task_branches.2/Mul`。
- 识别/排除：初始因符号轴无法计算完整输入元素数而列观察；人工反证为 输入 rank2、目标 rank3，确定改变接口 shape；修复后排除。
- 结构限制：输入值 fanout=1，输出 fanout=1，该 Reshape 输出非公开模型输出；最终可达 `['offset']`。输入值没有其它消费支路。
- 语义与研究价值：不能当作无变化转发删除。当前 no-op 优化不值得继续研究；如研究其它等价图改写，必须保留目标 rank/shape、所有消费者、公开输出名称及 dtype，在独立副本上做 checker、shape/interface 与 ONNX Runtime 代表输入数值及任务指标验证。本次未进行这些运行时验证。

#### OPT-0008 / main/node_000264

- 原名：`/model/model.16/task_branches.3/Reshape`；原分类 INSUFFICIENT_INFORMATION。
- 输入值：`/model/model.16/task_branches.3/cls_fc2/Gemm_output_0`，shape `['unk__42', 9016]`；producer `main/node_000262` / `/model/model.16/task_branches.3/cls_fc2/Gemm`。
- shape 输入：`/model/model.16/task_branches.3/Constant_output_0`，inline INT64 常量 `[1, 161, 56]`。
- 输出：`/model/model.16/task_branches.3/Reshape_output_0`，shape `[1, 161, 56]`；consumer `main/node_000274` / `/model/model.16/Unsqueeze_3`。
- 识别/排除：初始因符号轴无法计算完整输入元素数而列观察；人工反证为 输入 rank2、目标 rank3，确定改变接口 shape；修复后排除。
- 结构限制：输入值 fanout=1，输出 fanout=1，该 Reshape 输出非公开模型输出；最终可达 `['cls_logits']`。输入值没有其它消费支路。
- 语义与研究价值：不能当作无变化转发删除。当前 no-op 优化不值得继续研究；如研究其它等价图改写，必须保留目标 rank/shape、所有消费者、公开输出名称及 dtype，在独立副本上做 checker、shape/interface 与 ONNX Runtime 代表输入数值及任务指标验证。本次未进行这些运行时验证。

#### OPT-0009 / main/node_000268

- 原名：`/model/model.16/task_branches.3/Reshape_1`；原分类 INSUFFICIENT_INFORMATION。
- 输入值：`/model/model.16/task_branches.3/Tanh_output_0`，shape `['unk__42', 56]`；producer `main/node_000266` / `/model/model.16/task_branches.3/Tanh`。
- shape 输入：`/model/model.16/task_branches.3/Constant_1_output_0`，inline INT64 常量 `[1, 1, 56]`。
- 输出：`/model/model.16/task_branches.3/Reshape_1_output_0`，shape `[1, 1, 56]`；consumer `main/node_000270` / `/model/model.16/task_branches.3/Mul`。
- 识别/排除：初始因符号轴无法计算完整输入元素数而列观察；人工反证为 输入 rank2、目标 rank3，确定改变接口 shape；修复后排除。
- 结构限制：输入值 fanout=1，输出 fanout=1，该 Reshape 输出非公开模型输出；最终可达 `['offset']`。输入值没有其它消费支路。
- 语义与研究价值：不能当作无变化转发删除。当前 no-op 优化不值得继续研究；如研究其它等价图改写，必须保留目标 rank/shape、所有消费者、公开输出名称及 dtype，在独立副本上做 checker、shape/interface 与 ONNX Runtime 代表输入数值及任务指标验证。本次未进行这些运行时验证。

另外 node 187 / 188 的 Reshape：输入 [1,4,64,400]，目标 [1,256,20,20]，元素数虽相同，轴结构不同，初始与最终都没有列候选。两次 Transpose 的 perm 均 [0,1,3,2]，但 node178→179(MatMul)→181(Mul)→182(Softmax)→183，不能跨计算操作抵消；node183对 Softmax 结果做转置。

## 7. Skill 自主深入诊断结果

实际应用仓库 `.agents/skills/rdk-x5-onnx-doctor/SKILL.md`，按诊断事实选择节点，并未将所有算子套用相同建议。

| 触发证据 | 自主选取与核对 | 结果 |
| --- | --- | --- |
| 最大已知载荷及 Conv fanout | tensor 查询首层三个输出，inspect/trace node0/1/2/3 | Conv→Sigmoid→Mul，node0值被 node1/2共享；全部可达两个模型输出，不能按三个载荷直接相加推峰值 |
| 第一个未知 Slice | node8/9/11/14/16/18/19/22 及内联 shape 标量 | 静态特征遇到 shape 算术 Slice 后产生 unk 符号；data_prop对照不消除 |
| Top10 中注意力矩阵 | node176/177/178/179/181/182/183/184/187/188 | [1,4,400,400] FP32 2,560,000 B；多次计算/transpose不能被误认逆 Transpose |
| 9 个信息不足观察 | 全部 candidate 查询，再查 producer/consumer 和 shape目标 | 已知轴或 rank 能排除 no-op；最小修复后 9→0 |
| 高资源路径与候选关联 | 从 node2/8 到初始候选节点最短依赖路径 | 早期大特征可达注意力和四个任务分支；属于路径相关，未发现与确定违规或最终候选的关系 |
| Conv 状态全通过但多数资源未知 | Conv 逐规则与 tensor shape交叉核对 | BPU静态属性检查范围与资源shape完整性不同；不把未知资源解释为违规 |

路径证据见 [graph_evidence.json](graph_evidence.json) 的 selected_shortest_paths，每段包含真实 Tensor 标签。node2 为最大并列 Tensor 的生产者，直接上游 node0/1、下游 node3；trace 对 cls_logits 最短代表路径含 39 个节点，对 offset 含 41 个节点，其他分叉路径不枚举（paths_truncated=true）。

注意力路径 node176→177(Split)→178(Transpose)→179(MatMul)→181(Mul)→182(Softmax)→183(Transpose)→184(MatMul)→187(Reshape)；此外 node177 输出 V 被 node184与node188共享。node179、181、182、183 的输出都为 [1,4,400,400] FP32，2,560,000 B = 2.44140625 MiB；假设 INT8 为 640,000 B。它们并非全都出现在限制十条的 Top10 表里。

node222 的任务0分类路径是 node220(Gemm)→222(Reshape)→271(Unsqueeze)→275(Concat/cls_logits)；offset 支路 node224(Tanh)→226(Reshape)→228(Mul)→276(Unsqueeze)→280(Concat/offset)。因此 Reshape 负责终端 shape 组装，不能因 batch 符号未知就提出删除它。其余三任务分支有对应明确路径。

实际发现：节点/连边/尺寸数学一致、shape推断限制、冗余观察筛选缺陷。局部理论推导：首个 Slice 的通道边界可由有界标量算出。未验证假设：编译器是否融合 SiLU/Conv、是否复用缓冲、attention 与其它算子是否落 BPU、真实 DDR/SRAM、延迟、量化误差。路径相关不能证明后者。

## 8. Bug 根因、修复与回归

Bug：Reshape 检测器先要求 input shape 所有维度静态；遇到符号轴即报告 INSUFFICIENT_INFORMATION，即使已知 rank或其它轴已证明改变 shape。这是候选筛选缺陷，不是已确认冗余的误报。真实模型 8 个 rank2→rank3 与 1 个已知512→4 暴露该问题。

最小案例使用 tiny ONNX：输入 [batch,6] → Reshape 目标 [1,2,3]；输入 [batch,512,height,width] → [1,4,128,400]。修复前两例均产生无必要观察，回归断言均失败。对照 [batch,6] → [1,6] 仍不能证明no-op，必须保留信息不足。

| 修改文件 | 内容 |
| --- | --- |
| src/rdkx5_doctor/optimization_candidates.py | 12行：已审查标准语义+已知一维INT64正目标下，以rank/已知轴变化作为否定判据；不猜符号轴，不重写图 |
| tests/test_optimization_candidates.py | 新增参数化回归3项，验证rank改变、已知轴改变、真正未知仍保留 |
| docs/ANALYSIS_SCHEMA_V1_1.md | 说明符号shape下的已知反证筛选 |
| docs/DEVELOPMENT_LOG.md | 记录真实模型验收、环境失败、修复与限制 |

原有 138 项测试全部保留，完整结果 141 passed。没有变更 Conv2D YAML、通道或其它官方BPU限制。修复前/后 analysis 对比除 optimization_candidates 外每个顶层段完全相同。源码差异见 [source_changes.diff](source_changes.diff)。

## 9. V1.2 建议（仅建议，本次未开发新规则）

| 优先事项 | 真实证据 | 下一版应做什么 |
| --- | --- | --- |
| P0 有界 shape算术传播 | node9→11→14→16→18→19 链使172个Tensor未知；data_prop无改善 | 仅为Shape/Gather及小整数Add/Div/Mul/Slice传播静态shape事实，限制元素数/编码/整数语义，保持动态与无法证明值未知；无需模型结构优化 |
| P0 逐算子shape完整性解释 | 原图没有value_info；42个Conv输出等未知 | 显示原始/推断元信息来源及不确定传播起点；Top10醒目标注仅已知集合，避免误解全图最大 |
| P1 elementwise规则覆盖 | Mul51、Sigmoid38、Add14，大量SiLU/残差和广播链 | 核对对应工具链版本官方X5的dtype/广播/shape约束，再扩展逐规则证据；不凭结构猜CPU/BPU分配 |
| P1 Slice/Concat/Reshape/Transpose/Resize | Slice8、Concat11、Reshape11、Transpose2、Resize1 | 优先查官方轴/参数/版本/布局限制，特别Resize asymmetric/nearest/floor；保持支持条件UNKNOWN |
| P1 MatMul/Softmax/Gemm | attention两个MatMul+Softmax，末端12个Gemm | 核对官方rank、axis、transpose、dtype、常量参数条件；绑定node179/182/184和task分支 |
| P2 Pool/Flatten/Split等 | MaxPool3、AveragePool4、Flatten4、Split2 | 补齐真实图中剩余算子规则，并分清shape解析成功与BPU支持证据 |
| P2 候选否定证据与解释 | 本次排除9个观察，两个等元素Reshape也不冗余 | 可单独提供排除理由，扩展0/-1部分shape否定判据时增加版本/allowzero边界回归，保持不自动改写 |

新硬件条款必须先按目标工具链版本核对官方X5 BPU列，再升级规则包及边界测试；不得将CPU列、其它芯片、Tensor原始B或推测当作新BPU限制。本次源路径含 OpenExplorer 1.2.8 字样，只说明文件所在目录，不说明已运行或已验证该工具链。

## 10. 文件与验收

- [analysis.json](analysis.json)：修复后的完整机器事实。
- [report.md](report.md)：现有CLI生成的最终标准报告。
- [before_fix/analysis.json](before_fix/analysis.json)：初始9个观察及候选ID的历史证据。
- [independent_audit.json](independent_audit.json)：源ONNX逐节点/边界对照、尺寸oracle、规则计数与SHA256。
- [graph_evidence.json](graph_evidence.json)：完整未知/多消费者列表、局部常量和跨节点Tensor依赖路径。
- [command_log.jsonl](command_log.jsonl)：实际CLI参数、退出码、耗时；同名stdout/stderr文件保存内容。
- `scripts/`：本次验证脚本快照，运行前设置验证目录；不写模型、不运行部署工具。

验收：真实ONNX解析、Tensor资源分析、五种候选检查、重要节点查询、原有功能回归、完整报告和原始SHA256不变全部完成。未知资源、完整BPU覆盖、运行时数值/硬件/内存/性能明确无法在本次静态验收验证。`reports/` 按现有 .gitignore 被忽略，文件已保存本地；未执行git add/commit/push。
