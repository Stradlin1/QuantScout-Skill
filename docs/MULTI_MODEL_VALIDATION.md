# 多模型真实 ONNX 回归验收总结（V1.3）

验收日期：2026-10-09。最终范围仅为 `/home/xhm/lianghua_ws/testonnx/` 实际存在的 9 个 `.onnx`；未下载或假设额外模型，用户收敛范围后不再检查其他模型目录。本总结按用户最新要求保存在 docs；模型原始分析、查询日志及逐模型验收均保存在 Git 忽略的 reports。没有 commit、push 或上传操作。

## 结论

**9/9 ONNX checker 通过，9/9 完整 analyze 成功，9/9 原始 SHA256 保持不变。确认并修复 1 个未知 Shape 原因分类 Bug。** 修复前原有 321 项测试通过；6 项新增回归在修复前失败，修复后全套 **327 passed in 12.01s**。未发现错误的具体 Shape 补全、遗漏节点、异常终止或新的 BPU 判断回归。分析成功不等于模型可全部运行于 BPU。

## 环境与复现

基线 HEAD：`38edb76c9de834696d22b5a7ef36893f52569d53`，初始工作区干净。Python 3.10.12；onnx 1.23.2；numpy 2.2.6；networkx 3.4.2；pydantic 2.14.0；PyYAML 6.0.3；pytest 8.4.2。使用项目 `.venv`。系统 ROS 的 PYTHONPATH 会自动加载 pytest 插件并因缺少 lark 阻止启动；清除该变量后项目测试正常，不需要更改项目依赖。

在仓库根目录执行（每个模型独立输出目录）：

```bash
env -u PYTHONPATH .venv/bin/python -m pytest -q
env -u PYTHONPATH .venv/bin/python -m rdkx5_doctor --help
env -u PYTHONPATH .venv/bin/python -m rdkx5_doctor rules validate
env -u PYTHONPATH .venv/bin/python -m rdkx5_doctor analyze --model /home/xhm/lianghua_ws/testonnx/<模型>.onnx --out reports/<新的独立目录>
```

规则 Registry 0.3.0，schema 1.3，49 条规则、10 类算子；YAML 和规则限值未修改。所有实际命令、退出码和耗时在本地 `commands.jsonl`，初次结果在 initial，修复后结果在 final。验收脚本在报告根目录供审查。

本地结果目录：`reports/testonnx-multi-model-20261009/`。下文模型编号沿用本次发现编号；最终名单只有 M18～M26，共 9 个。

## 模型结果、Shape 与 BPU 汇总

节点状态列顺序：NO_VIOLATION_FOUND / VIOLATION / NEEDS_VERIFICATION / NOT_COVERED。UNKNOWN 是规则级状态，与节点 NEEDS_VERIFICATION 不同；非阻塞配置 UNKNOWN 可以与节点 NO_VIOLATION_FOUND 并存。

| 模型 | Opset | 节点 | 原始 I/O 数 | 已知载荷 Tensor 前→后 | 未知 Tensor | 节点状态（四类） | 规则 PASS / FAIL / UNKNOWN / N/A |
|---|---:|---:|---:|---:|---:|---|---|
| M18 fcn_resnet50.onnx | 12 | 157 | 1/2 | 144→144 | 128 | 49 / 0 / 32 / 76 | 713 / 0 / 68 / 418 |
| M19 mobilenetv2.onnx | 12 | 105 | 1/1 | 182→182 | 101 | 62 / 0 / 2 / 41 | 622 / 0 / 12 / 416 |
| M20 mobilenetv2_qdq.onnx | 12 | 307 | 1/1 | 557→557 | 272 | 62 / 0 / 3 / 242 | 624 / 0 / 19 / 416 |
| M21 resnet18.onnx | 8 | 69 | 103/1 | 102→102 | 70 | 28 / 0 / 1 / 40 | 260 / 0 / 9 / 160 |
| M22 semantic.onnx | 11 | 231 | 1/1 | 361→361 | 0 | 192 / 1 / 10 / 28 | 1143 / 1 / 25 / 484 |
| M23 super_resolution.onnx | 10 | 12 | 9/1 | 10→10 | 11 | 4 / 0 / 0 / 8 | 44 / 0 / 0 / 32 |
| M24 tinybert.onnx | 14 | 334 | 3/1 | 198→198 | 230 | 69 / 17 / 18 / 230 | 413 / 17 / 79 / 0 |
| M25 vit_tiny.onnx | 14 | 626 | 1/1 | 223→223 | 618 | 184 / 49 / 5 / 388 | 1122 / 49 / 164 / 8 |
| M26 yolo26n.onnx | 20 | 384 | 1/1 | 629→629 | 0 | 306 / 2 / 19 / 57 | 1858 / 2 / 40 / 816 |

覆盖类型总数：`{'NOT_COVERED': 1110, 'AUTO_CHECKED': 956, 'PARTIAL_OR_CONDITIONAL': 159}`；节点状态总数：`{'NOT_COVERED': 1110, 'NO_VIOLATION_FOUND': 956, 'NEEDS_VERIFICATION': 90, 'VIOLATION': 69}`；规则状态总数：`{'PASS': 6799, 'NOT_APPLICABLE': 2750, 'UNKNOWN': 416, 'FAIL': 69}`。逐算子数量/占比与完整覆盖矩阵见每个 MODEL_VALIDATION.md。

所有模型的新增 proved axis、新增已知载荷 Tensor、冲突均为 0，预算未超限。semantic 与 yolo26n 的 Tensor 元信息已经全部可计算载荷；其余模型保留真实符号维度/未知 Shape，未强行填 batch=1。后者意味着本次真实模型并未触发新的具体轴推导；非空证明验证仍依靠项目已有 Slice 等微型测试，不能宣称真实多模型覆盖了全部证明分支。

## 发现的问题和修复证据

### 确认 Bug：动态 Shape 参数被错误标记为语义冲突

- MobileNetV2 `main/node_000103`（`Reshape_103`），输入 `464`、`471`，输出 `472`；QDQ 版 `main/node_000299`（`Reshape_103_quant`），输入 `464_Reshape_103_dequantized`、`471`，输出 `472_QuantizeInput`。
- 两个模型的 `471` 都是 int64 长度 2 参数，tiny fact 为 `[None, -1]`、PARTIAL、`SYMBOLIC_UPSTREAM_DIM`。目标依赖符号 batch，不能静态确定；checker 通过且 conflict_count=0，不存在证明出的语义冲突。
- 根因：transfer 的参数读取将未知原因放入普通 ValueError；末尾异常处理只保留两种原因，把其他未知原因统一映射为 `ONNX_SEMANTIC_CONFLICT`。
- 最小修复：[static_shape_propagation.py](../src/rdkx5_doctor/static_shape_propagation.py) 用专用参数不可用异常保留原始原因，其余真实语义错误处理不变。没有放宽检查、改写模型或改动 YAML。
- 回归：[test_static_shape_propagation.py](../tests/test_static_shape_propagation.py) 新增动态 Shape→Gather→Unsqueeze→Concat→Reshape 合法微型模型，以及 unsupported/external/budget/override/depth 五种参数不可用原因，共 6 项。修复前 6 failed/15 passed；修复后局部 21 passed、完整 327 passed。证据为 reports 根目录 bug_reproduction.txt、final_pytest.stdout。
- 9 模型全部重跑，并比较 initial/final：所有 BPU diagnostics、nodes、edges、tensors、resource_analysis 完全相同；仅未知 Shape 原因分类/溯源修正。没有给动态 Reshape 产生新的具体轴证明。

### 静态违规核对与版本边界

全部 69 个 FAIL 对应 69 个违规节点，均为 Add/Mul 最小输入 rank=0，已用独立 ONNX 元信息核对 actual=0；不是新 Shape 补全产生的违规。

| 模型 | 规则 | FAIL 数 | 代表节点 / 原名 |
|---|---|---:|---|
| M22 semantic.onnx | `X5-MUL-INPUT-RANK-MIN` | 1 | `main/node_000127` / `/model.10/m/m.0/attn/Mul` |
| M24 tinybert.onnx | `X5-ADD-INPUT-RANK-MIN` | 13 | `main/node_000026` / `/bert/embeddings/LayerNorm/Add` |
| M24 tinybert.onnx | `X5-MUL-INPUT-RANK-MIN` | 4 | `main/node_000094` / `/bert/encoder/layer.0/intermediate/intermediate_act_fn/Mul_1` |
| M25 vit_tiny.onnx | `X5-ADD-INPUT-RANK-MIN` | 37 | `main/node_000019` / `/vit/encoder/layer.0/layernorm_before/Add` |
| M25 vit_tiny.onnx | `X5-MUL-INPUT-RANK-MIN` | 12 | `main/node_000061` / `/vit/encoder/layer.0/intermediate/intermediate_act_fn/Mul_1` |
| M26 yolo26n.onnx | `X5-MUL-INPUT-RANK-MIN` | 2 | `main/node_000125` / `/model.10/m/m.0/attn/Mul` |

每一个违规节点的完整原名、输入 Tensor/Shape、actual/expected 和规则来源都保存在各模型 MODEL_VALIDATION.md 与根目录 ALL_FAILURE_EVIDENCE.json，没有合并后丢弃节点证据。标量 ONNX 运算合法性与收录的 BPU rank 条款是不同问题；常量折叠、编译器支持和实际执行位置仍需工具链验证，本次未执行。
规则来源为项目已审查的 [X5 官方算子支持表](https://developer.d-robotics.cc/x5_sdk_doc/toolchain_development/intermediate/supported_op_list.html)，具体版本、列、条目在每条 results.source 中。本次执行现有规则，不重新编造或扩大已审核版本。

- TinyBERT / ViT Opset 14：33+96 个 MatMul、4+12 个 Softmax 均 NOT_COVERED；YOLO26n Opset 20：4 MatMul、2 Softmax、2 Resize 均 NOT_COVERED。新 attention 包仅审查指定版本，不能套用 Opset 11/13 结论。Add/Mul 原包的版本范围不同，其标量 FAIL 仍有效。
- semantic Opset 11：MatMul `main/node_000129`、`000132`，Softmax `000130`，Resize `000152`、`000188`、`000228` 的静态条件未发现违规。Softmax 的 `X5-SOFTMAX-RUN-ON-BPU-VERIFIED` 仍 UNKNOWN（非阻塞），不能据节点状态推断已分配 BPU。
- FCN Opset 12：Resize `main/node_000140`、`000156` 保留空间尺寸、坐标版本核对、ROI 等 UNKNOWN；MobileNetV2 QDQ MatMul `main/node_000303` 的 rank、矩阵尺寸、高维尺寸与 broadcast 保留 UNKNOWN。实际节点均经过 inspect/trace；没有把缺失信息转换为 FAIL 或 PASS。
- 本次没有 Conv 静态 VIOLATION；UNKNOWN/条件项和 NOT_COVERED 完整列在各模型报告，不能从未发现 Conv 违规推断部署可行。

## Shape、原始图与理论资源审计

独立审计重新直接调用 ONNX strict shape inference，将其元信息与报告的 onnx_inferred_shape 比较，确认 overlay 未无证据更改维度；检查每个已有具体轴、public I/O dtype/Shape、证明引用/生产者、节点身份、覆盖/状态计数守恒。独立用 ONNX dtype→NumPy itemsize 和 Shape 乘积检查所有已知字节及 INT8 情景，复核消费者计数与全部 FAIL 的标量 rank。
一致性检查共 20976 项、独立审计共 16434 项，两者失败为 0。全部 checker 与 strict shape inference 均 PASS，全部原始图无外部 initializer、本地 function 或嵌套子图；QDQ 有额外 domain import，但实际节点均在标准域，不能据 import 数量判定自定义算子。

| 模型 | 原始公开数据输入 → 输出 | 未知资源的主要限制 | 最大已知中间 Tensor（原始载荷） |
|---|---|---|---|
| M18 | `input ['batch', 3, 'height', 'width'] float32` → `out ['batch', 21, 'height', 'width'] float32; aux ['batch', 21, 'height', 'width'] float32` | 符号 batch/空间/序列维度或未覆盖 Shape transfer；128 Tensor | `335 [4] int64 32 B` |
| M19 | `input ['batch_size', 3, 224, 224] float32` → `output ['batch_size', 1000] float32` | 符号 batch/空间/序列维度或未覆盖 Shape transfer；101 Tensor | `465 [4] int64 32 B` |
| M20 | `input ['batch_size', 3, 224, 224] float32` → `output ['batch_size', 1000] float32` | 符号 batch/空间/序列维度或未覆盖 Shape transfer；272 Tensor | `classifier.1.weight_dequantized [1280, 1000] float32 5120000 B` |
| M21 | `data ['N', 3, 224, 224] float32` → `resnetv15_dense0_fwd ['N', 1000] float32` | 符号 batch/空间/序列维度或未覆盖 Shape transfer；70 Tensor | `无载荷完全已知的中间 Tensor` |
| M22 | `images [1, 3, 320, 320] float32` → `output0 [1, 320, 320] uint8` | 完整已知 | `/model.17/Resize_output_0 [1, 7, 320, 320] float32 2867200 B` |
| M23 | `input ['batch_size', 1, 224, 224] float32` → `output ['batch_size', 1, 672, 672] float32` | 符号 batch/空间/序列维度或未覆盖 Shape transfer；11 Tensor | `无载荷完全已知的中间 Tensor` |
| M24 | `input_ids ['batch_size', 'sequence_length'] int64; attention_mask ['batch_size', 'sequence_length'] int64; token_type_ids ['batch_size', 'sequence_length'] int64` → `logits ['batch_size', 'sequence_length', 9] float32` | 符号 batch/空间/序列维度或未覆盖 Shape transfer；230 Tensor | `/bert/Reshape_output_0 [4] int64 32 B` |
| M25 | `pixel_values ['batch_size', 'num_channels', 'height', 'width'] float32` → `logits ['batch_size', 1000] float32` | 符号 batch/空间/序列维度或未覆盖 Shape transfer；618 Tensor | `/vit/embeddings/Shape_output_0 [4] int64 32 B` |
| M26 | `images [1, 3, 640, 640] float32` → `output0 [1, 300, 6] float32` | 完整已知 | `/model.0/act/Mul_output_0 [1, 16, 320, 320] float32 6553600 B` |

- ResNet18 原始 graph inputs=103，其中 102 与 initializer 重叠，可由调用者覆盖；super_resolution inputs=9，其中 8 为可覆盖 initializer。表中只简列数据输入，完整原始接口在逐模型报告。Conv-BN 参数虽有 initializer，也不能忽略覆盖接口限制。
- 动态模型 FCN、MobileNetV2、TinyBERT、ViT 的已知 Top 10 往往是 int64 Shape 元数据，不能据 32 B 最大已知项称特征图很小；ResNet18、super_resolution 没有载荷完全已知的中间 Tensor。
- QDQ 最大已知中间 `classifier.1.weight_dequantized` 为 FP32 `[1280,1000]`，5,120,000 B，是反量化权重。semantic 最大中间 `/model.17/Resize_output_0` 为 FP32 `[1,7,320,320]`，2,867,200 B；YOLO26n `/model.0/act/Mul_output_0` 为 FP32 `[1,16,320,320]`，6,553,600 B。生产者/消费者和代表路径均由 tensor/inspect/trace 保存。
- 每个模型的全部输出资源、已知中间 Top 10、未知原因计数、多消费者数量在 final/Mxx/MODEL_VALIDATION.md。FP32 原始载荷与假设 INT8 明确分列；实际 BPU/DDR/峰值内存、布局、别名、复用、融合及精度/性能未测量。

## 自主选择的深入核对与候选

先从 MobileNetV2 的“冲突计数 0 / 语义冲突原因非零”矛盾选择 Reshape 参数及其 Shape 依赖路径，确认并最小修复原因分类。随后针对全部 FAIL 逐条独立复算 rank，选择各算子代表违规节点、待验证节点、attention/Resize 和最大已知中间生产者执行 inspect/trace；另查询每个模型的未知或典型 Tensor shape。
候选核对发现 FCN 有 2 个 CAST_SAME_DTYPE（局部语义冗余，首个 `main/node_000137` / Cast_137，int64→int64）；ResNet18 有 20 个 CONV_BN_FUSION_REVIEW（仅 REVIEW_REQUIRED）；MobileNetV2/QDQ 各 1、super_resolution 2、TinyBERT 16、ViT 49 个 RESHAPE_NOOP 均 INSUFFICIENT_INFORMATION。semantic 与 YOLO26n 无候选；没有制造 Identity/逆 Transpose 候选。首个候选额外使用 candidate、inspect、trace 核对，接口/分支/参数限制保留在候选证据中。
最大 Tensor 与违规节点的有向路径关系逐模型记录；有路径只能证明拓扑关联，不意味着资源问题或违规向下游传播。所有结构优化方向需回到训练/导出工程实施，本次未改 ONNX、未搜索训练工程、未生成训练补丁。

## SHA256 和结果索引

以下全部文件在最终分析及独立复核后重新读取 SHA256，与初始值完全一致。

| 模型 | SHA256 | 本地结果 |
|---|---|---|
| fcn_resnet50.onnx | `eb5017d1b80372eb0b58552655274698817ba2e774437e2c2d3c0a613d2e99bd` | [逐模型验收](../reports/testonnx-multi-model-20261009/final/M18/MODEL_VALIDATION.md) · [analysis.json](../reports/testonnx-multi-model-20261009/final/M18/analysis.json) · [report.md](../reports/testonnx-multi-model-20261009/final/M18/report.md) |
| mobilenetv2.onnx | `c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5` | [逐模型验收](../reports/testonnx-multi-model-20261009/final/M19/MODEL_VALIDATION.md) · [analysis.json](../reports/testonnx-multi-model-20261009/final/M19/analysis.json) · [report.md](../reports/testonnx-multi-model-20261009/final/M19/report.md) |
| mobilenetv2_qdq.onnx | `41a36090dafe98f4ad8f9b7fe0b218c56ac3c031e547f0367c30655d2702bffe` | [逐模型验收](../reports/testonnx-multi-model-20261009/final/M20/MODEL_VALIDATION.md) · [analysis.json](../reports/testonnx-multi-model-20261009/final/M20/analysis.json) · [report.md](../reports/testonnx-multi-model-20261009/final/M20/report.md) |
| resnet18.onnx | `4e8f8653e7a2222b3904cc3fe8e304cd8b339ce1d05fd24688162f86fb6df52c` | [逐模型验收](../reports/testonnx-multi-model-20261009/final/M21/MODEL_VALIDATION.md) · [analysis.json](../reports/testonnx-multi-model-20261009/final/M21/analysis.json) · [report.md](../reports/testonnx-multi-model-20261009/final/M21/report.md) |
| semantic.onnx | `680ffd95be30dad4a697a89d0366fee85ae673d96d107990dfbeb4699168cd95` | [逐模型验收](../reports/testonnx-multi-model-20261009/final/M22/MODEL_VALIDATION.md) · [analysis.json](../reports/testonnx-multi-model-20261009/final/M22/analysis.json) · [report.md](../reports/testonnx-multi-model-20261009/final/M22/report.md) |
| super_resolution.onnx | `85f36ff88cc504a24af5e0602148bc56a8aa09a58eca8c0da2756f3e8186035e` | [逐模型验收](../reports/testonnx-multi-model-20261009/final/M23/MODEL_VALIDATION.md) · [analysis.json](../reports/testonnx-multi-model-20261009/final/M23/analysis.json) · [report.md](../reports/testonnx-multi-model-20261009/final/M23/report.md) |
| tinybert.onnx | `33477f79dd9a106971dc8c5201daae6be310e02d6a63ca559a990dc8b8e97ee8` | [逐模型验收](../reports/testonnx-multi-model-20261009/final/M24/MODEL_VALIDATION.md) · [analysis.json](../reports/testonnx-multi-model-20261009/final/M24/analysis.json) · [report.md](../reports/testonnx-multi-model-20261009/final/M24/report.md) |
| vit_tiny.onnx | `32f1cf3b4868cc44757814cbf6237ae5a3658a45980bf0fe93914dca77b815e3` | [逐模型验收](../reports/testonnx-multi-model-20261009/final/M25/MODEL_VALIDATION.md) · [analysis.json](../reports/testonnx-multi-model-20261009/final/M25/analysis.json) · [report.md](../reports/testonnx-multi-model-20261009/final/M25/report.md) |
| yolo26n.onnx | `00e2d1062178fca312fee35cf5fc9b9c783b3e5c9675d48b84929f69ee391807` | [逐模型验收](../reports/testonnx-multi-model-20261009/final/M26/MODEL_VALIDATION.md) · [analysis.json](../reports/testonnx-multi-model-20261009/final/M26/analysis.json) · [report.md](../reports/testonnx-multi-model-20261009/final/M26/report.md) |

GitHub 不含 reports，因此上述本地结果链接在线不会展开；本 docs 总结包含验收主要数字、问题根因和代表节点证据，供线上审查。没有把详细检测文件移入 docs。

## 下一步与无法验证条件

1. 按官方来源补充审核 attention Opset 14/20 与 Softmax 坐标/axis 语义，当前 NOT_COVERED 保留。
2. 优先扩展 Q/DQ、Clip、GlobalAveragePool、BatchNormalization、ReduceMean、Pow、Sqrt、Div、Shape 参数路径的元信息传递；有界且保留符号维度，不据动态 batch 猜载荷。BPU 规则扩展另需官方条款证据。
3. 在 Top 10 查询显著区分特征激活、Shape 参数和反量化权重，并提醒已知子集排名，不改变资源公式。
4. 后续采集经授权的固定输入/序列尺寸另行导出模型，增加会触发非空逐轴证明的真实样本；本次不能从 0 新证明推断覆盖所有证明分支。
5. run_on_bpu 配置、工具链实际接受情况、常量折叠/融合、量化、数值精度、任务指标和板端内存/性能仍未验证；本次完全未调用 Docker、hb_mapper 或量化工具链。

## 最终工作区

本次修改仅为 static_shape_propagation.py、对应回归测试，以及新增 docs/MULTI_MODEL_VALIDATION.md；原有用户修改无覆盖，rulesets、模型、Git 历史不变。reports/ 仍被忽略。CLI help、rules validate、完整 pytest 和 git diff --check 均通过。没有 commit/push。
