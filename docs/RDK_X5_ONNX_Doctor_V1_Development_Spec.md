# RDK X5 ONNX Doctor — V1 开发实施文档（Codex 执行规范）

> **文档版本**：V1.0 ｜ **目标平台**：Ubuntu + VS Code + Codex ｜ **目标设备**：RDK X5（Bayes-e）<br>
> **开发阶段**：第一版（MVP）｜ **首个支持的硬件约束算子**：ONNX `Conv`（以 Conv2D 为主）<br>
> **执行指令**：Codex 应完整阅读本文档，按阶段在当前仓库内实现、运行测试、修复问题并提交可验证的成果。不要只输出设计建议或伪代码。<br>
> **注意**：本轮只开发本地静态分析工具与 Codex Skill 入口；**不接入 Docker、不执行 hb_mapper、不执行量化、不改写原始 ONNX、不实现其他算子的 BPU 规则**。

---

## 0. 项目目标和不可变更的约束

### 0.1 我们究竟要做什么

输入一个 ONNX 模型，自动完成：

1. 通过 ONNX Python API 解析计算图；允许使用 Netron 打开原始模型查看结构。
2. 枚举算子和节点，提取 Conv2D 参数、输入输出张量形状、数据类型和依赖连接。
3. Codex **根据指定版本的 RDK X5 官方算子约束文档总结**适用的 Conv 约束，形成有官方来源的、可复用的本地规则文件。
4. 用确定性 Python 检查程序逐个验证 Conv 节点的可判定约束；不能可靠判定的条件输出“待验证”，不得猜测。
5. 对异常 Conv 节点进行上下游计算图追踪，识别其能够影响的模型输出与相关节点路径。
6. 输出 `analysis.json`、`report.md`、`graph.html`；`graph.html` 具有类似 Netron 的交互式计算图，异常节点可点击查看证据、官方依据和优化建议。
7. 提供本地 Codex Skill 入口，使用户输入“分析这个 ONNX 是否适合 RDK X5”，Codex 就调用工具并解释结果。

### 0.2 V1 明确不做

- **不做节点风险打分、不建立风险评分模型。** 只展示事实状态与原因。
- 不为 Add、Mul、Concat、Reshape 等其他算子开发 BPU 检查规则；它们仍必须正确显示在图中。
- 不根据 Conv 个数、参数量或 FLOPs 直接推断量化精度下降。
- 不自动删除、替换、融合模型中的计算节点；只允许生成优化建议。
- 不直接声称某节点一定会分配到 BPU 或 CPU；静态检查**不等于**真实工具链编译结果。
- 不运行 Docker、`hb_mapper checker`、PTQ、`hb_perf` 或板端推理；这些由另一个分支后续接入。
- 不开发 VS Code 专用扩展、不修改 Netron 源码、不以 Netron 内部 JavaScript 私有接口作为解析 API。
- 不在运行时要求大模型每次重新阅读全部官方文档；首版整理的、已标注来源的规则应可缓存复用。

### 0.3 成功标准（一句话）

> 给定真实 YOLO ONNX 或测试 ONNX，能正确定位违反**已收录官方 Conv 约束**的具体节点，解释违反了哪一条规则，沿 Tensor 连接追踪到相关模型输出，并在交互式计算图中定位展示。

---

## 1. 核心设计原则

### 1.1 Netron 与 Python 分工

| 能力 | 采用方式 | 原因 |
|---|---|---|
| 查看原始 ONNX 结构 | Netron（可选启动） | 成熟的模型图浏览器，避免重复造轮子 |
| 批量读取节点/属性/形状 | Python `onnx` 包 | 可测试、结构化、稳定 |
| 计算节点拓扑与影响路径 | 自建轻量 Graph IR + NetworkX | 需要精确的生产者/消费者关系 |
| 转换官方文字约束 | Codex 根据官方资料整理成本地 YAML | 保留 AI 的专业文档理解作用 |
| 确定性约束判断 | Python 规则检查器 | 避免 LLM 比较数字和处理边界条件出错 |
| 优化建议和文字解释 | Codex/Skill | 使用节点证据及来源，提出非自动执行的方案 |
| 诊断结果可视化 | HTML + Cytoscape.js（或同类成熟库） | 可点击、搜索、突出显示路径 |

**Netron 不作为程序解析的唯一数据来源。** 用户可以通过 `netron model.onnx` 或 `netron.start(...)` 打开原始模型；诊断图由我们从 ONNX 解析结果和异常标注生成，以免依赖 Netron 未承诺稳定的外部染色、交互注入 API。

### 1.2 AI 不直接决定约束数值的真假

正确分工：

`官方文档 → Codex 抽取/解释适用规则 → 带来源的规则 YAML → Python 读取真实节点参数 → PASS/FAIL/UNKNOWN → Codex 给出解释与建议`

不要使用：

`把全部节点文本发给 LLM → LLM 即兴判断全部参数 → 直接输出“BPU 兼容”`

### 1.3 结果用状态，不用分数

**单条规则**仅返回：`PASS`、`FAIL`、`UNKNOWN`、`NOT_APPLICABLE`。

**节点汇总**显示：`VIOLATION`（存在明确已检规则违规）、`NO_VIOLATION_FOUND`（已执行规则未发现违规）、`NEEDS_VERIFICATION`（有关键条件无法判定或要求工具链验证）、`NOT_COVERED`（V1 未提供该算子的规则）。

`NO_VIOLATION_FOUND` 只能理解为“本地已执行的规则未发现问题”，**不是**“官方确认 BPU 可以执行”。非 Conv 算子不按“通过”统计。

---

## 2. 技术栈、运行方式和目录

### 2.1 环境

- Ubuntu；使用项目内 `.venv`，Python >= 3.10。
- 核心 Python 库：`onnx`、`networkx`、`pyyaml`、`pydantic`、`pytest`。
- Netron：推荐可选安装 `pip install netron`；工具缺失时不阻塞 ONNX 分析。
- HTML 图：推荐 Cytoscape.js + dagre 布局插件，**将前端 JS 依赖固定版本并打包成离线可用的本地资源**；若实现上明显更简洁，可选成熟的替代图形库，但必须达到本文件交互要求。
- V1 不依赖 GPU、RDK X5 开发板、Docker、量化工具链、模型训练框架。

### 2.2 推荐仓库目录

```text
rdk-x5-onnx-doctor/
├── README.md
├── AGENTS.md
├── pyproject.toml
├── .gitignore
├── src/rdkx5_doctor/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── onnx_reader.py          # 合法性检查、形状推断、解析
│   ├── graph_ir.py             # 节点、Tensor 和连接关系
│   ├── conv_extractor.py       # Conv2D 派生参数
│   ├── rules.py                # YAML 加载、校验、判定
│   ├── trace.py                # 节点上游/下游追踪
│   ├── report.py               # JSON / Markdown
│   └── visualize.py            # graph.html + 静态资源
├── rulesets/
│   └── x5-bayes-e/
│       ├── manifest.yaml       # 平台、官方文档版本、审查状态
│       └── conv2d.yaml         # Codex 根据官方文档整理的 Conv2D 规则
├── references/
│   ├── official_sources.md    # 精确官方 URL、标题、版本、检索日期
│   └── conv_rule_notes.md     # 文档歧义、未自动化检查的条款
├── web/
│   ├── template.html
│   ├── app.js
│   ├── styles.css
│   └── vendor/                # 固定版本的 JS/CSS 静态资源
├── tests/
│   ├── conftest.py
│   ├── fixtures/               # pytest 生成微型 ONNX；不提交巨型权重
│   ├── test_onnx_reader.py
│   ├── test_conv_extractor.py
│   ├── test_rules.py
│   ├── test_trace.py
│   └── test_report_visual.py
├── .agents/skills/rdk-x5-onnx-doctor/
│   └── SKILL.md
├── examples/                  # 可选：小型示例模型与调用示例
└── reports/                   # 生成产物，建议 gitignore
```

不需要为 1 个算子建立十几层抽象。优先可运行性、可测试性和数据流清晰。规则加载器应能后续扩展其他算子，但 V1 只写 Conv2D 规则。

### 2.3 期望命令

完成项目后，至少支持：

```bash
python -m rdkx5_doctor analyze \
  --model ./models/example.onnx \
  --ruleset ./rulesets/x5-bayes-e \
  --out ./reports/example
```

输出：

```text
reports/example/
├── analysis.json
├── report.md
├── graph.html
└── assets/             # 如果图形依赖未内嵌，则放离线静态资源
```

另外提供：

```bash
python -m rdkx5_doctor rules validate \
  --ruleset ./rulesets/x5-bayes-e

python -m rdkx5_doctor open-netron \
  --model ./models/example.onnx
```

`open-netron` 是可选命令，Netron 未安装或无法启动时给出安装说明并优雅退出；默认 `analyze` 绝不主动打开浏览器。CLI 应有 `--help`、合理非零退出码和清晰的错误消息。

---

## 3. 模块 A — ONNX Reader 与计算图 IR

### 3.1 加载、校验、形状推断

实施步骤：

1. 校验传入路径存在、文件可读、扩展名为 `.onnx`；对非法输入给出明确异常。
2. `onnx.load()` 读取模型；使用 `onnx.checker.check_model()` 做 ONNX 结构合法性检查。
3. 对可用模型调用 `onnx.shape_inference.infer_shapes()`，**只修改内存副本，不覆盖源文件**。
4. 记录模型的 opset imports（包括 domain）、graph inputs/outputs、initializer 元信息、所有 node 和 value_info。
5. 形状推断失败时区分“模型本身无效”和“形状推断不完整”；尽可能保留可解析结构并在报告中说明。
6. 不要把权重全部转成 NumPy 数组以取 Shape；通常用 TensorProto 的 `dims` 即可。保持内存占用可控。
7. 若模型使用 external data，应优先安全读取元信息；缺少权重文件时明确报告，并允许继续进行已知部分的解析。不得静默假设缺失维度。

### 3.2 稳定节点 ID

ONNX 的 `node.name` 可能为空、重复或随导出改变。统一以**图路径 + 节点原始顺序下标**生成稳定内部 ID，例如：

`main/node_000128`

保留原始 `name` 作为显示名；**所有规则结果、路径追踪和图形标注都使用内部 ID 关联**。

V1 首先完整支持模型主图；遇到 If/Loop 等嵌套子图，明确标为“子图解析未覆盖”，不能默默忽略后声称全图通过。

### 3.3 Tensor / Node 元数据要求

Node 至少保存：

```json
{
  "id": "main/node_000128",
  "original_name": "/model.3/conv/Conv",
  "op_type": "Conv",
  "domain": "",
  "inputs": ["tensor_input", "weight_128", "bias_128"],
  "outputs": ["tensor_output"],
  "attributes": {"group": 1, "strides": [2, 2], "pads": [1, 1, 1, 1]}
}
```

Tensor 至少保存：

```json
{
  "name": "tensor_output",
  "shape": [1, 64, 160, 160],
  "dtype": "float32",
  "producer": "main/node_000128",
  "consumers": ["main/node_000129", "main/node_000132"],
  "kind": "intermediate"
}
```

若某个维度是 `dim_param`（如 `batch`）或未知，保留符号或 `null`，不可填充一个“猜测值”。记录输入和输出是否为 graph 边界、initializer、Constant 生成值、中间特征图。允许空的可选输入位，但不能把空字符串当成真实依赖 Tensor。

### 3.4 根据 Tensor 构造边

维护：

- `tensor_producer[tensor_name] -> node_id | graph_input | initializer | unknown`
- `tensor_consumers[tensor_name] -> [node_id, ...]`
- `graph_edge = (source_node_id, target_node_id, tensor_name)`

如果一个 Tensor 同时供两个算子使用，保留两条消费关系。不可通过节点名称中的数字猜测连边；不可仅按 ONNX 节点排列顺序连接。

使用 NetworkX `DiGraph` 即可做节点可达性；如果多条 Tensor 连接同两个节点，需要额外保存 Tensor 标签，或使用 `MultiDiGraph`，确保报告能说明是哪条 Tensor 造成依赖。

**验收点**：对含分支的 Conv → Add 计算图，能够正确找到其前驱、后继、共享 Tensor 与最终输出。

---

## 4. 模块 B — Conv2D 参数提取

### 4.1 解析范围

V1 仅将**标准 ONNX 域中的 Conv、输入 rank=4、权重 rank=4**视为 Conv2D 约束检查对象。非标准域、Conv1D/Conv3D、缺少关键 Shape 的 Conv，保留节点并输出 `UNKNOWN` 或 `NOT_APPLICABLE`，不得直接判定超出 BPU 能力。

ONNX Conv 输入约定：

- `X`: `[N, C_in, H, W]`
- `W`: `[C_out, C_in_per_group, K_h, K_w]`
- `B`: 可选偏置
- `group`: 默认 1，且 `C_in = C_in_per_group × group`
- `C_out` 应与 `group` 的关系符合 ONNX Conv 规范

提取的派生字段：

```text
input_rank
input_n, input_c, input_h, input_w
output_rank, output_n, output_c, output_h, output_w
weight_shape
out_channels
in_channels_per_group
kernel_h, kernel_w
kernel_elements_per_group = in_channels_per_group * kernel_h * kernel_w
strides_h, strides_w
dilation_h, dilation_w
pads_top, pads_left, pads_bottom, pads_right
auto_pad
group
input_dtype, output_dtype
```

### 4.2 默认值与细节

- `kernel_shape` 缺省时从 `W` 的最后两个维度推断；如果属性与可得的权重 Shape 冲突，应报告模型信息不一致。
- `strides` 默认 `[1, 1]`；`dilations` 默认 `[1, 1]`；`group` 默认 `1`；`pads` 缺省默认 0；`auto_pad` 缺省为 `NOTSET`。
- ONNX `pads` 顺序是 `[H_begin, W_begin, H_end, W_end]`；不要按 `[left,right,top,bottom]` 解析。
- `auto_pad=SAME_UPPER/SAME_LOWER/VALID` 需按 ONNX 语义处理；没有足够信息确认有效 padding 时记录 `UNKNOWN`，不得直接假设为零。
- `group>1`、depthwise Conv 必须正确处理 **per-group channel**，绝对不要把总输入通道数当成权重的 `C_in_per_group`。
- 输入维度符号化、initializer 缺失、动态生成的权重 Shape 不可获知时，将对应字段设为 unknown 并输出原因。
- **浮点 ONNX 的 `output_dtype=float32` 并不代表量化后的 BPU 实际输出精度也是 float32；涉及量化后 INT8/INT16 的条件在本阶段不得擅自推断。**

---

## 5. 模块 C — 官方文档 → Conv 规则（由 Codex 先整理一次）

### 5.1 官方来源与版本控制

Codex 开发规则前，先核对指定官方资料：

- X5 RDK 算子约束（中文，见文末链接）；**明确跳转到 RDK X5 的 ONNX Conv 行，而不是 X3 或 Caffe Conv 行**。
- X5 芯片手册对应版本的算子约束资料（作为交叉核对）。
- ONNX Conv 官方规范，用于解析属性而**不是**代替 BPU 约束。

**必须做版本标记**：规则适用的资料标题、URL、章节、文档版本（可确认时）、检索日期、待验证的工具链版本。若工具链版本尚未知（本项目本轮就是如此），写 `toolchain_version: unverified`。

由于官方页面和工具链迭代，**不要把曾经看到的数值当成永久真理**。同一约束若存在版本冲突，保留两份来源并标记 `needs_review`；不得偷偷选择一个“看着更合理”的结论。

### 5.2 V1 规则生产流程

1. Codex 阅读官方文档中的 **X5 / ONNX / Conv2D** 小节。
2. 将每一条约束拆成：适用范围、依赖字段、判断表达式、条件分支、失败说明、官方出处。
3. 明确区分以下三类：
   - **auto_check**：仅用已提取的静态值即可计算；V1 写成可执行 YAML。
   - **conditional**：需要知道编译器融合/量化子图位置/量化输出精度等信息；V1 保留提示，结果 `UNKNOWN`。
   - **review_only**：自然语言歧义、版本冲突或解析尚未实现；V1 明确列入 `conv_rule_notes.md`，不伪装成已检查。
4. Codex 生成 `conv2d.yaml` 并通过 schema 校验；规则要附 `source_url` 和 `source_section`。
5. 若官方页面不可读取，先生成规则模板和来源待办；**不得凭模型记忆填充未经核实的 BPU 数值**。
6. Codex 使用单元测试验证边界，不允许只把说明文档放进 references 后就结束。

### 5.3 建议的规则 YAML 模式（示例）

下面只是**格式示例**；具体阈值由 Codex 依据所选官方文档确认后写入版本化规则文件：

```yaml
ruleset_id: x5-bayes-e-onnx-conv2d
ruleset_version: "0.1.0"
model_domain: ""
operator: Conv
scope: input_rank_4
source_document: "RDK X5 ONNX operator constraints"
source_version: "to_be_verified"
source_url: "https://developer.d-robotics.cc/rdk_x_doc/Advanced_development/toolchain_development/intermediate/supported_op_list"
toolchain_version: "unverified"
rules:
  - id: X5-CONV2D-KERNEL-H
    check_type: range
    field: kernel_h
    min: 1
    max: 31
    checkability: auto_check
    source_section: "RDK X5 / ONNX / Conv"
    message: "Conv2D kernel height out of documented range"
  - id: X5-CONV2D-KERNEL-VOLUME
    check_type: max_value
    field: kernel_elements_per_group
    max: 32767
    checkability: auto_check
    source_section: "RDK X5 / ONNX / Conv"
    message: "Conv2D kernel C*H*W exceeds documented limit"
```

上例中的 `31` 和 `32767` 与已公开的 X5 Conv2D 限制一致，但**仍须以 Codex 最终选定和记录的官方文档版本复核**。不得将它们复制给 X3 或 Conv3D 规则。

### 5.4 规则引擎最小能力

- 校验 YAML 结构，不符合 schema 时以明确错误终止，不静默忽略。
- 支持基础 `range`、`max_value`、`equals`、`one_of` 等通用操作，不在代码里写死“Conv kernel 必须小于 31”。
- `field` 必须从白名单派生字段读取；**禁止对 YAML 中的字符串执行 `eval/exec`**。
- 字段缺失返回 `UNKNOWN`，并说明缺失什么数据。
- 条件规则如果无法判断触发条件，返回 `UNKNOWN`，而非默认为 PASS。
- 检查结果记录 `rule_id`、`field`、`actual`、`expected`、`source_url`、`status`、`reason`。
- 对目前没有规则的算子统一 `NOT_COVERED`；不能凭单个算子名称就判定 CPU/BPU 归属。
- 即使所有可运行规则通过，也必须在报告显著提示“尚未用实际工具链验证”。

### 5.5 本版 Conv2D 规则覆盖策略

**优先实现**：可静态获得的卷积核 H/W 范围、每组输入通道与卷积核乘积上限、明确的 Stride/Dilation/Padding 数值区间（前提是官方该版本无歧义）。

**有条件则实现**：分组通道数约束、与后继 Add 的特殊 Stride 约束、非对称 padding 等（必要时涉及图模式和 ONNX 语义，需要独立测试）。

**不能强行自动判定**：只对“量化子图最后一个 Conv”成立的放宽限制、实际量化后 INT8/INT16 输出条件、真实 CPU/BPU 划分、实际峰值 DDR、延迟及精度。此类列为 `UNKNOWN/工具链待验证`。

由于有官方资料版本中的 `Conv → Add` 特殊措辞存在差异，**V1 不要仅凭下一节点 op_type=Add 就套用 ResNet shortcut 规则**；需要有可靠的模式识别与确切条文，否则在 `conv_rule_notes.md` 记录，不阻断本轮交付。

---

## 6. 模块 D — 节点风险溯源（必须完成）

### 6.1 输入和输出

输入：一个或多个检查结果为 `VIOLATION` 的 **内部节点 ID**。<br>
输出：它们的直接前驱、直接后继、由 Tensor 依赖可到达的图输出、到输出的代表路径，以及原始违规证据。

例如：

```text
Input → Conv_001 → Conv_128 [违规] → Add_132 → Concat_138 → Output
                              └────────→ Mul_140 ─────→ Output_2
```

显示时区分：

- **红色：违规源节点**（有确定规则失败）。
- **橙色边/节点轮廓：由该节点出发的相关路径**（表示依赖关系，**不表示下游节点本身违规**）。
- **中性色：其他已解析节点**。
- **问号/虚线：当前信息不足或未覆盖**。

### 6.2 实现要求

- 必须依据 Tensor producer/consumer 的真实边构造路径。
- 支持存在分叉、汇合、残差连接和共享中间特征图的模型。
- `graph_output` 可以由任何 Tensor 提供；追踪到模型输出的 Tensor 名称，而非把最后一个 Conv 误认为输出。
- 如果违规节点与任何图输出均无连通路径，明确报告“未发现到模型输出的依赖路径”，可能存在死分支。
- 对含大量节点的 YOLO，避免枚举**所有**简单路径（可能指数爆炸）：只展示每个输出的有限条代表路径，并记录路径展示发生了截断；仍可返回所有可达输出的名称。
- 前端默认显示异常节点周围 `2` 层邻居 + 到输出的代表路径，允许切换查看完整计算图。
- 下游受到“依赖影响”的标注不能转换成“CPU 回退已确认”或“量化精度下降已确认”的结论。

### 6.3 建议追踪算法

1. 从违规 `node_id` 逆向 BFS 找到指定层数的上游节点。
2. 正向 BFS 求所有后继节点集合。
3. 根据模型输出 Tensor 的 producer 映射，识别可达输出。
4. 使用有界最短路径或有限条路径枚举生成可展示的节点序列。
5. 记录通过哪些 Tensor 边连接，便于用户点击查看数据依赖。

**不设计任何 0～100 分风险等级，不编造性能量化值。**

---

## 7. 模块 E — Netron 风格交互式计算图

### 7.1 设计目标

输出 `graph.html`，用户无需安装 VS Code 插件，在 Ubuntu 浏览器中即可打开：

- 支持缩放、拖拽、适配屏幕、节点搜索。
- 默认按 ONNX 拓扑（左→右或上→下）排布，布局清楚，体现分支和汇合。
- 节点至少显示“算子类型 + 简短名称”；选中节点后侧栏展示全部参数和诊断证据。
- 支持按 `VIOLATION / NEEDS_VERIFICATION / NOT_COVERED` 过滤，尤其能“一键定位违规节点”。
- 点击违规节点，突出显示上游邻居、下游代表路径与可达图输出。
- 展示官方规则来源链接、实际参数值、限制条件，以及非自动执行的建议。
- 如果无违规 Conv，仍能正常展示图，并清楚显示“未发现已检查规则违规（非工具链结论）”。
- 图的配色表示**状态**，不是分数。

### 7.2 推荐实现

- UI 图层：Cytoscape.js + dagre 布局（也可使用同等成熟的库）。
- 后端：`analysis.json` 导出**节点数组、连边数组和诊断标注**；生成 HTML 时将展示所需数据安全地内嵌到 `graph.html`（例如 HTML 转义后的 `application/json` 数据块），前端只负责渲染。**不要从 `file://` 页面直接 `fetch('analysis.json')`，否则可能被浏览器跨源限制阻止**；`analysis.json` 仍作为独立的机器可读结果保留。
- 图结构建议保留真实节点 ID 和 Tensor 名称，节点侧栏可通过 node ID 查找完整对象。
- 不依赖运行时 CDN；将固定版本静态资源复制到 `reports/<run>/assets/`，`graph.html` 在离线 Ubuntu 上仍应可打开。
- 需要考虑几百至几千节点时的性能：默认聚焦违规节点子图，提供展开完整图；不可在首次加载时无条件对所有节点进行高成本自动布局。
- 所有 ONNX 原始名称/属性以**文本**插入 UI，防止 HTML/JS 注入；不要把模型提供的字符串直接拼接成可执行 JavaScript。
- 如用户要求原始 Netron 图，提供可选 `open-netron`；**不要试图直接通过修改 Netron 的私有内部状态给原图染色**。

### 7.3 点击节点的详情栏（必要字段）

```text
节点 ID：main/node_000128
原始名称：/model.3/conv/Conv
算子：Conv
输入/输出 Tensor：...
输入/输出 Shape：...
Conv 参数：Kernel、Stride、Dilation、Group、Pads...
检查状态：VIOLATION
违反规则：X5-CONV2D-KERNEL-H
实际值：32
规则范围：1～31
规则来源：官方文档链接 + 章节
影响分析：可到达的模型输出 + 代表路径
建议：优先重新检查导出结构；必要时改变网络结构并重新训练
提醒：未运行 hb_mapper checker，无法确认真实 CPU/BPU 划分
```

上面数值和路径仅用于界面示例，最终由工具实测与规则输出填充。

---

## 8. 模块 F — 报告、数据协议和 Skill 接入

### 8.1 analysis.json（机器事实层）

建议版本化 JSON 结构：

```json
{
  "schema_version": "1.0",
  "model": {
    "path": "./models/example.onnx",
    "sha256": "<computed_hash>",
    "opset_imports": [{"domain": "", "version": 11}],
    "inputs": [],
    "outputs": [],
    "node_count": 0,
    "operator_counts": {}
  },
  "ruleset": {
    "id": "x5-bayes-e-onnx-conv2d",
    "version": "0.1.0",
    "toolchain_verified": false
  },
  "nodes": [],
  "tensors": [],
  "edges": [],
  "diagnostics": [],
  "traces": [],
  "unverified_assumptions": [],
  "limitations": ["V1 only checks supported Conv2D rules"]
}
```

JSON 中所有数字由程序计算，缺失值用 `null` 或明确状态，不能伪造。模型 SHA256 用于确认“同一份模型分析结果”；规则包版本用于确认“按照哪份约束得到结论”。原始文件不应被修改。

### 8.2 report.md（人类阅读层）

固定章节：

1. 模型概况：输入输出、opset、节点总数、算子计数。
2. 本次规则来源：X5 资料 URL/版本、已覆盖的 Conv2D 条款和未覆盖条款。
3. Conv2D 检查汇总：共检查多少个，分别有多少违反规则、需要验证、已检规则未发现异常。
4. 异常节点清单：原始名称/内部 ID、规则号、实际值、允许值、理由。
5. 节点风险溯源：违规节点到模型输出的代表路径和 Tensor 连接。
6. 优化建议：**具体到节点**，区分“无需修改”“图优化候选”“可能需要改变网络并重训”。
7. 尚不能下结论的事项：如真实 BPU 执行位置、量化误差、延迟等。
8. 文件索引：`analysis.json`、`graph.html`，以及可选 Netron 打开命令。

### 8.3 Codex 如何生成“智能建议”

- `analysis.json` 是可信的静态事实来源。
- Codex 只依据事实、官方文档和已声明的约束缺口写建议；没有证据不要断言一定提高 FPS、一定提升精度。
- 对违规 Conv 的常见建议可以是“检查导出错误”“调整网络架构”“需重新训练/微调”，但**不能自动删去计算节点**。
- 每条建议最好绑定 `node_id`、`rule_id`、`evidence`、`validation_needed`。
- V1 命令行在不联网、没有 Codex 回答时仍须生成基本的确定性报告；AI 建议可以在 Skill 交互阶段补充。

### 8.4 本地 Codex Skill 入口

在 `.agents/skills/rdk-x5-onnx-doctor/SKILL.md` 写**精简工作流**，不是重复本文件的全部架构说明。应包括：

- YAML frontmatter：`name: rdk-x5-onnx-doctor`，`description` 明确触发语句（检查 ONNX、分析 Conv、X5 BPU 兼容性、查看风险路径）。
- 收到模型路径后，先检查 Python 工具是否可用；读取指定规则集，执行 `python -m rdkx5_doctor analyze`。
- 检查生成文件是否存在、CLI 返回是否正常，读取 `report.md` 和 `analysis.json`。
- 结合 `references/official_sources.md` 解读约束并生成不夸大结论的优化建议。
- 给用户报告路径、异常节点以及如何打开 `graph.html` 与 Netron。
- 规则缺失或来源不确定时明示，不得自行填充未经确认的限制。
- 默认只读；不得执行 Docker、量化、原始 ONNX 改写。

项目根部 `AGENTS.md` 只保存仓库级开发规则、启动和测试命令；不要在多个位置重复维护相互矛盾的规则文本。

---

## 9. 必须编写的自动化测试

### 9.1 微型 ONNX 测试模型（用 onnx.helper 在 pytest 中生成）

| 编号 | 测试模型 | 验证内容 |
|---|---|---|
| T01 | 规范 Conv2D（3×3） | 默认属性、形状读取、合理规则 PASS |
| T02 | 超出所选规则上界的 Conv2D | 对应规则明确 FAIL，实际/阈值正确 |
| T03 | 恰好达到规则边界的 Conv2D | 边界包含/不包含处理正确 |
| T04 | Group Conv2D | `C_in_per_group` 计算正确 |
| T05 | Depthwise Conv | Group 与通道语义正确，不产生误报 |
| T06 | `kernel_shape` 省略 | 从权重 Shape 推导 Kernel |
| T07 | 动态 Shape Conv | 能判断的项目照常判断，不可判定的字段 UNKNOWN |
| T08 | 权重 Shape 不可获知 | 不崩溃，不伪造维度 |
| T09 | Conv → 分支 → 汇合 → 双输出 | 依赖图、可达输出与路径正确 |
| T10 | 空名称/重复名称节点 | 稳定内部 ID、结果可关联 |
| T11 | 包含 Add/Reshape/Concat 等非 Conv 节点 | 正确展示，但 `NOT_COVERED` |
| T12 | 非 Conv2D（例如 Conv3D） | 不误套 Conv2D 规则 |
| T13 | 非法 ONNX 文件 | 可解释错误和非零退出码 |
| T14 | 缺失/畸形规则 YAML | 清晰失败，不忽略规则 |
| T15 | HTML 报告 | 文件生成、异常节点数据可查询、离线依赖齐全 |
| T16 | 网络大型分叉图 | 路径条数有限制，且不会无限枚举所有简单路径 |

测试权重应足够小、可快速生成。对于 Conv Kernel 边界，可构造尺寸足够但整体很小的有效测试模型；数值上必须通过 ONNX checker。**先从官方资料确定边界，再构造边界测试，不先写死假设。**

### 9.2 人工体验验收

- [ ] CLI 可以独立运行，无须打开 VS Code、Netron 或 Docker。
- [ ] 自带的有效小模型能够生成完整三个文件。
- [ ] 双输出分支图中，违规节点到两个输出的路径都可以查看。
- [ ] 点击图中红色 Conv，出现规则号、实际值、阈值、官方链接。
- [ ] 非 Conv 节点全部存在于图中，并标注“V1 未检查”而非“BPU 兼容”。
- [ ] 可从界面搜索节点，拖拽、缩放、聚焦；不依赖外网。
- [ ] Netron 缺失不会导致 `analyze` 失败。
- [ ] 原始 ONNX 的 SHA256 前后相同。
- [ ] 报告明确写明“未经过 OpenExplorer/hb_mapper 实测”。
- [ ] `pytest -q` 全部通过。

---

## 10. Codex 分阶段执行计划与每阶段交付物

**开发过程中按阶段实际编写代码并运行测试；不要只创建空文件或停留在方案讨论。**

### P0 — 初始化仓库（先完成）

任务：创建 `src` 包、`pyproject.toml`、CLI 框架、pytest、`.gitignore`、README、AGENTS.md。锁定主要依赖的合理版本范围。确认 `.venv` 可运行。

交付：

```bash
python -m rdkx5_doctor --help
pytest -q
```

### P1 — ONNX Reader + Graph IR

任务：实现模型元信息、节点/张量提取、依赖边、动态维度标记、稳定 ID。先完成 T01、T07、T09、T10、T13。

交付：程序可输出结构完整的 `analysis.json` 初版，不做 BPU 判断。

### P2 — Conv2D extractor + 官方资料抽取

任务：实现所有本文件列出的 Conv 参数与默认值；Codex 阅读**指定官方 X5 ONNX**文档，生成 `manifest.yaml`、`conv2d.yaml`、`conv_rule_notes.md`；保留来源与待验证条件。通过 `rules validate`。

交付：Conv2D 参数和 Conv 规则文件均可单独检查。实现 T02～T06、T12、T14。

### P3 — Rule Engine

任务：只执行已标注 `auto_check` 的 Conv 条件，返回逐规则证据及节点汇总。缺字段返回 UNKNOWN，不覆盖其他算子。

交付：JSON 可以准确标出违规 Conv 的内部 ID、规则、实际值、阈值和来源。

### P4 — Node Risk Tracer

任务：从违规节点到 graph outputs 做真实拓扑追踪；对分支合流图正确处理；限制路径枚举。

交付：`traces[]` 可供报告和 HTML 直接读取。实现 T09、T16。

### P5 — HTML/Markdown/Netron

任务：生成 `report.md` 与离线 `graph.html`；前端可搜索、缩放、选中节点、过滤异常、展示证据和路径。添加可选 Netron 打开命令。

交付：完成 T15，实际在本机浏览器打开并检查。若 Codex 无浏览器自动化能力，至少检查生成文件、依赖路径和 DOM 数据，并在交付说明中标记视觉测试未完成。

### P6 — Codex Skill + 最终回归

任务：编写可触发的精简 SKILL.md，编写 README 使用说明；使用一个完整实例从 Codex 自然语言调用到报告产出。运行 `pytest -q`，列出通过/失败测试。

最终交付：仓库源码、规则集、测试、三个报告产物示例（小型模型）、README、SKILL.md。**不要把大体积真实 ONNX 权重和编译结果提交进 Skill 目录。**

每阶段完成时，Codex 在执行记录中写明：修改的文件、运行的命令、测试结果、剩余问题。不要声称未经实际运行的检查“已通过”。

---

## 11. 未来扩展点（仅记录，不在 V1 实现）

- **V2 多算子**：增加 Add/Mul/Concat/Reshape/Transpose 等，优先由真实 YOLO 算子分布决定；每种算子继续走“官方文档总结→来源标注→机器检查→节点溯源”流程。
- **V3 图优化建议**：Identity 消除、连续逆 Transpose、常量折叠、Conv-BN 融合候选；先在模型副本上执行，验证 ONNX 合法性与数值等价性，绝不直接破坏原始模型。
- **V4 工具链验证**：从另一分支获得 Ubuntu/Docker 环境信息后，新增 `ToolchainAdapter`，使用 `hb_mapper checker` 核验静态判断、实际 CPU/BPU 分配和子图切分，并标记“静态预测 vs 工具链结果”。
- **V5 量化诊断**：引入校准数据、量化配置、相似度、逐节点误差与真实性能数据。不要将高 Conv 数量简单作为量化精度下降原因。
- **V6 假设性优化分析**：比较候选修改前后的静态约束状态，明确这种模拟不代表精度或延迟改善已经得到验证。

现在不做上述扩展。**V1 的完成质量优先于覆盖算子数量。**

---

## 12. 给 Codex 的首条任务指令（可直接复制）

> 请阅读仓库中的 `RDK_X5_ONNX_Doctor_V1_Development_Spec.md`，按该文档开发 RDK X5 ONNX Doctor V1。不要停留在架构解释，必须实际创建可运行的代码、CLI、测试、规则集、Netron 风格交互式诊断图和 `.agents/skills/rdk-x5-onnx-doctor/SKILL.md`。先处理 P0，再按 P1→P6 分阶段完成，并在每阶段运行对应测试。V1 只支持 **Conv2D 的 BPU 规则**，其他算子只解析展示，不评分、不预测量化精度，不进入 Docker，不运行 OpenExplorer，不修改原始 ONNX。约束规则必须从**RDK X5 的 ONNX 官方文档**提炼，标注来源与版本；无法验证时返回 UNKNOWN，不能凭空填数。先用测试生成的小型 ONNX 验证，再使用可获取的真实模型。输出 `analysis.json`、`report.md` 和离线可用的 `graph.html`；最终给出实际执行的测试命令与结果，以及尚未覆盖的约束和风险。遵循文档中的验收清单；遇到缺少真实模型时不要阻塞，用微型 ONNX 完成验收。

---

## 13. 参考资料（开发时应优先使用官方原始资料）

1. **地瓜机器人：RDK X3/X5 模型算子支持列表（中文）**，注意切换至“RDK X5 支持的 ONNX 算子列表”中的 Conv 行：<br>
   https://developer.d-robotics.cc/rdk_x_doc/Advanced_development/toolchain_development/intermediate/supported_op_list
2. **X5 芯片用户手册 1.1.2：算子支持列表**，交叉核对版本：<br>
   https://developer.d-robotics.cc/x5_sdk_doc/toolchain_development/intermediate/supported_op_list.html
3. **X5 芯片用户手册 2.0.0：算子支持列表（英文）**，用于对照可能的版本差异：<br>
   https://developer.d-robotics.cc/x5_sdk_doc_v2.0.0/en/toolchain_development/intermediate/supported_op_list.html
4. **OpenExplorer：hb_mapper checker**（V1 只作未来扩展依据，不调用）：<br>
   https://developer.d-robotics.cc/oe_x5_doc/cn/oe_mapper/source/ptq/ptq_tool/hb_mapper/hb_mapper_checker.html
5. **ONNX Conv 算子规范**（属性与权重格式）：<br>
   https://onnx.ai/onnx/operators/onnx__Conv.html
6. **Netron 官方仓库**（模型可视化及 Python 启动方式）：<br>
   https://github.com/lutzroeder/netron
7. **Codex 仓库级 Skill 组织参考**：<br>
   https://developers.openai.com/blog/skills-agents-sdk

**最后一条原则：**所有“违反约束”结论都必须能回溯到 `ONNX 节点 ID → 真实字段值 → YAML 规则 ID → 官方来源`。这条证据链比评分系统更重要，也是本课程作品的核心特色。
