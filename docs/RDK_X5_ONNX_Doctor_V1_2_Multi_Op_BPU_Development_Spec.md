# RDK X5 ONNX Doctor V1.2 — 多算子 BPU 约束规则引擎开发规范

> **交付对象**：Ubuntu / WSL2 中 VS Code + Codex，直接在已克隆的 `Stradlin1/skillzuoye` 仓库执行。
>
> **本轮范围**：升级版本化 YAML 规则引擎，新增 **Mul、Sigmoid、Add、Concat、Slice、Gemm** 六类 ONNX 算子的 RDK X5 静态 BPU 约束诊断；同步重构 Markdown 报告生成策略、报告版本及 `.gitignore`；更新 Codex Skill。
>
> **禁止事项**：不进入 Docker，不运行 hb_mapper/PTQ/板端，不修改/重写/精简用户的 ONNX，不做自动图优化、不删节点、不评分、不开发 GUI/HTML。不臆测真实 CPU/BPU 分配、量化精度、耗时或设备内存。
>
> **工程原则**：诊断工具只定位 ONNX 问题并提供证据。需要改变网络结构、算子或导出方式时，**回到原始训练/导出工程修改 PyTorch 模型代码或导出逻辑，重新导出 ONNX，再用本工具复检**。不得在本项目里对模型 ONNX 文件进行原地修补。

## 目录

1. [目标与已验证的基线](#1-目标与已验证的基线)
2. [规则证据来源与版本策略](#2-规则证据来源与版本策略)
3. [现有代码与改造边界](#3-现有代码与改造边界)
4. [多算子规则库数据结构](#4-多算子规则库数据结构)
5. [六类算子的开发要求](#5-六类算子的开发要求)
6. [节点级结论、证据和溯源](#6-节点级结论证据和溯源)
7. [报告生成规则 V1.2](#7-报告生成规则-v12)
8. [`.gitignore` 与历史报告清理](#8-gitignore-与历史报告清理)
9. [CLI、JSON 协议、Skill 和文档](#9-cli-json-协议skill-和文档)
10. [测试用例矩阵与真实模型回归](#10-测试用例矩阵与真实模型回归)
11. [按阶段执行的 Codex 工作计划](#11-按阶段执行的-codex-工作计划)
12. [验收标准与最终输出](#12-验收标准与最终输出)

---

## 1. 目标与已验证的基线

### 1.1 仓库状态：先阅读，不要推倒重写

仓库：`https://github.com/Stradlin1/skillzuoye`。

开始时必须读取：

- `README.md`、`AGENTS.md`、`.gitignore`、`pyproject.toml`。
- `.agents/skills/rdk-x5-onnx-doctor/SKILL.md`。
- `docs/ANALYSIS_SCHEMA_V1_1.md` 和 `docs/DEVELOPMENT_LOG.md`。
- `src/rdkx5_doctor/onnx_reader.py`、`graph_ir.py`、`conv_extractor.py`。
- `src/rdkx5_doctor/rules.py`、`report.py`、`cli.py`、`trace.py`。
- `src/rdkx5_doctor/tensor_resource.py`、`optimization_candidates.py`、`resource_queries.py`。
- `src/rdkx5_doctor/resources/rulesets/x5-bayes-e/manifest.yaml`、`conv2d.yaml`。
- `reports/yolo26_lane_robot-20261009-181829/REAL_ONNX_VALIDATION.md` 与 `analysis.json`（如本地存在）；所有已提交的历史报告只作回归证据。

当前基线（以仓库真实检查为准）：包版本 `0.2.0`、analysis schema `1.1`、Conv 规则包 `0.1.0`。V1.1 已有计算图解析、Conv2D 19 条规则、Tensor 资源估算、五种优化候选、终端查询及 Codex Skill。真实 YOLO26 验收后仓库记录 **141 项测试通过**；启动时重新运行，不把历史日志当作本次测试。

### 1.2 真实验证模型

```text
/home/xhm/lianghua_ws/rdktoolchain/horizon_x5_open_explorer_v1.2.8/horizon_x5_open_explorer_v1.2.8-py310_20240926/samples/ai_toolchain/horizon_model_convert_sample/04_detection/15_tasknav/model/yolo26_lane_robot.onnx
```

已记录的模型事实：ONNX opset 11、281 个节点、23 种算子、47 个 Conv、1 输入、2 输出。以下六类节点数来自上一轮分析：

| 新增算子 | 真实模型节点数 |
|---|---:|
| Mul | 51 |
| Sigmoid | 38 |
| Add | 14 |
| Gemm | 12 |
| Concat | 11 |
| Slice | 8 |
| **合计** | **134** |

目标是在现有 47 个 Conv 上，把这 **134 个新增目标节点** 纳入静态规则分析工作流；不意味着它们最终全部得到 PASS，允许 `NEEDS_VERIFICATION` / `UNKNOWN`。剩余 100 个其他节点应保留 `NOT_COVERED`，不能称作兼容。

### 1.3 本轮应真正完成的事情

1. 将当前只能读取 `conv2d.yaml`、`operator: Literal['Conv']` 的单算子架构升级成安全、可版本化的**多算子规则注册与执行架构**。
2. 六类算子的 `Rule Extractor` 能从 GraphIR / ONNX 元信息提取可靠字段，不能依靠原始节点名称猜值。
3. 依据 RDK X5 官方文档，对“可静态判定”的约束执行确定性检查；对推导不足或工具链依赖的条件保留未知。
4. 每个节点的诊断都能追溯到 `node_id → tensor/attribute → rule_id → expected/actual → 官方 X5 文档版本/章节/URL`。
5. 压缩 `report.md`：摘要、按算子覆盖矩阵、异常、未知及必要建议；**不再为所有正常 Conv 打印“无需修改”**。
6. 让 `reports/` 恢复默认忽略，Git 仓库仅保留简洁的、可复现的精选验证摘要；清理 Git 索引中之前已跟踪的大量日志（不删除本地文件）。
7. 保持 V1/V1.1 的所有既有功能和命令可用，运行真实模型验证。

## 2. 规则证据来源与版本策略

### 2.1 官方资料（由 Codex 在线重新核对，不能只信本文件）

首选官方 RDK X5 ONNX 列表：

- `https://developer.d-robotics.cc/rdk_x_doc/Advanced_development/toolchain_development/intermediate/supported_op_list`
- X5 芯片用户手册 1.1.2：`https://developer.d-robotics.cc/x5_sdk_doc/toolchain_development/intermediate/supported_op_list.html`
- X5 Chip User Manual 2.0.0：`https://developer.d-robotics.cc/x5_sdk_doc_v2.0.0/en/toolchain_development/intermediate/supported_op_list.html`
- ONNX 算子标准：`https://onnx.ai/onnx/operators/`（解释 ONNX 语义，**不是** BPU 限制来源）。

**必须定位到“RDK X5 支持的 ONNX 算子列表”章节**；不要误读同页面的 X3 ONNX 表、Caffe 表或最右侧 CPU 约束列。阅读官方版本变化、默认 CPU 运行说明、opset 10/11 的条件差异。

用户 ONNX 位于名称带 `open_explorer_v1.2.8` 的目录，只能说明文件所在路径；**不能当作已通过 hb_mapper 或已确认该工具链与最新网页完全一致的证据**。本轮不接入工具链，保留 `toolchain_version: unverified` / `toolchain_verified: false`，新增规则同时记录所依据的官方文档版本与网页抓取时间。

### 2.2 AI 从文档总结到 YAML 的受控流程

对每个算子严格执行：

1. 记录页面标题、官方 URL、文档版本、算子行、**BPU 约束栏**、与其他文档的差异。
2. 把每条原始约束拆解为“适用前提 / 检查字段 / 比较逻辑 / 静态能否证明 / 缺失信息 / 官方证据”。
3. 将有明确判断方法的条款写为 YAML + 白名单谓词；复杂条件只通过**已审查的 Python predicate** 完成，不允许 YAML 执行表达式。
4. 每条规则配置 `checkability: auto_check | conditional | review_only`；不确定的不要硬转为 `FAIL` 或 `PASS`。
5. 每条 YAML 都关联对应的边界/反例测试；未获得可靠来源的规则不能进入可执行 BPU 规则库。
6. 保存 `references/x5_multiop_sources.md` 和 `references/x5_multiop_rule_review.md`：列出逐算子原文摘要、证据链接、冲突、被排除条件、解释依据及尚未解决的问题。

**禁止事项**：不允许 AI 仅根据训练经验猜出“Mul 容易量化失败”等规则；不允许从 Torch/NVIDIA/X3 限制推断 X5 约束；不允许自动把 ONNX 类型合法性限制伪装成 BPU 硬件限制。

### 2.3 规则版本不是工具链实测版本

必须拆分四个概念：

- `ruleset_version`：本项目 YAML / predicate 语义的版本。
- `source_version`：官方网页/手册的文档版本。
- `onnx_opset`：当前模型的 ONNX 标准版本，真实模型为 11。
- `toolchain_version/toolchain_verified`：是否已在特定 OpenExplorer/hb_mapper 上实际验证；本轮为未验证。

源文档存在差异时优先保守处理：保留双方来源与冲突，不擅自选取更宽松或更严格值当作“通用 X5 限制”。缺乏版本匹配证据的判断用 `UNKNOWN` 或 `REVIEW_ONLY`，不是用二者混合出来的常量。

---
## 3. 现有代码与改造边界

### 3.1 当前耦合点（必须修改）

当前 `src/rdkx5_doctor/rules.py` 存在这些 V1 限制：

- `RuleSet.operator` 仅接受 `Conv`。
- `load_ruleset(directory)` 固定加载 `manifest.yaml` 与 `conv2d.yaml`。
- `rules.py` 校验器字段白名单仅来自 `conv_extractor.FIELDS`。
- `check_rule(rule, fields, scope)` 和 `check_node()` 的语义隐含 Conv2D。
- `report.py:analyze()` 只对 Conv 执行规则，其余节点返回 `NOT_COVERED`；汇总只有 `conv_status_counts`。
- `report.py:markdown()` 将所有通过的 Conv 的“无需修改”逐行写入最终报告。

应当**渐进拆耦**，不要删除稳定的 Conv 实现。V1.1 的 `GraphIR`、资源模块、优化候选、CLI 的 `nodes/inspect/trace/tensors/candidates` 接口必须保留。

### 3.2 推荐新增/调整的模块

```text
src/rdkx5_doctor/
  graph_ir.py                  # 已有，尽量不动
  onnx_reader.py               # 已有，如确有需求仅增量补元信息
  conv_extractor.py            # 已有，回归不变
  operator_facts.py            # 新：通用字段提取接口及来源判定
  elementwise_extractor.py     # 新：Add/Mul 广播、输入种类、rank
  shape_op_extractor.py        # 新：Concat/Slice/Sigmoid 适用字段
  gemm_extractor.py            # 新：Gemm 矩阵维度/属性/转换待证
  rules.py                     # 改：schema、Registry、规则执行调度
  rule_predicates.py           # 新：经测试的谓词白名单
  report.py                    # 改：多算子 analyze + 精简 Markdown
  reporting_sections.py        # 可选：报告段落生成，防止 report.py 膨胀
  cli.py                       # 改：查询和 analyze 汇总
  trace.py                     # 原实现保留
  tensor_resource.py           # 原实现保留
  optimization_candidates.py   # 原实现保留

src/rdkx5_doctor/resources/rulesets/x5-bayes-e/
  manifest.yaml                # 升级：多算子注册信息
  conv2d.yaml                  # 原 19 条约束语义不变
  add.yaml                     # 新
  mul.yaml                     # 新
  sigmoid.yaml                 # 新
  concat.yaml                  # 新
  slice.yaml                   # 新
  gemm.yaml                    # 新

references/
  x5_multiop_sources.md       # 新：官方条款与版本清单
  x5_multiop_rule_review.md   # 新：已实现/条件/冲突/排除记录

docs/
  ANALYSIS_SCHEMA_V1_2.md     # 新
  REPORT_POLICY_V1_2.md       # 新：人类报告生成规则
  validations/                # 可选：少量精选、脱敏后的真实模型验收摘要

tests/
  test_multiop_registry.py
  test_add_mul_rules.py
  test_sigmoid_concat_slice_gemm.py
  test_report_policy_v1_2.py
  test_git_hygiene.py         # 可选：检验忽略规则和示例文件仍被跟踪
```

允许根据现有工程习惯合并模块；不强制增加无必要文件。**不要复制一份旧 rules.py，不能用两套规则引擎并行判断。**

### 3.3 数据流

```text
ONNX 文件（只读）
    ↓
onnx_reader → GraphIR（node / tensor / edge / opset）
    ↓
OperatorFactExtractor（按 op_type 取字段、保持 UNKNOWN）
    ↓
RuleRegistry（按 operator + domain + opset + source-version 分派）
    ↓
Rule Engine（白名单操作符 + 条件判断）
    ↓
每个节点：规则结果 + 原始值 + 允许值 + 官方来源 + 限制
    ↓
trace_node（仅对确有需要的节点追踪）
    ↓
analysis.json 1.2（完整事实）
    ↓
report.md（异常优先、简洁） + 终端 inspect/nodes 查询
    ↓
Codex Skill 结合证据提出“训练/导出工程中的修改建议”
```

### 3.4 兼容契约

- V1 原有 19 条 Conv2D 规则 ID / 边界/状态不被改写（除非明确另立独立勘误流程）。
- 新注册机制加载内置资源时仍适用于 wheel / sdist 和工作目录外安装。
- `nodes`、`inspect`、`trace` 能读取历史 schema `1.0`、`1.1` 和新 `1.2`（如之前规则结果字段差异，提供只读兼容层）。
- `tensors`、`tensor`、`candidates`、`candidate` 继续能读 `1.1`/`1.2`，遇到 `1.0` 保留当前清晰报错。
- `analysis.json` 在 `1.2` 继续保留旧顶层键 `model/ruleset/nodes/tensors/edges/diagnostics/traces/summary/unverified_assumptions/limitations/resource_analysis/optimization_candidates`。
- `--ruleset`、`rules validate`、默认包内规则均继续可用。
- 任何新增规则不得依赖大型原始权重物化。常量只用 initializer 元信息、GraphIR 中已缓存的有限小常量；若确需标量，沿用 bounded decode 的安全策略。

## 4. 多算子规则库数据结构

### 4.1 RuleRegistry

现有 `load_ruleset` 返回 `RuleSet` + `Manifest`。改造时引入 `RuleRegistry`，示意：

```python
@dataclass
class RuleRegistry:
    manifest: Manifest
    by_operator: dict[str, OperatorRuleSet]
    rules_by_id: dict[str, Rule]

    def get_rules(self, operator: str, domain: str, opset: int):
        # 只接收已确认的标准域/正确版本；未匹配不得当作 PASS
        ...
```

`OperatorRuleSet` 需要有 `operator`、`model_domain`、`scope`、`ruleset_version`、`rules`。允许 `scope` 为明确的判定范围（如 `standard_onnx`、`conv2d`、`rank_1_to_10`），不再写死为 Conv 的 `input_rank_4`。

要求：

1. 文件加载顺序稳定，manifest 明确注册算子→相对 YAML 文件路径；不要运行时递归执行未知 YAML 文件。
2. 不允许重复 `rule_id`、重复 operator 文件、越出规则目录的 `../` 路径、绝对路径、软链接逃逸。
3. 对所有 YAML 用 `yaml.safe_load` 与 Pydantic `extra='forbid'` 严格校验。
4. 所有字段名、谓词名有注册白名单；不支持 `eval`、`exec`、`lambda`、任意 import、用户传入的脚本。
5. 校验 manifest 与每个算子文件的版本、域、来源、OPSET 条件一致。
6. 保证已有单算子直接调用测试尽量兼容；若迁移了公开 API，同步添加明确兼容包装与测试。

### 4.2 新 manifest 的建议结构（仅示意，需编程后做严格 schema）

```yaml
platform: "RDK X5"
march: "bayes-e"
ruleset_id: "x5-bayes-e-onnx-multiop"
ruleset_version: "0.2.0"
toolchain_version: "unverified"
toolchain_verified: false
source_policy: "X5_ONNX_BPU_COLUMN_ONLY"
source_review_status: "reviewed_with_exclusions"
operators:
  Conv: conv2d.yaml
  Add: add.yaml
  Mul: mul.yaml
  Sigmoid: sigmoid.yaml
  Concat: concat.yaml
  Slice: slice.yaml
  Gemm: gemm.yaml
```

历史 `conv2d.yaml` 仍可能内含旧 `ruleset_id/version`。先设计清晰的迁移策略：**Conv 规则子文件版本可保持 0.1.0，Registry 总版本为 0.2.0**，而非悄悄改写历史 Conv 来源。所有子版本记录到分析结果，且自动验证唯一性。

### 4.3 单规则必须保存的证据

至少包含：

```yaml
id: X5-SIGMOID-INPUT-RANK
operator: Sigmoid
check_type: range
field: input_rank
min: 1
max: 10
checkability: auto_check
source_url: https://developer.d-robotics.cc/x5_sdk_doc/toolchain_development/intermediate/supported_op_list.html
source_document: "X5 芯片用户手册"
source_version: "1.1.2"
source_section: "X5 支持的 ONNX 算子列表 / Sigmoid / X5 BPU 支持约束"
source_column: "X5 BPU 支持约束"
reason_code: "BPU_RANK_OUT_OF_RANGE"
message: "输入 rank 未落在所引用 X5 文档列出的范围内"
```

以上展示的是官方列出的 rank 约束如何转成规则的**格式示例**；正式提交前仍需由 Codex 查源版本、核对 ONNX 适用场景并添加上下边界测试。不要把“int16 输入输出支持”理解为“原始 FP32 ONNX 必须是 int16”。

每条检查运行结果必须含：

```json
{
  "rule_id": "X5-SIGMOID-INPUT-RANK",
  "status": "PASS",
  "node_id": "main/node_000001",
  "operator": "Sigmoid",
  "observed": {"field": "input_rank", "value": 4},
  "expected": {"min": 1, "max": 10},
  "checkability": "auto_check",
  "evidence": {"input_tensor": "example_feature", "shape": [1, 32, 64, 64]},
  "source": {
    "document": "X5 芯片用户手册",
    "version": "1.1.2",
    "section": "X5 ONNX / Sigmoid / BPU 栏",
    "url": "https://developer.d-robotics.cc/x5_sdk_doc/toolchain_development/intermediate/supported_op_list.html"
  },
  "reason": "已核对的输入 rank 为 4",
  "unverified": ["尚未执行编译器设备分配检查"]
}
```

示例节点仅作协议说明，不是用户真实模型中的结果。实际 JSON 不必逐字采用命名，但以上证据语义必须完整且可机器查询。

### 4.4 Predicate 白名单

沿用已有 `range / max_value / equals / one_of`，根据本轮实际约束按需增加：

- `min_value`：下界。
- `is_false` / `is_true`：严格布尔条件（不可把 `None` 当 false）。
- `integer_rank_range`：rank 上下界。
- `broadcast_compatible`：ONNX 广播语义层面的确定性判断。
- `x5_elementwise_broadcast_mergeable`：X5 文档中的广播维度合并条件，使用经单测覆盖的算法；**无法证明时 UNKNOWN**。
- `at_most_one_fixed_constant_input`：识别输入来源与数量。
- `concat_axis_not_batch`：按规范化 axis 与 rank 识别是否在 N 维拼接。
- `gemm_conversion_review`：只记录可证矩阵维度和 Conv 转换待验证条件，不能冒充已知最终 Conv 布局。

所有新 predicate 必须具备独立单元测试；不得因算子较复杂就退回由 Codex 每次即兴给出 PASS/FAIL。多值条件可由白名单内的 AND/OR 描述树组合，但禁止任意数学表达式字符串。

### 4.5 Operator Fact Extractor 通用字段

每个检查器应至少输出：

```text
node_id, original_name, op_type, domain, imported_opset
input_tensors[]: name, shape, dtype, kind, producer, consumers,
                 fixed_constant_status, graph_input_override
output_tensors[]: name, shape, dtype, is_graph_output
attributes, derived_fields, field_evidence, extraction_issues
```

关键规则：

- **张量连接依据真实 Tensor name**，不得依据节点下标相邻或者 Conv/Sigmoid/Reshape 名称推测。
- **常量来源**：Initializer 且不是可覆盖的 graph input；或经检查的 Constant 节点输出。其它不明来源为 UNKNOWN；不能把所有没有 producer 的输入都当常量。
- 区分轴的原始值和归一化值：`axis=-rank` 对应 batch/N 轴；`axis=0` 不一定在未知 rank 时可直接验证语义。
- 不猜测通道是第几维，除非算子规则/官方来源明确采用 NCHW/对应布局。
- 不确定的 shape 维度使用 `null`/符号值并记录来源；不能用 1 代替未知。
- 大模型权重仅元信息读取。禁止为了判断广播/通道而对 100 MB 的模型做大规模 ndarray 转换。

---
## 5. 六类算子的开发要求

> 以下为对官方 X5 ONNX 表的**研究起点和最低测试要求**。Codex 开发时必须逐算子对照文档版本，将真正写入 YAML 的条件与证据一一对应；如官方不同版本有差异，记录冲突并保守处理。本节绝不是让开发者忽略核验直接硬编码。

### 5.1 第一批开发顺序

建议按 **Sigmoid → Concat → Slice → Add → Mul → Gemm** 实施：先跑通简单一元/轴约束，再完善复杂广播与 Gemm 转换；每完成一个算子先做 fixture 与真实模型子集回归，不要六个同时写完才测试。

#### A. Sigmoid（真实模型 38 个）

官方 X5 ONNX 表给出的重点：支持非四维场景，存在输入/输出 `1–10` 维的约束，并描述 int16 输入输出能力。后者**不是**原始 ONNX `float32` 不兼容的证据。

Extractor：

- 标准 ONNX domain、真实导入 opset。
- 输入/输出数量、shape、rank、dtype。
- rank 完整时执行来源确认的 rank 范围检查。
- 源 Shape 未知时记录 `UNKNOWN`；不把 shape-inference 推断的符号轴误认为 rank 不存在。
- 若文档仅写“支持 int16”，不得据此对 float32 输入判 FAIL。

验收 fixture：rank=1、4、10；低于支持范围和高于范围的合法 ONNX 对照；缺失输入 shape；自定义域同名节点；ONNX schema 不适用的版本。

#### B. Concat（真实模型 11 个）

官方 X5 ONNX 约束重点：**不支持在 N 维度拼接**；文档亦列出 int16 输入输出能力。

Extractor：

- 获取 `axis`，按模型输入的 rank 做负轴归一化；校验 input rank 是否一致及轴是否在 ONNX 有效范围（ONNX 语义与 BPU 条件分开报告）。
- 当标准域且 axis 已确定对应 N/batch 维时，以来源确认为前提输出 `FAIL`。
- 当 axis 明确为其它维度时，该条规则可 `PASS`，但不等于保证整个 Concat 必定分配 BPU。
- 输入 rank / 属性 / 轴意义不确定时用 UNKNOWN；不要强制把“张量第一维”推广到不存在 NCHW 前提的所有布局。
- 针对模型输出 `cls_logits`、`offset` 的末端 Concat，保留 output Tensor 名称及 shape，不假设导出时的 axis 与上游 Conv 相同。

验收 fixture：axis=0、1、-rank、-1；静态与符号化 shape；不同 rank 的 ONNX 不合法反例；可追踪到模型输出的多个 Concat。至少有一个来源证明正确的 X5 BPU FAIL 例。

#### C. Slice（真实模型 8 个）

官方 X5 ONNX 表将 Slice 列为 BPU 可加速，描述“无限制，支持非四维输入输出”，并指出 int16 能力；这**不等于**允许运行时对任意动态 starts/ends 的转换一定成功，也不能臆造 X3 的 Slice 约束。

Extractor：

- ONNX opset 11：识别 data、starts、ends、可选 axes、steps 的输入张量来源、shape、dtype、是否固定常量。
- 小型安全整数常量可按现有 `small_constants.py` 有界解析；外部大权重/动态参数保留未知。
- 区分两个判断：① ONNX 自身的索引规范和输入类型；② 官方 X5 BPU 列明确列出的硬件条件。
- 对没有进一步 BPU 数值限制的文档，不凭空生成 max starts / max steps 规则；静态属性已核对可以返回“已核对的明确条款未发现限制”，而将动态参数的可部署性单独标为待工具链确认。
- 对由 Shape/Gather 算出来的 Slice 边界，只做来源溯源，不趁本轮加入庞大的 Shape 常量传播器。

验收 fixture：固定 starts/ends 与动态参数、axes/steps 省略、反向 step、非四维 data、缺失元信息、合法标准 ONNX 的小示例。**不能因没有数值硬约束就在规则集里制造无意义的 FAIL 测试。**

#### D. Add（真实模型 14 个）

官方 X5 ONNX 表：支持 featuremap / 常量输入、最多一个常量；存在广播与高维广播合并规则，支持某些非四维输入；ResNet shortcut 的 Add 可能融合至前面的 Conv。

Extractor：

- 两个输入的 shape/rank、dtype、固定常量来源、生产者/消费者。
- 先检查 ONNX 广播的基本合法性：从尾维对齐，维度相等或其中一个为 1；符号维度冲突若无法证明则 UNKNOWN。
- 再按 X5 文档检查 BPU 的 1–10D 适用范围及 `>4D` 的合并约束；绝不能只要 ONNX 广播合法就判 X5 BPU 通过。
- 对两个可证明为**不可覆盖固定常量**的输入，检查官方“最多一个常量输入”条款；来源未知不能判定两个常量。
- ResNet shortcut / Conv+Add 融合属于图模式/编译器行为：仅输出 `review_only`，不能单看“Conv 后面是 Add”就断言已融合。
- 特别处理 `NCHW ↔ 特殊广播` 与不同 rank 的广播：若合并规则暂时没有完备算法，先覆盖可证明的常见场景，对其余返回 UNKNOWN，并明确“广播语义已知，BPU 合并可行性未证明”。

验收 fixture：同形状、单边广播、双方互相广播、高维可合并/不可合并、两个常量、一个常量、动态 Shape、不覆盖 initializer、自定义域。**分别验证 ONNX 合法和 X5 BPU 条件。**

#### E. Mul（真实模型 51 个，最重要的数量覆盖）

官方 X5 ONNX 表与 Add 类似：最多一个常量、多个广播形式、1–10D 及高维合并规则。**严禁误将旧 X3 的“Mul 必须 4D / C ≤ 2048”硬编码为 X5 规则。**

Extractor 和规则：

- 复用 Add 的通用 `elementwise_extractor.py` 与 `x5_elementwise_broadcast_mergeable`，但每条规则仍持有自己的 Mul ID 和官方 Mul 行的来源，不能笼统引用 Add 行。
- 识别 `Conv → Sigmoid → Mul` 中真实 Tensor 的生产消费关系，可作为 `pattern_evidence` 描述 SiLU-like 结构；但**不要**因此推断实际融合、运行设备或量化误差。
- 对 Mul 的广播 shape、rank、常量来源的检查，与 Add 共享算法以避免语义分叉。
- 分清“已静态验证 X5 文档条款”与“FP32 浮点图实际 PTQ 后会选择哪种精度”两个概念。

验收 fixture：和 Add 共享参数化广播 oracle；另有 Conv/Sigmoid/Mul 连接路径、动态输入、graph-output 消费路径。必须确保用户真实模型 51 个 Mul 逐个产生明确的 checked/unknown/violation 记录，而非直接 NOT_COVERED。

#### F. Gemm（真实模型 12 个，正确处理条件依赖很重要）

官方 X5 ONNX 列的重点：**Gemm 将转换成 Conv 实现，边界约束参照 Conv**。但未提供“任意 2D Gemm 原始 Shape 可以直接代入 4D Conv 维度限制”的证明。

Extractor：

- 输入 A/B/C 的实际 shape / rank、是否 initializer/constant、dtype、连接关系。
- 按 ONNX 导入 opset 读取 `alpha`、`beta`、`transA`、`transB`（应用 ONNX 默认值）。
- 静态推导 Gemm 的逻辑 M、K、N 与矩阵乘法 Shape 是否相容；此处是 **ONNX 语义核验**，不要直接标记成 X5 BPU 违规。
- 明确记录 `conversion_target: Conv`、`conversion_layout_known: false`（除非确有官方/工具链证据使其可证明）。
- 不能把 `Conv2D` 的 W/H/stride/padding 直接拿 2D Gemm 原始张量填充；若转换细节未知，应把“转换后是否满足 Conv 条款”设为 `UNKNOWN`/`REVIEW_ONLY`。
- 任何 alpha、beta、transpose 相关的 BPU 失败结论都必须找到**明确的 X5 BPU 文档约束**；不能只因其不是默认值就判违规。

验收 fixture：标准 Gemm、transA/B、可选 C / bias、缺失 B shape、动态 M/K/N、二维逻辑 Shape 不匹配的 ONNX 反例，以及“ONNX 逻辑合法，但 X5 Gemm→Conv 实现仍未知”的典型结果。

### 5.2 不扩大范围的提醒

真实模型还有 MatMul 2、Softmax 1、Resize 1，以及 Gemm 周围的 Flatten/Tanh 等节点。**本轮优先完成以上六类**。MatMul、Softmax、Resize 可以在下一批扩展；只在 `report.md` 标注它们仍未覆盖和需进一步检查，尤其 Softmax 的默认 CPU、run_on_bpu 条件不能因为尚未实现就杜撰成本轮的失败。

### 5.3 普遍约束的处理原则

官方 X5 文档有一般 BPU 输入输出 rank、N/C/H/W、Tensor 尺寸描述，也有对特定算子的显式例外。本轮如果实现 `general.yaml`，必须：

- 支持算子例外覆盖一般规则，不得把“一般 4D”套在明确支持 1–10D 的 Add/Mul/Sigmoid 等算子上。
- 对官方写法中“元素数量”与“bytes”混合表述、有对齐/量化配置前提的资源条款，先记为 `review_only`；不能拿 V1.1 的 FP32 逻辑载荷作 BPU 实际分配的 FAIL 判据。
- 一般规则不能掩盖六类 op 自身的证据或因版本不明引入大量假阳性。

## 6. 节点级结论、证据和溯源

### 6.1 兼容的节点四状态

保留 CLI 已使用的四个节点状态：

| 节点 status | 严格含义 |
|---|---|
| `VIOLATION` | 至少一条**已核实且适用的 X5 BPU 硬约束**确定 FAIL；不是模型无法运行的证明 |
| `NO_VIOLATION_FOUND` | 至少一条适用的自动检查已执行、所有适用已知硬约束未违规、无阻碍性 UNKNOWN；**不是 BPU 加速保证** |
| `NEEDS_VERIFICATION` | 关键字段未知、工具链/转换条件未知、官方条款冲突或存在未完成的适用必要检查 |
| `NOT_COVERED` | 无该算子的有效规则集，或标准域/opset/规则范围尚未覆盖 |

逐规则仍为 `PASS / FAIL / UNKNOWN / NOT_APPLICABLE`，但必须附 `reason_code` 和证据。`NOT_APPLICABLE` 不能算 PASS；没有执行任何适用规则时不得自动得到 `NO_VIOLATION_FOUND`。

计算优先级：**可信 FAIL > 未能完成必要检查的 UNKNOWN > 有实际检查的 no-violation > NOT_COVERED**。ONNX 自身结构错误应单独报告，不当成 BPU FAIL。

### 6.2 不得制造硬件结论

- `VIOLATION`：写“违反已收录版本的静态 BPU 约束”；不能写“必定 CPU 回退/量化失败”。
- `NO_VIOLATION_FOUND`：写“已检查条件未发现违规”；不能写“该节点在 BPU 上运行”。
- `UNKNOWN`：必须告诉用户缺的是什么：shape、opset、源版本、Gemm 转换布局、动态 Slice 形参、广播可合并性、是否固定常量等。
- “支持 int16 输入输出”表示官方能力陈述；没有 PTQ 输出 dtype 时不能把原图 float32 直接判失败。
- “默认 CPU / 必须 run_on_bpu”只在官方明确针对相应算子时可报告，不将其与 BPU 形状 FAIL 混同。

### 6.3 风险溯源沿用当前 GraphIR

对 VIOLATION 节点：

1. 输出 node ID 和原始名称、触发的 rule_id。
2. 用当前 `trace_node()` 返回真实 Tensor 边、直接前后继、可达模型输出和每输出一条最短代表路径。
3. 保持算法上限：不枚举指数数量的全部分支路径。
4. 下游影响仅解释为**数据依赖相关**，不把所有下游节点标红或说它们同样违规。

对大量 `NEEDS_VERIFICATION` 节点：不要默认对每一个跑全图追踪。仅在用户 `inspect/trace` 指定，或报告中选取最关键的 K 个代表问题时按需计算。

### 6.4 修改建议只能指向训练/导出工程

每个实际 FAIL / 重要 UNKNOWN 建议必须分出两类：

**A. 导出层问题**（如 axis、ONNX 输出布局、动态输入参数、错误导出选项）：说明优先检查训练代码中的 `torch.onnx.export` / 导出脚本 / 后处理导出逻辑；若只需要修正导出，重新导出并验证输入输出语义。

**B. 网络架构问题**（如确实不支持的算子使用方式）：指出需要在 PyTorch 的 `nn.Module.forward()`、模型 YAML 配置或相应模块源码中修改，再重新训练/微调（是否必需须根据修改范围说明）、重新导出和比较任务指标。

建议模板：

```text
发现：node_id / 原始节点名称 / op / input-output Tensor
证据：rule_id + observed + expected + 官方 URL/版本/章节
根因：明确违反、推断不足、工具链待验证，三选一
建议回源位置：原模型定义/forward/导出脚本（如果没有真实路径，不臆造具体 .py 文件）
可能修改：结构/导出流程中的备选方案；不作为已执行操作
重新训练要求：无需 / 可能需要微调 / 架构变化需重训，说明依据
复验：导出新 ONNX → onnx.checker → 本工具 analyze → 输出接口与精度/任务指标验证
```

**严禁输出本项目内的 ONNX graph surgeon / onnxoptimizer 原地修改脚本。** 本版本任何查询或报告不得自动执行这些操作。

---
## 7. 报告生成规则 V1.2

> **这不是只改文案。Codex 必须修改 `report.py`（必要时拆分 reporting_sections），修改相应 `analysis.json` 汇总与 CLI 输出，并添加精确测试。** 同时写入 `docs/REPORT_POLICY_V1_2.md` 作为长期规范。

### 7.1 当前问题与新的总原则

当前 `report.md` 会对 47 个正常 Conv 重复生成“无需修改”。多算子扩展后如果对每个通过的 Mul/Sigmoid 都这么做，报告会变得不可读。

新策略：**异常优先、未知可定位、正常汇总、覆盖透明、来源可追溯、建议回到训练/导出工程**。

不做分数、星级、风险排名；“Top 10 大 Tensor”是资源列表排序，不是风险评分。

### 7.2 `report.md` 必须包含的章节（建议顺序）

**第 0 节：执行范围与核心结论（报告顶部）**

- 模型标识（最好使用文件 basename 与 SHA256，绝对机器路径放“执行环境”或 JSON，避免主报告不必要的路径暴露）。
- `model_check`/ONNX checker 状态，opset，节点总数，Tensor 总数，输入输出。
- BPU 静态检查“覆盖节点数 / 未覆盖节点数 / 需要验证节点数 / 违规节点数”。
- 显眼声明“静态规则检查不是工具链 CPU/BPU 分配，也不意味着模型不能运行”。
- 源版本及 toolchain 未验证状态。

**第 1 节：算子覆盖与结论矩阵**

必须按 `op_type` 列出：

| 算子 | 模型中数量 | 已检查无违规 | 违反硬约束 | 待验证 | 未覆盖 | 规则来源版本 |
|---|---:|---:|---:|---:|---:|---|
| Conv | ... | ... | ... | ... | ... | ... |
| Mul | ... | ... | ... | ... | ... | ... |
| Sigmoid | ... | ... | ... | ... | ... | ... |
| Add | ... | ... | ... | ... | ... | ... |
| Concat | ... | ... | ... | ... | ... | ... |
| Slice | ... | ... | ... | ... | ... | ... |
| Gemm | ... | ... | ... | ... | ... | ... |
| 其它 | ... | ... | ... | ... | ... | ... |

**总数守恒**：四状态各列之和按节点汇总必须等于模型 node_count；按算子所有行之和也必须相等。兼容部分覆盖也要如实反映：某个 Gemm 运行了一条形状检查但关键转换条件 UNKNOWN，应在“待验证”而非“无违规”列。

**第 2 节：确定违反 BPU 约束的节点**

仅展示 `VIOLATION`：

- node_id + 原始节点名 + op_type；输入输出 Tensor（必要时 Shape）；
- rule ID、实际值、允许值、为什么失败；
- 精确官方 URL + 文档版本 + 章节；
- 到输出的相关路径简述；
- 回源修改位置建议（训练模型还是导出流程）。

默认最多展示前 50 个（按原图顺序，非凭空风险评分），余下数量明确写“其余 N 个请用 CLI/JSON 查看”。**全部违规节点仍必须完整保存在 analysis.json**。

没有违规时输出一行：`已收录且实际执行的 X5 BPU 约束未发现确定违规；尚有未覆盖/待验证项。` 不要出现“全部 BPU 兼容”。

**第 3 节：需要工具链或更多元信息确认的节点**

按 `op_type + reason_code` 分组统计，给出具有代表性的前 10～20 个节点 ID 及缺失条件，剩余数提示 CLI 查看。优先可定位的原因：广播高维合并、符号化 Shape、常量是否可覆盖、Slice 动态参数、Gemm→Conv 布局、文档版本冲突。

`UNKNOWN` 是诊断结果，不是代码失败；不应让几十个相同来源警告挤满整份报告。不能默认把待验证节点都追踪全图。

**第 4 节：重要 Tensor 资源**

沿用 V1.1：

- 真实输入和模型输出的 Shape/dtype/raw bytes。
- 前 10 个**已知大小**的中间 Tensor，包括 producer、消费者数量。
- 未知资源的 Tensor 数量及按原因汇总。
- INT8 仅展示“假设原始载荷”；绝对不说实际 BPU/DDR/峰值。
- 已知中间总大小如果有未知成员标为 `PARTIAL`，不能直接称作全图资源总量。

**第 5 节：结构优化候选**

沿用现有 V1.1 五个 pattern：只在实际存在候选时展开各条。如果为 0，一行记录即可。不得修改 ONNX；一旦建议影响网络结构，必须说明回到训练/导出工程。

**第 6 节：建议的后续行动（真正有区分度的 Codex Skill）**

按实际结果整理 0～N 个可执行动作：

- A：已确认 BPU 约束违规 → 在源模型/导出工程修改；必要时重训/微调、重新导出。
- B：关键条件未证实 → 补什么信息/等到哪个工具链阶段验证。
- C：未覆盖的算子 → 下次应补什么规则，为什么（基于节点数或路径）。
- D：静态检查无确定问题 → 不制造“必须优化”建议。

任何建议都有 `node_id/rule_id` 或“无节点级证据，仅提出后续规则覆盖计划”的区别。**不生成自动 ONNX 改写脚本**。

**第 7 节：规则来源、未覆盖与限制**

仅列本次**实际使用**的官方文档/版本，去重后列出；每条违规结果中仍有独立来源。列出未覆盖算子和关键例外（源码/编译器/量化条件未知）。

**第 8 节：查询方法和复现元数据**

给出当前仓库的 `nodes --status`、`inspect`、`trace`、`tensors`、`candidate`、`rules validate` 的正确命令模板。根据 V1.2 CLI 修改更新帮助文字；旧命令仍可运行。

### 7.3 JSON 的完整性与 Markdown 的简洁性

`analysis.json` 是机器事实库：**逐节点、逐规则完整保留 PASS/FAIL/UNKNOWN/NOT_APPLICABLE**，包括新增六类规则的元信息，不对 JSON 隐藏“正常节点”。

`report.md` 是人阅读的摘要：不能为了省字数删掉 JSON 里的可追溯证据；不能为 47 个正常 Conv、51 个正常 Mul 等重复冗余提示。需要按节点详情时使用 CLI 而不是默认全文展开。

报告输出要稳定：相同模型 + 相同规则包，节点顺序、规则顺序、结果摘要固定；不要把 AI 自由生成文本混入“事实段落”。Codex 的自由建议必须独立成“分析建议”，关联机器证据，并标明仅是建议。

### 7.4 防止报告错误与冗余

新增以下必测规则：

- 没有 FAIL 不能显示“完全兼容 RDK X5”。
- 没有已执行规则不能显示 `NO_VIOLATION_FOUND`。
- 规则不适用不算通过；不能把 UNKNOWN 变成 FAIL。
- 日志/报告展示官方文档链接时，禁止模型原始节点名造成 Markdown/终端控制字符注入。复用现有 `text_utils.py` 的文本安全逻辑。
- 旧 Conv 模型的报告仍能清楚显示所有确定异常节点，而不是只显示六类新增算子。
- 真实模型的 `report.md` 不能出现几十行相同“无需修改”。
- 报告中的“覆盖”是规则实现范围，不是 BPU 实际运行比例，不输出风险分数。

---

## 8. `.gitignore` 与历史报告清理

### 8.1 仓库现状与目标

当前 `.gitignore` 已经**不再包含 `reports/`**。上次 GitHub 提交把 `reports/yolo26_lane_robot-20261009-181829/` 内的大量 `.stdout/.stderr`、`analysis.json`、`before_fix` 调试文件一起跟踪了。

本轮必须恢复：**GitHub 默认不跟踪所有本地分析运行产物，但精选、可复现的简洁测试摘要应该保留在 `docs/validations/`**。

避免以后每测一个 ONNX 就提交几百个日志文件。

### 8.2 建议 `.gitignore` 内容（保留已有有效规则，按实际仓库补充）

```gitignore
# Python / environments
.venv/
__pycache__/
*.pyc
*.egg-info/
.pytest_cache/
.mypy_cache/
.ruff_cache/
build/
dist/

# Generated diagnostic sessions (never track by default)
reports/

# Local model assets; tiny generated test fixtures are an explicit exception
*.onnx
!examples/*.onnx
*.onnx.data

# Local-only secrets and IDE noise
.env
.env.*
!.env.example
.vscode/*.log
```

注意：`!examples/*.onnx` 要继续生效，以免误删原项目自带的微型 fixture。不要无差别忽略所有 `.md`、`.json` 或 `tests/`，否则会失去重要规则、文档和测试。

如果仓库已有其它必要忽略项，合并保留；不要为了照抄而删除已有工作区设置。

### 8.3 已跟踪报告必须解除索引跟踪（不删除本地）

**先检查** `git status --short` 和 `git ls-files reports/`，不得覆盖用户尚未提交的修改。

在确认当前大量报告都是生成产物、没有用户需要保留的独立手写内容以后，再运行：

```bash
# 只更新 Git 索引；--cached 不删除本地磁盘中的文件
git rm -r --cached -- reports/

# 验证忽略规则生效
git check-ignore -v reports/example-run/analysis.json

git status --short
```

如果报告目录存在尚未备份的用户原创手写文档，先将其移到 `docs/validations/` 或征求用户意见，不要无条件全部解除跟踪。命令产生的“已暂存删除”是**Git 索引变更**，不是本地报告丢失。

**不要执行** `rm -rf reports`、`git clean -fdx`、`git reset --hard`、`git filter-repo`、`git push --force`。历史 commit 依旧保留旧文件（不在本次范围做历史重写），只是从新的 main 快照中移除生成文件。

### 8.4 精选验证摘要的保留方式

建立 `docs/validations/Yolo26_V1_2_Summary.md`，或者等价简洁文件：

- 模型文件只写 basename 和 SHA256，不把本机完整绝对路径当成对外主要标识。
- 当前提交 SHA、包版本、规则包版本、ONNX opset、主要算子节点数。
- 真实模型多算子覆盖矩阵，以及 `VIOLATION / NEEDS_VERIFICATION / NOT_COVERED` 汇总。
- 重要未知条件、源头问题和可复现命令（模型路径使用 `<MODEL_PATH>` 占位符）。
- 测试结果和关键证据目录说明（本地 logs 位于被忽略的 `reports/`）。
- 不复制原始权重、不复制全量 JSON、不把数百条 stdout/stderr 提交进 Git。

如果尚未成功跑真实 ONNX，则不能伪造该精选摘要；先保留 `docs/validations/` 的规范说明，最终验收实际完成后再填事实。

### 8.5 `.gitignore` 验收

- `git check-ignore` 能识别 `reports/new-run/analysis.json`。
- 微型 `examples/*.onnx` 继续可跟踪。
- 新运行一次 `analyze`，`git status --short` 不会显示 `reports/new-run/analysis.json` 的新增未跟踪文件。
- `git ls-files reports/` 在清理索引后为空；本地 `reports/` 文件仍然存在。
- 没有改动或删除用户指定的真实 ONNX 源文件。
- 不自动 commit/push，最后向用户汇报 Git 的暂存变更，留给用户决定何时提交。

---

## 9. CLI、JSON 协议、Skill 和文档

### 9.1 CLI 保持既有命令

已有命令：

```bash
python -m rdkx5_doctor analyze --model /path/to/model.onnx --out reports/my-model
python -m rdkx5_doctor rules validate
python -m rdkx5_doctor nodes --analysis reports/my-model/analysis.json --status VIOLATION
python -m rdkx5_doctor inspect --analysis reports/my-model/analysis.json --node main/node_000000
python -m rdkx5_doctor trace --analysis reports/my-model/analysis.json --node main/node_000000
python -m rdkx5_doctor tensors --analysis reports/my-model/analysis.json --kind intermediate --sort bytes --limit 10
python -m rdkx5_doctor candidates --analysis reports/my-model/analysis.json
```

新增（推荐，可按实际架构提供同等功能）：

- `rules validate` 输出总规则数、每个 op 的规则数、来源版本及校验状态。
- `rules list --operator Mul`：列出该算子已经收录的 BPU 规则和文档来源；全部从 YAML 读取。
- `nodes --status NEEDS_VERIFICATION`：多算子检查后可靠过滤，不限 Conv。
- `nodes --search Mul`、`inspect --node <id>`：展示 Mul 的真实输入形状、广播类别与逐规则证据。
- 可选 `--report-detail summary|full`，但**默认必须为简洁报告**；即使加此选项也不能引入 UI。

本轮不是增加命令数量比赛；不要无故增加与旧命令同义的子命令。返回码维持 `0` 成功、`2` 输入/规则/文件/查询失败；**存在 BPU 规则违规但分析成功**时仍返回 `0`，除非另设计一个明确的 CI-only 严格模式并单独文档化。

### 9.2 JSON schema

建议版本：`analysis.json` 顶层 `schema_version="1.2"`；包版本 `0.3.0`；总规则包版本 `0.2.0`。新字段推荐：

```json
{
  "schema_version": "1.2",
  "ruleset": {
    "id": "x5-bayes-e-onnx-multiop",
    "version": "0.2.0",
    "toolchain_version": "unverified",
    "toolchain_verified": false,
    "operator_rule_versions": {
      "Conv": "0.1.0",
      "Add": "0.2.0",
      "Mul": "0.2.0"
    }
  },
  "summary": {
    "all_status_counts": {},
    "conv_status_counts": {},
    "operator_coverage": [],
    "covered_node_count": 0,
    "uncovered_node_count": 0
  },
  "diagnostics": [],
  "resource_analysis": {},
  "optimization_candidates": {}
}
```

此处 `0` 是协议形状占位符，**不是实际模型数值**。必须保留真实旧键、Tensor 和已实现候选字段；`operator_rule_versions` 对所有已启用 op 完整输出。每个节点应标明规则覆盖类型：`AUTO_CHECKED / PARTIAL_OR_CONDITIONAL / NOT_COVERED`，但不是新的风险分数。

新 schema 文档必须说明：

- 旧 `1.0/1.1` 查询兼容性。
- 老 Conv 字段行为是否保持一致。
- `operator_coverage` 与 `all_status_counts` 的汇总方式及其守恒测试。
- 旧 JSON 没有 `1.2` 来源字段时如何只读展示和标明缺失，不得猜补。

### 9.3 Codex Skill 的职责（不是把报告重述一遍）

在 `.agents/skills/rdk-x5-onnx-doctor/SKILL.md` 保留原触发条件，新增：

1. 首先运行模型 analyze，检查规则包版本与是否包含目标算子。
2. 用多算子覆盖矩阵选择最需要深入检查的类别（确定违规/关键 UNKNOWN/重要输出路径）。
3. 对选中的节点使用 `inspect`/`trace`，输出真实原因、涉及的 Tensor 和官方依据；若没有确定违规，严禁捏造一个。
4. 需要变更网络时只提供**回到训练/导出工程**的建议，不使用 ONNX 编辑工具；是否重训由改变的计算语义决定。
5. 如果用户以后提供训练仓库路径，再进入对应源码搜索或定位；本轮不得自行去用户其它路径搜索/修改训练源码。
6. 结论附带“静态规则范围/工具链未验证”边界说明。
7. 明确要求普通运行报告与临时日志保存在 `.gitignore` 忽略的 `reports/` 下；精选验收材料放 `docs/validations/`。

推荐把变更较长的六算子规则说明放入 `references/`，让 `SKILL.md` 保持简短，避免把全部官方表格复制进 Skill 入口文件。

### 9.4 文档更新清单

- 更新 `README.md`：项目 V1.2 支持的七类算子、命令、状态含义、报告策略、已知限制。
- 更新 `AGENTS.md`：优先读本开发规范、禁止修改 ONNX、报告与 Git 忽略规则。
- 更新 `docs/ANALYSIS_SCHEMA_V1_2.md`、`docs/REPORT_POLICY_V1_2.md`。
- 新增 `references/x5_multiop_sources.md` 和 `references/x5_multiop_rule_review.md`。
- 更新 `docs/DEVELOPMENT_LOG.md`：各开发阶段实际执行的命令、结果、Bug、边界测试和真实模型核对。
- 不要修改历史 `docs/ANALYSIS_SCHEMA_V1_1.md` 来假装当时已经支持多算子；新版本使用新文档。

---
## 10. 测试用例矩阵与真实模型回归

### 10.1 开发前基线测试

在仓库根目录：

```bash
pwd
git status --short
.venv/bin/python -m rdkx5_doctor --help
.venv/bin/python -m rdkx5_doctor rules validate
env -u PYTHONPATH .venv/bin/pytest -q
```

首次没有 `.venv` 则按 README 创建；WSL/ROS 环境可能通过 `PYTHONPATH` 注入 `launch_testing`，因此测试前隔离 ROS 环境，**不要通过删除项目测试或把失败隐藏来“通过”**。

原仓库最后记录 `141 passed`；这只是历史基线，不表示此刻或新环境中必然通过。若现有工作树不干净，先记录用户修改，禁止 `git reset --hard`。

### 10.2 必须实现的自动化测试

| 测试组 | 最低测试覆盖 |
|---|---|
| Registry | 7 算子注册、规则文件缺失、重复 ID、版本冲突、非法 YAML、字段不存在、危险路径、越界软链、未知谓词 |
| Conv 回归 | 原 19 条规则与旧 fixture 结果不变、真实模型 47 Conv 结果与旧运行一致 |
| Sigmoid | rank 边界、未知 rank、来源链接、未验证 dtype 说明、自定义域 |
| Concat | axis 0/1/-rank/-1、缺失/越界 axis、batch 维证明、符号维度 |
| Slice | opset 11 固定/动态 starts-ends-axes-steps、非四维、有限常量解析、无编造限制 |
| Add/Mul | 单边/双边广播、同 shape、不同 rank、可证明不允许的合并、复杂 UNKNOWN、常量来源分类 |
| Gemm | alpha/beta/transA/transB、静态 M/K/N、未知转换布局、权重 Shape 缺失、混淆 ONNX 语义与 X5 硬件限制的反例 |
| 状态机 | 确定 FAIL + UNKNOWN 并存的优先级、仅 UNKNOWN、仅 NOT_APPLICABLE、无规则、正确 PASS |
| JSON schema | 1.0/1.1/1.2 兼容、每节点有且仅有诊断、规则/节点唯一 ID、来源版本完整、全状态数量守恒 |
| CLI | rules validate、rules list（如实现）、nodes/inspect/trace、tensors/candidates 历史兼容、返回码 |
| 报告 | 异常优先、未知按原因分组、正常节点不重复列建议、coverage 守恒、前 N 项截断提示 |
| 安全 | 原 ONNX SHA256 不变、报告路径不覆盖输入、未加载外部权重、不解释任意 YAML 代码 |
| Git hygiene | 新 reports 不受跟踪、已有用户源文件不删除、examples 小型 ONNX 仍可跟踪 |
| 安装 | 构建 wheel/sdist，脱离 repo 工作目录验证 package 内置 YAML 和 CLI 可用 |

**要求至少补充每个新算子一例可信 PASS、一例可信 FAIL（如果官方 BPU 约束可明确构造且 ONNX 模型合法）、一例真实 UNKNOWN；对“官方没有额外硬数值限制”的 Slice，不要伪造 FAIL，改测来源/元信息保守判定。**

每条可执行 BPU 规则必须有边界内、边界外的测试；如只有宽泛的能力陈述，使用 `review_only` 并验收不会产生虚假 FAIL。

### 10.3 六个算子最容易发生的误判（必须专测）

1. **X3/X5 混淆**：X3 的 Mul C≤2048 不能套到 X5；测试应断言加载的 X5 规则没有来源为 X3 的旧条款。
2. **BPU/CPU 列混淆**：如官方 CPU 列写 float-only，不得拿它作为 X5 BPU 对输入的硬限制。
3. **常量双输入识别**：initializer graph input 可覆盖时不能作为确定不可变常量；无 producer 不一定是常量。
4. **广播证据缺失**：ONNX 合法广播不保证 X5 的高维合并一定可实施；在不能证明时应 UNKNOWN。
5. **Concat -rank**：若 rank=4、axis=-4，归一化后是 axis 0，必须与 N 维约束联动。
6. **Slice 数据/参数区分**：starts/ends 的 INT64 元信息不是主 Slice 输出 dtype，更不等于量化输出是 INT64。
7. **Gemm→Conv**：不能直接把二维 Gemm 输入当作四维 Conv，不能仅因 `transB=1` 宣称 BPU 不支持。
8. **量化 dtype**：原始 float32 ONNX 不能被“支持 int16”的句子直接排除。
9. **无规则通过**：只靠“官方列 BPU 加速”而未执行任何具体条件的节点不应被当成全部条件通过。
10. **节点溯源**：真实 Tensor 连接不能被显示顺序误导；同名节点必须用稳定 ID 区分。

### 10.4 真实 YOLO26 回归

确认模型文件存在、记录原始 SHA256，再运行：

```bash
MODEL=/home/xhm/lianghua_ws/rdktoolchain/horizon_x5_open_explorer_v1.2.8/horizon_x5_open_explorer_v1.2.8-py310_20240926/samples/ai_toolchain/horizon_model_convert_sample/04_detection/15_tasknav/model/yolo26_lane_robot.onnx

.venv/bin/python -m rdkx5_doctor analyze \
  --model "$MODEL" \
  --out reports/yolo26-v1_2-test

.venv/bin/python -m rdkx5_doctor nodes \
  --analysis reports/yolo26-v1_2-test/analysis.json \
  --status VIOLATION

.venv/bin/python -m rdkx5_doctor nodes \
  --analysis reports/yolo26-v1_2-test/analysis.json \
  --status NEEDS_VERIFICATION

.venv/bin/python -m rdkx5_doctor tensors \
  --analysis reports/yolo26-v1_2-test/analysis.json \
  --kind output

.venv/bin/python -m rdkx5_doctor candidates \
  --analysis reports/yolo26-v1_2-test/analysis.json
```

**实测验收必须逐项验证：**

- 281 个节点全部解析；总数变化只能解释为输入文件确已变化，否则算回归 Bug。
- 原 47 个 Conv2D 的规则判断结果不被新引擎篡改。
- 51 Mul、38 Sigmoid、14 Add、11 Concat、8 Slice、12 Gemm 都能在 registry 中找到正确规则集和结果；若其中某个检查不确定，显示 `NEEDS_VERIFICATION`，不得直接都标 PASS。
- 新六类共 134 个目标节点，加上 47 Conv，总共 **181 个候选纳入规则范围的节点**；这不是“181 个 BPU 已通过”。其余 100 个节点仍按覆盖范围如实显示（当前六类开发范围内不应冒称其它算子已检查）。
- 对每一种实际出现的 FAIL 和重要 UNKNOWN，选取代表性 node ID 执行 `inspect/trace`，核对 actual/expected/source 和真实 Tensor 边。
- V1.1 的输出 Tensor、部分中间资源已知值和优化候选结果不因规则升级意外变化。原记录：输出 `cls_logits` 144,256 B、`offset` 896 B；172 个中间未知 Shape 是已知限制，本轮**不承诺清零**。
- 重新分析前后源模型 SHA256 完全一致。
- `report.md` 确实采用新精简策略：包含七类覆盖矩阵，没有 47 条或更多“无需修改”模板行。
- 报告与 stdout/stderr 默认在被忽略的 reports 下；Git 不出现数百个新报告文件。

### 10.5 反向核对（不能只依赖自己生成的 JSON）

用独立 ONNX API 读取真实模型，验证每个算子数量与节点 ID/输入输出元信息；对复杂 Add/Mul 广播至少使用**与规则引擎独立的测试 oracle**或手工精确计算的 fixture。

如果出现新增违规，必须核实：官方来源版本、节点真实 Shape、是否适用、是否由于误把 CPU/X3 列当 X5、是否 ONNX 自身非法、是否未知转 PASS/FAIL。可追溯后才保留结论。

如果真实模型六类算子最终没有确定 FAIL，也完全允许。不得为了展示创新性强行制造规则违规。

## 11. 按阶段执行的 Codex 工作计划

> 阶段编号从 V1.1 后续接续。每阶段都要修改源码、运行相关测试、记录结果；不要先写最终报告而不实现代码。

### P12 — 环境、Git 和文档基线

- 阅读第 1 节指定的所有现有文件，检查 working tree。
- 检查真实 ONNX 和 `.venv`，执行 `--help`、`rules validate`、全量 pytest。
- 记录包、规则和 schema 版本；建立 `docs/DEVELOPMENT_LOG.md` V1.2 小节。
- 将官方 X5 六类算子原始条款/版本记录为审查草案；不要先大规模改代码。
- **阶段验收**：旧命令和旧模型回归成功；六算子官方来源列表已建立。

### P13 — RuleRegistry + schema + 规则可信度机制

- 使单算子 RuleSet 可多算子注册；兼容历史 `conv2d.yaml` 和已有测试。
- 建立字段、谓词和 YAML 路径白名单；拒绝失效/危险配置。
- 实现文档版本、适用 domain/opset 与规则检查状态。
- 同步 `rules validate` 显示多算子规则数；必要时提供 `rules list`。
- **阶段验收**：Registry 正/反测试全部通过，旧 Conv 检查无变化。

### P14 — 六算子参数提取与逐算子规则

按 Sigmoid、Concat、Slice、Add、Mul、Gemm 顺序增量开发：

1. 获取实测 GraphIR 字段与 ONNX schema 默认值。
2. 根据 X5 官方 BPU 列形成该 op 的 YAML + 来源条目。
3. 实现仅针对必要事实的 extractor/predicate，缺值返回 UNKNOWN。
4. 每个 op 做最小 valid/invalid/unknown fixture，重点检查版本与常量来源。
5. 对真实 YOLO 对应节点做子集分析，不到全量最终结果才发现所有节点被误判。

- **阶段验收**：六个算子都有明确覆盖类型、来源和可查询证据；不是六个空 YAML。

### P15 — `analysis.json 1.2` / CLI / 节点溯源

- 把所有算子的检查结果并入 `diagnostics`，保留完整数据和旧键。
- 新增 `operator_coverage`，按真实节点数计算汇总；四状态守恒。
- `nodes --status`、`inspect`、`trace` 能看到新增算子细节。
- 对 VIOLATION 继续有效追踪；大量 UNKNOWN 避免全图重复路径展开。
- **阶段验收**：历史 report 加载兼容，真实 281 节点可以逐节点查询。

### P16 — 报告生成策略重构

- 按第 7 节实现简洁 `report.md`，不再批量列出正常节点“无需修改”。
- 修正文档中“V1 仅 Conv”的固定描述，显示实际版本/范围。
- 保留所有规则证据于 JSON，Markdown 优先展示异常、待验证、覆盖矩阵。
- 新增 `docs/REPORT_POLICY_V1_2.md`、报告快照与汇总守恒测试。
- **阶段验收**：真实模型报告可读，差异可由测试确定；不漏任何确定违规节点计数。

### P17 — Git 报告产物卫生与 Skill 更新

- 检查 `.gitignore`，增加 `reports/`，保留示例 `.onnx` 的跟踪例外。
- 先检查用户文件，再用 `git rm --cached` 解除历史 reports 产物跟踪，确保本地文件存在。
- 创建必要且简洁的 `docs/validations/` 摘要，不放庞大的 `analysis.json` / 终端日志。
- 更新 `README.md`、`AGENTS.md`、Skill 的多算子分析及回源修改准则。
- **阶段验收**：`git status` 不出现运行产物、新报告被忽略、历史本地文件未被删除。

### P18 — 真实 YOLO 全面验收与打包

- 对原 ONNX 测试，保留原始 SHA256。
- 完整测试六类算子覆盖、未知/违规/正常解释、输出资源与候选不回归。
- 执行 pytest 全量回归与 `python -m build`，新 venv 安装 wheel 从仓库外运行。
- 更新 `docs/DEVELOPMENT_LOG.md`、精选 `docs/validations/Yolo26_V1_2_Summary.md`。
- 检查 diff，无原 ONNX 修改、无重复 YAML 规则副本、无非必要网页功能、无静默失败。
- **阶段验收**：达到第 12 节全部条件，给出最终 CLI 实际运行结果。

---

## 12. 验收标准与最终输出

### 12.1 必须交付的代码能力

- [ ] 不破坏 V1/V1.1 的 Conv2D 检查、风险路径、Tensor 资源和优化候选。
- [ ] 支持从 manifest 发现多算子 YAML，来源、版本、作用范围可检查。
- [ ] 完成六种算子事实提取 + 可信的、在官方文档可找到出处的 BPU 规则。
- [ ] 每条规则的 PASS/FAIL/UNKNOWN 都可追溯 node、Tensor、字段和值。
- [ ] UNKNOWN/不覆盖不会被默默算 PASS；BPU 约束与 ONNX 自身合法性分离。
- [ ] `analysis.json 1.2` 保存所有机器事实，旧 schema 查询兼容。
- [ ] `report.md` 覆盖矩阵准确、没有正常节点模板刷屏，建议只指向训练/导出工程。
- [ ] `.gitignore` 忽略 reports，之前跟踪的生成报告从索引安全移除且本地不删除。
- [ ] `SKILL.md` 更新为多算子智能诊断工作流，不写自动 ONNX 改写流程。
- [ ] 添加版本化的来源审查文档、协议文档和回归测试。

### 12.2 必须交付的测试证据

- [ ] 旧 V1/V1.1 所有测试继续通过；新增测试也全部通过，无故意 skip。
- [ ] 针对六种算子各有官方规则、正常例、未知例和适当的违规边界例。
- [ ] 对真实 `yolo26_lane_robot.onnx` 成功分析，281 个节点数量不被虚假改变。
- [ ] 六类 134 个目标节点均有规则入口；不能说所有节点都已确认可用 BPU。
- [ ] 原 47 个 Conv 的旧规则评估与原结果一致；Tensor 资源功能保持正常。
- [ ] 模型 SHA256 不变，不运行 Docker/量化/板端，不自动修改模型。
- [ ] Wheel / sdist 可构建，独立目录安装后内置 7 算子规则可加载。
- [ ] 只有精选摘要进入 Git，原始 `reports/` 继续留在本机。

### 12.3 最终回答用户时的格式

Codex 完成开发后，用终端事实写简洁总结，至少包括：

1. **代码提交范围**：新增/修改了哪些文件，具体实现何种规则和提取器。
2. **官方证据**：每个算子的来源、规则数量、哪些自动检查、哪些保守 UNKNOWN，文档是否冲突。
3. **真实 YOLO 结果**：每 op 节点数、VIOLATION/NO_VIOLATION_FOUND/NEEDS_VERIFICATION/NOT_COVERED 数量，列出有证据的异常节点。
4. **报告改进**：精简后的报告章节、仍保留的 JSON 详细证据、相关测试。
5. **Git 状态**：`.gitignore` 是否生效，之前跟踪的报告是否只从索引移除，是否仍在本地；展示 `git status --short` 中的暂存变更说明。
6. **测试结果**：pytest 数量、编译/打包、真实模型校验、SHA256。
7. **限制和后续**：其他算子未覆盖；工具链未验证；任何修改需要回到训练/导出工程完成。

### 12.4 可以直接粘贴到 Codex 的启动指令

> 阅读仓库根目录的 `RDK_X5_ONNX_Doctor_V1_2_Multi_Op_BPU_Development_Spec.md`，按 P12–P18 顺序在本仓库**实际完成** V1.2 开发与测试，不要只输出计划。
>
> 本次只做 Mul、Sigmoid、Add、Concat、Slice、Gemm 六种算子的 RDK X5 BPU 官方约束检查，并升级通用 YAML Registry；同步重构报告生成策略，恢复 reports/ 忽略规则，安全停止跟踪历史生成日志。保持 V1/V1.1 功能，原 Conv 检查不变。每条判断必须带真实节点和官方文档版本来源；复杂或不确定条件返回 UNKNOWN，严禁猜测。
>
> 严禁自动编辑 ONNX。任何修改建议必须回到 PyTorch 训练/导出工程重新导出，必要时重训或微调。本轮不调用 Docker/hb_mapper/量化/板端，不做评分或可视化。测试时用真实 `yolo26_lane_robot.onnx`，保存完整本地报告到被忽略的 reports/，只把精选摘要放进 docs/validations/。完成后报告真实命令、测试结果、规则数量、覆盖矩阵、发现的确定违规和待验证事项；不自动 git commit/push。

---

## 附录 A：实现时必须遵循的来源优先级

1. 官方 RDK X5 **ONNX BPU 支持约束**栏（记录页面时间、文档版本和算子行）。
2. X5 用户手册其它版本作交叉对照；若冲突，记录为条件/未知而不是悄然覆盖。
3. ONNX 对应 opset 的 `onnx.defs.get_schema(...)` 与 ONNX 官方 operator 文档只负责参数语义、默认值和广播合法性。
4. 用户本次没有要求的 Docker / `hb_mapper checker` / 量化验证留待独立分支；此版本始终标记工具链未验证。

## 附录 B：正式开发前的自查问题

- 我是否把 X3 的约束误写成 X5？
- 我是否把 CPU 支持列误写成 BPU 支持列？
- 当前规则是否真的可以从 ONNX 静态信息判断？不能的话是否返回 UNKNOWN？
- 官方“支持 int16”是否只是能力描述而非原始 ONNX 的硬 dtype 要求？
- Gemm 的实际转换 Conv 参数是否有证据，还是我凭空代入二维输入？
- 高维广播是否有完备证明，还是只知道 ONNX 广播合法？
- 没有规则的节点是否被误当成“检查通过”？
- 报告是否只突出异常与未知，同时给出全量 JSON 查询路径？
- Git 的删除是否只是索引删除，用户本机历史 reports 是否仍完整？
- 我的优化建议是否回到训练/导出工程，而不是改 ONNX？

**全部回答清楚后，才可以宣称 V1.2 已完成。**
