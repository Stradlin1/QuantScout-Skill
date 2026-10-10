# QuantScout-Skill

**面向地平线 RDK X5（Bayes-e）的 ONNX 量化前静态诊断 Agent Skill**

本项目是 AIGC 通识课程的个人结课作业，来源于 RDK X5 模型部署中的实际需求：在进入 OpenExplorer 量化工具链前，先检查 ONNX 模型结构、版本、部分 BPU 算子约束和 Tensor 信息，并把检查结果转化为有证据的中文说明。

**QuantScout 不是一键量化器。** 它由可独立运行的 Python 静态诊断器，以及负责理解自然语言、选择工具和组织证据的 Agent Skill 组成。它不修改 ONNX，也不运行 Docker、`hb_mapper`、量化、编译或板端推理。

> **当前版本快照（2026-10-10）：** Skill **V1.5-S1.1** · Python 包 `rdkx5-onnx-doctor` **0.5.0** · Analysis Schema **1.3** · 目标平台 **RDK X5 / Bayes-e**。代码已实现三种 Agent 事实总结模式；**V1.5-S2 Tensor 分类增强和 V1.5-S3 量化敏感结构提示尚未实现**。

[快速上手](docs/GETTING_STARTED.md) · [项目介绍与设计](docs/PROJECT_OVERVIEW.md) · [验证现状与限制](docs/VALIDATION_STATUS.md) · [完整文档导航](docs/README.md) · [Skill 入口](.agents/skills/rdk-x5-onnx-doctor/SKILL.md)

## 1. 项目能做什么？

| 已实现能力 | 具体内容 | 必须保留的边界 |
| --- | --- | --- |
| ONNX 结构读取 | 格式校验、主图节点、OpSet、输入输出、计算图连接 | ONNX 合法不等于 X5 可编译 |
| 当前 Profile 预检 | 比较标准 ONNX 主域 OpSet 与用户指定配置，输出 MATCH / MISMATCH / UNKNOWN | 仅比较当前配置条件，不证明硬件兼容 |
| BPU 静态规则检查 | 对已覆盖算子逐节点记录实际值、允许条件、规则 ID 和来源 | 未覆盖算子不是 PASS，也不是“不支持” |
| Shape 推断及证据 | 追踪已知/未知维度、有限静态证明、冲突与阻塞项 | 不猜测未知 Shape |
| Tensor 理论载荷 | 根据可确认的 Shape 与 dtype 统计原始字节数 | 不等于实际 DDR、SRAM 或峰值内存 |
| 节点查询与图追踪 | 使用 nodes、inspect、trace、shapes、tensors 等命令定位证据 | 图上可达不等于异常已实际传播 |
| 结构优化候选 | 标记需要人工审查的有限结构模式 | 不自动改图、删除节点或预测收益 |
| 官方知识查证 | Agent 按需检索官方资料或已固定的离线手册，并区分来源状态 | 查到资料不等于已运行编译器 |
| **Agent 事实总结** | 从已有 JSON 提取事实，按自然语言选择 overview / anomalies / io 模式，生成 `summary.md` | **只总结，不分析、不提供建议** |

现有规则库覆盖 **14 类算子、66 条检查或复核条目**。这不是 ONNX 全算子覆盖率，更不代表已验证模型在 BPU 上的实际执行情况。

### 两种不同的输出方式

**完整诊断：** Agent 按 `SKILL.md` 选择 `analyze`、`preflight`、`inspect`、`trace` 或必要的官方查询，生成结构化证据并解释已确认的发现与未知条件。

**事实摘要（V1.5-S1.1）：** Agent 根据自然语言选择摘要模式。Python 提取并校验结构化事实；Agent 在受限句式中选择事实、标题和顺序；发布器重新核验数值、引用和来源哈希后保存 Markdown。整个流程不需要额外的 LLM SDK、API Key 或单独的生成服务，使用当前 Agent 本身即可。

```text
用户自然语言
    │
    ├── 完整诊断 → Agent 编排 Python/资料查询 → analysis.json + report.md
    │
    └── 仅总结 → summary-facts → Agent 组织草稿 → summary-publish → summary.md
                            │                                │
                      可信事实和来源                   重算校验、独占新建
```

普通 `summary` CLI 仍提供**纯 Python 的确定性简版摘要**；`--detailed` 可输出旧版七段详细模板。**CLI 自动排版不应被称为 AI 撰写。**

## 2. 快速体验（Ubuntu / WSL 终端）

需要 Python **3.10+**、Git 和可创建虚拟环境的 Linux/WSL 工作区。静态检查无需 GPU、Docker、开发板或 OpenExplorer。

```bash
git clone https://github.com/Stradlin1/QuantScout-Skill.git
cd QuantScout-Skill

python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m rdkx5_doctor --help
.venv/bin/python -m rdkx5_doctor rules validate

# 运行一次静态分析，使用全新的输出目录
.venv/bin/python -m rdkx5_doctor analyze \
  --model examples/demo.onnx --out reports/first-check

# 不依赖 Agent：生成纯 Python 简版事实摘要
.venv/bin/python -m rdkx5_doctor summary \
  --analysis reports/first-check/analysis.json
```

`reports/` 默认被 Git 忽略。摘要和事实包采用独占新建策略；如果目标文件已存在，不会自动覆盖。重复演示请使用新的报告目录，或者在 Agent 工作流中由用户指定不同的新摘要文件名。

### 在 VS Code + Codex 中使用

在 **WSL Ubuntu 工作区**打开仓库，让 Codex 能读取完整的 `.agents/skills/rdk-x5-onnx-doctor/` 以及 Python 工具，再以自然语言提出任务：

> 使用当前仓库的 QuantScout Skill 检查 `examples/demo.onnx` 的 RDK X5 量化前静态约束，把报告写入一个新的 `reports/` 子目录。不要修改 ONNX，也不要执行量化。

生成 `analysis.json` 后，可继续提出：

| 自然语言请求 | Skill 模式 | 输出重点 |
| --- | --- | --- |
| “简短总结检查结果，只需要事实，不要分析和建议。” | `overview` | 模型、四状态、代表 FAIL、检查边界 |
| “只列出异常，不要分析和建议。” | `anomalies` | 已确定 FAIL、待验证、未覆盖分别陈列 |
| “只看模型输入和输出。” | `io` | Graph I/O 的名称、Shape、dtype 与省略数量 |

Agent 路径使用 `summary-facts` → **Agent 撰写受限草稿** → `summary-publish`。已有 `analysis.json` 时不重复分析模型。手动操作、输出路径和更多命令见 [快速上手](docs/GETTING_STARTED.md) 与 [摘要契约](.agents/skills/rdk-x5-onnx-doctor/references/summary_only_contract.md)。

### 当前 OpenExplorer 配置

开发者实际使用 **OpenExplorer v1.2.8**。仓库提供的 [工具链 Profile](.agents/skills/rdk-x5-onnx-doctor/references/toolchain_profile_opset11.yaml) 要求标准 ONNX 主域 **OpSet 11**，这是**此配置的版本条件**，不能推广为“所有 RDK X5 工具链仅支持 OpSet 11”。

```bash
.venv/bin/python -m rdkx5_doctor preflight \
  --analysis reports/first-check/analysis.json \
  --profile .agents/skills/rdk-x5-onnx-doctor/references/toolchain_profile_opset11.yaml \
  --json > reports/first-check/preflight.json
```

`MATCH` 仅表示与该 Profile 的版本条件相符，不代表已通过 `hb_mapper` 编译。

## 3. 公开演示与结果解读

仓库的 [示例分析](examples/demo-report/analysis.json) 已更新到 **Analysis Schema 1.3**；[公开简要摘要](examples/demo-report/summary.md) 来自独立 Codex Agent 会话，配有 [事实包](examples/demo-report/summary_facts.json) 和 [核对说明](docs/validations/V1_5_S1_1_Public_Demo.md)。旧 Schema 1.0 快照保存在 `examples/legacy-demo-report/`，仅用于历史兼容测试。

示例 `demo.onnx` 含 **5 个已解析主图节点**，最终节点状态分布为：

| 状态 | 数量 | 含义 |
| --- | ---: | --- |
| `NO_VIOLATION_FOUND` | 2 | 已执行规则未发现确定冲突，不等于完整兼容 |
| `VIOLATION` | 1 | 存在确定静态 FAIL 的节点 |
| `NEEDS_VERIFICATION` | 0 | 本示例没有最终落入此状态的节点 |
| `NOT_COVERED` | 2 | 当前规则没有覆盖这些节点 |

其中 `main/node_000000`（Conv，原名 `oversized_kernel`）触发规则 `X5-CONV2D-KERNEL-H`：卷积核高度 **32**，超出当前规则的 **1～31** 范围。**这只是静态规则冲突的演示，不是实际编译失败记录。**

**统计范围：** 节点和诊断计数只包含已解析的**主图节点**，不递归展开嵌套子图；ONNX checker 通过也不代表子图算子已经完成 BPU 规则检查。Graph inputs 在 ONNX 中可能包含 initializer，不应全部解释为图像等外部数据输入。

## 4. 测试与已知限制

截至 **2026-10-10**，以代码提交 [`60b81f5`](https://github.com/Stradlin1/QuantScout-Skill/commit/60b81f55edce40c461e44e43e41af727dfa7990b) 为核验基线：

- 本地开发记录：**528 个 pytest 通过、0 失败**；这是开发日志报告的数据。
- [GitHub Actions 第 1 次运行](https://github.com/Stradlin1/QuantScout-Skill/actions/runs/38021157135)：Python **3.10 / 3.12** 两个作业均已成功完成，包含规则校验、离线测试与构建。
- 独立 Codex 会话：累计 **11 次，10 PASS / 1 FAIL**。原 E10 曾绕开发布器替换已有摘要；修正契约后，新会话 E10_retry 通过。**不能因此宣称 Agent 在所有情况下都无法覆盖文件。**

事实发布器校验来源 SHA、声明和限定的事实占位符；**这不是对任意中文进行形式化语义验证**。Agent 允许选择事实、排列顺序、标题和有限连接短语，但不能任意扩写分析性结论。

本项目未运行量化、编译和板端实测；不输出真实 FPS、量化精度或 CPU/BPU 放置保证。只有 Codex 完成了本项目记录的真实 Agent 端到端验收；其他 Agent 的可移植性尚待实测。完整证据与边界见 [验证现状](docs/VALIDATION_STATUS.md)。

## 5. 跨 Agent 与项目结构

仓库遵循 `SKILL.md` + `references/` + `scripts/` 的 Agent Skills 组织方式：

```text
QuantScout-Skill/
├── .agents/skills/rdk-x5-onnx-doctor/
│   ├── SKILL.md                 # Agent 意图路由与执行约束
│   ├── references/             # 摘要、诊断、官方查询契约
│   └── scripts/
├── src/rdkx5_doctor/           # 确定性 Python CLI、规则和报告
├── references/                 # 审核资料与离线官方手册
├── examples/                   # 公开模型与报告
├── docs/                       # 开发规范、使用与验收文档
├── tests/                      # pytest
├── .github/workflows/ci.yml    # Python 3.10 / 3.12
└── pyproject.toml
```

主要开发与验证环境是 **WSL Ubuntu / Ubuntu + Codex**。Cursor、GitHub Copilot、Gemini CLI、OpenCode 等是否能直接发现本项目 Skill，取决于其版本、工作区和执行权限，**目前没有跨客户端完整验证记录**。Claude Code 还需适配其 Skill 目录及相对路径。迁移时不能只复制 `SKILL.md`，必须同时提供 Python 包、规则及被引用的资料。

**名称说明：** GitHub 仓库名为 `QuantScout-Skill`；Skill 内部名称为 `rdk-x5-onnx-doctor`；Python 包和 CLI 模块为 `rdkx5_doctor`，这些接口没有因仓库改名而改变。

## 6. 后续开发计划（尚未实现）

| 计划阶段 | 目标 | 边界 |
| --- | --- | --- |
| **V1.5-S2** | 增强 Tensor 来源/用途分类，区分权重、特征、Shape 参数及派生数据 | 不把理论载荷冒充实际硬件内存 |
| **V1.5-S3** | 识别值得审查的量化敏感结构候选 | 不预测量化误差、编译结果或 FPS |
| **独立评测** | 比较同条件下普通 Prompt、CLI 和 Agent Skill 的任务完成与事实忠实度 | **目前尚无已发布的正式对照实验结果** |

详细文档请从 [docs/README.md](docs/README.md) 查阅。历史规范与开发日志保留各自的时间点，不应把旧日志中的 `NOT_RUN` 当成当前 CI 状态。

---

QuantScout-Skill 的目标是让 RDK X5 量化前的 ONNX 静态检查更容易执行、事实更容易复核、Agent 的解释更可追溯，而不是替代地平线实际工具链验证。
