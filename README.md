# QuantScout-Skill

**AIGC 通识课程个人结课作业｜面向 RDK X5 的 ONNX 量化前诊断 Agent Skill**

QuantScout-Skill 是我结合 **地平线 RDK X5 模型部署实践**开发的 AI Agent Skill。它把原本需要人工逐项完成的 ONNX 结构检查、BPU 静态约束核对、节点定位和报告整理，组织为一套可以通过自然语言调用的诊断流程。

项目采用 **“Agent 智能编排 + Python 确定性分析 + 可追溯规则证据”** 的设计：Agent 理解检查需求、选择工具并组织中文结果；Python 负责模型解析、规则计算和事实校验。它不仅能够发现部分可确定的静态约束冲突，还能保留未知条件、定位问题节点，并按用户要求生成有证据的事实摘要。

**这是一项可跨 Agent 复用的 Skill 工作流，不依赖 Codex 专有 API。** Codex 是当前主要开发与实际验收环境；其他具备 Agent Skills 加载能力和终端权限的 Agent 也可以尝试使用同一套 Python 工具与规则。

| 项目信息 | 内容 |
| --- | --- |
| 作业类型 | **AIGC 通识课程个人 Skill 开发** |
| 项目名称 | **QuantScout-Skill** |
| 实际应用 | 地平线 **RDK X5（Bayes-e）** 的 ONNX 模型量化前静态检查 |
| 当前版本 | **V1.5-S1.1**；Python 包 `0.5.0`；Analysis Schema `1.3` |
| 使用方式 | **自然语言 + 终端**，可由不同支持 Agent Skills 的 Agent 编排 |
| 运行环境 | Ubuntu / WSL，Python 3.10+；静态检查**无需 GPU、Docker 或开发板** |

## 一、项目成果概览

当前版本已包含 **完整的 Skill 入口、独立 Python 分析器、RDK X5 规则资料、公开 ONNX 演示报告及自动化测试**。无需安装环境，也可以先通过以下材料了解项目的实现效果。

| 项目成果 | 直接查看 |
| --- | --- |
| **实际模型检查报告** | [完整诊断报告](examples/demo-report/report.md) · [结构化分析数据](examples/demo-report/analysis.json) |
| **AI 事实摘要** | [Agent 生成的摘要](examples/demo-report/summary.md) · [校验用事实包](examples/demo-report/summary_facts.json) |
| **Agent 工作流设计** | [`SKILL.md` 入口](.agents/skills/rdk-x5-onnx-doctor/SKILL.md) · [事实摘要契约](.agents/skills/rdk-x5-onnx-doctor/references/summary_only_contract.md) |
| **真实运行与测试** | [独立 Agent 验收记录](docs/validations/V1_5_S1_1_Agent_E2E.md) · [GitHub Actions](https://github.com/Stradlin1/QuantScout-Skill/actions/runs/38021773010) |
| **使用与技术文档** | [快速上手](docs/GETTING_STARTED.md) · [架构介绍](docs/PROJECT_OVERVIEW.md) · [测试说明](docs/VALIDATION_STATUS.md) |

项目面向 **RDK X5 ONNX 量化前静态诊断**，不依赖 Docker、GPU 或开发板完成这一阶段的检查，也不替代后续真实量化与部署验证。

## 二、我为什么选择这个题目？

在将神经网络部署到 RDK X5 时，需要先把训练模型导出为 **ONNX**（神经网络交换格式），再交给地平线工具链进行量化与编译。即使 ONNX 文件本身符合格式规范，也不代表所有计算节点都满足目标 **BPU**（神经网络加速单元）的静态约束。

以卷积算子为例：模型的卷积核尺寸可能符合 ONNX 规范，却超出当前已收录的硬件规则范围。人工核查往往需要反复翻阅模型节点、Tensor、算子文档和不同版本的工具链配置。

我希望用 Skill 解决三个具体问题：

1. **把重复的静态检查自动化。** 提取 OpSet、算子、节点、输入输出、Shape、Tensor 信息，并核对已有规则。
2. **让结论可追溯。** 当发现确定冲突时，保留具体节点 ID、规则 ID、实际值、期望范围和来源，而不只回答“模型有问题”。
3. **约束 AI 的不确定判断。** 未覆盖的算子不直接说成“不支持”，静态检查通过也不说成“已经成功部署”。

因此，它是一个**量化前诊断助手**，不是自动量化器、ONNX 编辑器或编译器。

## 三、已实现的核心成果

目前项目已经形成 **从 ONNX 模型读取、约束检查、证据定位，到 Agent 组织结果的完整静态诊断工作流**。相比单纯输出一份模型信息表，它更关注“具体发现了什么、证据在哪里、哪些地方尚不能确认”。

| 核心模块 | 已实现能力 | 具体成果 |
| --- | --- | --- |
| **ONNX 静态解析与结构审计** | 读取主图节点、OpSet、输入输出和 Tensor 连接，进行格式检查、有限 Shape 推断及已知载荷统计 | `analysis.json`、`report.md` |
| **RDK X5 BPU 专项规则诊断** | 将实际节点参数与已收录条件逐项比对，记录规则来源、实际值、允许范围与节点状态 | 版本化规则及可定位的逐规则证据 |
| **计算图深度查询** | 追踪节点与模型输出的依赖关系，查询 Tensor、Shape、资源信息和部分需要人工复核的结构候选 | `inspect`、`trace`、`shape`、`tensor` 等命令 |
| **Agent 自然语言任务编排** | 根据需求选择完整诊断、版本核对、局部追问或官方资料查证，不要求用户记住全部工具命令 | `SKILL.md` 与分支式诊断流程 |
| **受事实约束的 AI 摘要** | 根据“简短总结”“仅列异常”“只看输入输出”切换摘要模式，Agent 组织内容，Python 复核后发布 | `summary_facts.json`、Agent 草稿、`summary.md` |

当前规则库包括 **14 类算子、66 条自动检查或复核条目**，覆盖了部分常见卷积、张量运算和注意力相关静态约束。**这些规则不是整个 ONNX 算子体系的完整兼容性证明**；未覆盖项会明确标记，而不是被自动判为通过。

### 技术特色：AI 不直接“猜测”诊断结果

项目没有把全部判断交给语言模型，而是将能力分成两层：

**底层分析器负责可复核的计算。** 包括节点计数、参数读取、规则检查、状态归类、Tensor 和 Shape 证据。Python CLI 可以独立运行，因此不同 Agent 使用时能够共享同一套检查口径。

**Agent 负责理解任务和组织流程。** 通过 Skill 提供的工作流，它能根据不同请求选取所需的工具与证据，而不是固定执行一连串无关命令。

```text
“检查这个 ONNX 能否进入当前 RDK X5 量化流程”
          ↓
Agent 读取 Skill → 分析模型 → 核对 Profile 与静态规则
          ↓
定位已确认的冲突、未覆盖算子和未知条件
          ↓
引用实际节点、参数和规则依据组织答复

“只总结已经得到的检查结果，不要分析和建议”
          ↓
Python summary-facts → Agent 组织中文草稿
          ↓
Python summary-publish → 核验事实并发布 Markdown
```

### AI 事实总结：支持三种自然语言模式

| 用户描述 | 模式 | 输出侧重点 |
| --- | --- | --- |
| “简短总结，只要事实。” | `overview` | 模型概况、四类诊断状态、代表冲突、检查边界 |
| “只列出异常，不要建议。” | `anomalies` | 已确定 FAIL、待验证和未覆盖项分类呈现 |
| “只看模型输入输出。” | `io` | Graph I/O 名称、Shape、dtype 和真实省略数 |

V1.5-S1.1 增加了**机器可读的有界事实包与二次发布校验**。Agent 在摘要中引用 `{{FACT.ID}}`，发布器重新计算并核对事实、来源 SHA256、必须引用的内容及安全输出路径，减少自行编造数字或超出事实进行扩写的风险。

这里的 AI **确实参与事实选择、顺序及受限中文组织**，而不是将 Python 固定模板冒充 AI 输出。但目前不允许任意自由改写，也不声称可对任意中文做形式化“零幻觉”保证。

## 四、一个可直接检查的实际案例

仓库中包含 `examples/demo.onnx` 和对应公开报告。这个小模型有 **5 个已解析主图节点**：

| 最终状态 | 数量 | 含义 |
| --- | ---: | --- |
| `NO_VIOLATION_FOUND` | 2 | 已执行规则未发现确定违规，不等于完整兼容 |
| `VIOLATION` | 1 | 至少有一条确定的静态规则 FAIL |
| `NEEDS_VERIFICATION` | 0 | 本例无最终属于此状态的节点 |
| `NOT_COVERED` | 2 | 当前规则未覆盖，不能视作通过 |
| **合计** | **5** | 统计范围仅限已解析主图 |

其中，`main/node_000000`（Conv，原名 `oversized_kernel`）的卷积核高度为 **32**；已收录规则 `X5-CONV2D-KERNEL-H` 的范围为 **1～31**，因此工具记录了具体静态冲突。

这是一个**可解释、可复核的规则命中示例**，不是已经运行地平线编译器后得到的失败结果。

可以依次查看：[`analysis.json` 中的原始事实](examples/demo-report/analysis.json) → [`report.md` 完整报告](examples/demo-report/report.md) → [Agent 生成的 `summary.md`](examples/demo-report/summary.md)。

## 五、如何使用与体验

仓库提供公开示例模型 `examples/demo.onnx`，可以直接查看已提交的[诊断报告](examples/demo-report/report.md)和[事实摘要](examples/demo-report/summary.md)，也可以在自己的 Ubuntu / WSL 环境中运行。

### 1. 安装并运行静态诊断

要求 Python **3.10+**，无需配置 OpenExplorer Docker 或连接 RDK X5 开发板：

```bash
git clone https://github.com/Stradlin1/QuantScout-Skill.git
cd QuantScout-Skill

python3 -m venv .venv
.venv/bin/python -m pip install -e .

# 对示例模型执行静态诊断
.venv/bin/python -m rdkx5_doctor analyze \
  --model examples/demo.onnx \
  --out reports/quickstart

# 查询有确定静态冲突的节点
.venv/bin/python -m rdkx5_doctor nodes \
  --analysis reports/quickstart/analysis.json \
  --status VIOLATION
```

运行后，`reports/quickstart/` 中会有 `analysis.json` 和 `report.md`。如果目录已经存在，请指定新的输出目录；诊断和摘要工具默认保护已有输出。

### 2. 与 AI Agent 自然语言交互

在具备 **Agent Skills 加载能力、工作区文件访问权限和终端工具调用能力**的 Agent 中打开本仓库，使用例如下面的请求：

> 使用项目中的 `rdk-x5-onnx-doctor` Skill，检查 `examples/demo.onnx` 的 RDK X5 量化前静态约束，说明已确认的异常及其规则依据，把报告保存到一个新的 `reports/` 子目录。不要修改模型或执行量化。

如果已经有 `analysis.json`，也可以直接要求：

> 总结已有的 `reports/quickstart/analysis.json`，只保留检查事实，不分析原因，不给修改建议，输出到同目录中尚不存在的 `summary.md`。不要重新分析模型。

将这句话改为“只列出异常”或“只看输入输出”，即可触发不同的摘要工作流。**这一过程不限定使用 Codex**，具体的 Skill 发现方式和终端权限由各 Agent 客户端决定。

不使用 AI 时也可以运行 `python -m rdkx5_doctor summary` 得到确定性摘要，但它只是 Python CLI 功能，不应当作 Agent 撰写的演示。更多操作见 [详细使用教程](docs/GETTING_STARTED.md)。

## 六、多 Agent 复用：不局限于 Codex

QuantScout-Skill 采用 [Agent Skills](https://agentskills.io/) 的 `SKILL.md` 指令结构，将工作流与 Python 分析引擎解耦。**检查规则、报告 Schema 和 Python CLI 不依赖某个特定大模型或 Agent 厂商**，因此不同 Agent 可以复用同一套静态诊断工具，差异主要在于如何发现 Skill、调用终端及组织回答。

当前仓库的 Skill 入口为 `.agents/skills/rdk-x5-onnx-doctor/SKILL.md`，可供支持相应项目级 Skill 目录的客户端读取。可考虑的使用环境包括：

| Agent 环境 | 接入方式 | 本项目验证情况 |
| --- | --- | --- |
| **OpenAI Codex** | 当前仓库的 `.agents/skills/` | **已有独立自然语言端到端测试** |
| **Cursor Agent** | 在支持项目级 Agent Skills 的版本中读取 Skill | **结构可复用，尚未完成本项目 E2E** |
| **GitHub Copilot Agent** | 使用支持 Agent Skills 的工作区 / Agent 模式 | **结构可复用，尚未完成本项目 E2E** |
| **Gemini CLI** | 在支持 Skill 发现的版本中加载项目 Skill | **结构可复用，尚未完成本项目 E2E** |
| **OpenCode** | 在支持 Agent Skills 的版本中加载项目 Skill | **结构可复用，尚未完成本项目 E2E** |
| **Claude Code** | 需适配 `.claude/skills/` 目录或设置映射，并核对相对引用 | **需适配，尚未完成本项目 E2E** |

**跨 Agent 的关键不是简单复制一个 `SKILL.md`。** 目标客户端需要能读取对应的参考文件、在 Python 3.10+ 环境调用终端命令，并安装本仓库的 `rdkx5_doctor` 包。对于缺少联网检索能力的 Agent，纯离线静态检查与事实摘要仍有可使用的流程，官方资料查证则必须如实标明是否实际执行。

目前已经实测和记录的是 **Codex**；表格中的其他工具是**可迁移的目标环境，并非已全部验证通过的兼容性认证**。项目对外名为 `QuantScout-Skill`，内部 Skill 仍叫 `rdk-x5-onnx-doctor`，Python 模块仍叫 `rdkx5_doctor`，更换 Agent 不需要改动规则算法。

## 七、测试结果与实际边界

| 测试项目 | 已有记录 | 查看证据 |
| --- | --- | --- |
| Python 本地自动化测试 | 开发日志记录 **528 passed / 0 failed** | [V1.5-S1.1 实施日志](docs/QuantScout_V1_5_S1_1_Development_Log.md) |
| GitHub Actions | **Python 3.10 和 3.12 均成功**，执行安装、规则校验、pytest 和构建 | [在线 CI 运行记录](https://github.com/Stradlin1/QuantScout-Skill/actions/runs/38021532036) |
| 独立 Codex Agent 端到端测试 | **累计 11 次：10 PASS、1 FAIL** | [E01～E10 和复验记录](docs/validations/V1_5_S1_1_Agent_E2E.md) |
| 公开示例 | 已提交 Schema 1.3 报告与独立 Agent 摘要 | [公开演示核对](docs/validations/V1_5_S1_1_Public_Demo.md) |

**为什么会有一次 FAIL？** 第一次 E10 中，Agent 绕过专门的发布器，通过文件操作覆盖了已有摘要。之后加强了 Skill 的“目标已存在就停止”要求，并使用新的独立会话复验通过。失败被如实保留，说明**发布器本身的防覆盖检查不等于对 Agent 所有文件权限的绝对保护**。

GitHub Actions 只验证确定性程序逻辑；独立 Codex 测试用于验证真实自然语言工作流。项目尚未开展正式的“普通 Prompt 与 Skill”对照实验，不应声称当前 Skill 必然优于所有其他工作方式。

也可以本地运行回归测试：

```bash
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest -q
```

## 八、已知限制和后续工作

**本项目刻意不做：** Docker 量化、`hb_mapper` 编译、ONNX 模型改写、板端运行，以及实际 FPS、精度或 BPU 分配预测。它只负责进入量化流程之前的静态诊断。

目前只统计 ONNX **主图节点**，未递归展开 If / Loop 等嵌套子图；`NOT_COVERED` 不代表硬件不支持，`NO_VIOLATION_FOUND` 也不是可部署保证。当前已有 Codex 的真实验收记录；同一套 Skill、规则和 Python 工具可以移植到其他支持 Agent Skills 的客户端，但其他 Agent 尚未完成相同级别的端到端验证。

**尚未实现、不属于当前交付的功能：** V1.5-S2 Tensor 资源来源细分类、V1.5-S3 量化敏感结构候选提示，以及正式的无 Skill / 有 Skill 对照实验。

## 九、项目文件与开发记录

```text
QuantScout-Skill/
├── .agents/skills/rdk-x5-onnx-doctor/  # 可复用 Agent Skill 入口与执行契约
├── src/rdkx5_doctor/                   # Python 静态诊断、规则与发布校验
├── references/                         # 官方文档与规则审核依据
├── examples/                           # 可查看的 ONNX 示例和报告
├── tests/                              # 自动化测试
├── .github/workflows/ci.yml            # Python 3.10 / 3.12 CI
├── docs/                               # 开发规范、操作说明、验收证据
└── pyproject.toml
```

其他材料：[项目背景与技术架构](docs/PROJECT_OVERVIEW.md) · [完整操作教程](docs/GETTING_STARTED.md) · [测试与限制](docs/VALIDATION_STATUS.md) · [文档导航](docs/README.md)。

---

**项目总结：** QuantScout-Skill 把实际的 RDK X5 量化前诊断任务整合为一套可跨 Agent 迁移的工作流：由 Agent 理解自然语言需求、调用工具并组织结果，由确定性 Python 引擎提供规则检查和可追溯证据。项目已有可运行的代码、公开报告、独立 Agent 验收及 CI 记录；后续可以在更多支持 Agent Skills 的工具中开展跨平台验证。