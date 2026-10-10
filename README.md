# QuantScout-Skill

**AIGC 通识课程个人结课作业｜面向地平线 RDK X5 的 ONNX 量化前诊断 Skill**

QuantScout-Skill 是我为 AIGC 通识课程开发的 AI Skill。项目来自实际使用地平线 **RDK X5（Bayes-e）**部署神经网络模型时遇到的问题：在把 ONNX 模型交给量化工具链之前，往往需要先确认模型版本、输入输出、计算图结构，以及部分算子是否满足目标 BPU 的约束。

我希望把这些重复的检查交给 AI：**向 Agent 提出“帮我检查这个 ONNX”，它就可以按需调用确定性的 Python 分析工具、参考已审核规则或查询官方资料，最终给出有依据的中文诊断报告。**

它不是 ONNX 编辑器，也不是一键量化软件，而是一个**量化前的静态诊断助手**。

- **目标平台：** 地平线 RDK X5（Bayes-e）。
- **使用方式：** 自然语言 + 终端；不提供网页界面。
- **主要开发环境：** Windows VS Code + WSL Ubuntu + Codex 插件，也可使用原生 Ubuntu。
- **兼容性设计：** 基于 `SKILL.md` 和 `.agents/skills/` 组织，便于在其他支持 Agent Skills 的 Agent 中复用。
- **已发布基线：** V1.4；Python 包 `rdkx5-onnx-doctor` 0.5.0，分析 JSON Schema 1.3。后续版本以仓库实际提交为准。
- **基本原则：** 只读 ONNX、保留来源与不确定性、不将静态检查等同于量化或部署验证。

## 一、项目背景：为什么需要它？

### 先认识几个概念

| 名词 | 通俗解释 |
| --- | --- |
| **ONNX** | 一种常用的神经网络模型交换格式，保存模型的计算节点、连接关系、参数及输入输出等信息。 |
| **OpSet** | ONNX 算子的版本集合。不同工具链可能对可接受的版本有所要求。 |
| **量化** | 让模型中的部分计算和数据采用更适合目标硬件的数值表示，例如 INT8。 |
| **BPU** | RDK X5 上用于加速神经网络计算的处理单元。模型符合 ONNX 规范，不代表每个算子都满足 BPU 的具体约束。 |
| **Skill** | 一组可复用的 Agent 工作指令，可以配合脚本、规则和参考资料执行特定任务。 |

例如，某个卷积层可能是完全合法的 ONNX 节点，但它使用的卷积核尺寸等属性仍可能超出当前已收录的 X5 BPU 规则范围。QuantScout-Skill 的作用就是提前发现这种**“ONNX 合法，但需要进一步核对目标硬件约束”**的情况。

### 我的地平线工具链环境

我实际使用的是 **OpenExplorer v1.2.8**，对应的量化工具链目录为：

```text
horizon_x5_open_explorer_v1.2.8/
└── horizon_x5_open_explorer_v1.2.8-py310_20240926/
```

当前项目使用的工具链 Profile 要求 **标准 ONNX 主域 OpSet 11**。这是本项目的具体配置要求，**不是说所有版本的 RDK X5 工具链都只能使用 OpSet 11**。

### 为什么使用 Ubuntu / WSL？

完整的 RDK X5 量化工作通常需要在 **Ubuntu/Linux 环境中配置和使用 OpenExplorer 的 Docker 工具链**。我选择 Windows VS Code 连接 WSL Ubuntu，是为了同时使用熟悉的编辑器与 Codex，并与后续的 Linux/Docker 量化工作目录衔接。

但需要区分两个阶段：

- **QuantScout-Skill 当前实现的阶段：** 量化前 ONNX 静态诊断，直接在 Python 环境中运行，**不需要启动 Docker**。
- **后续实际量化阶段：** 使用者自行在配置好的 OpenExplorer Docker 环境中执行校准、转换和编译；**本 Skill 不负责执行该阶段**。

因此，选择 Ubuntu/WSL 主要是为了贴合真实量化流程，并不是说 ONNX 静态分析本身依赖 Docker。

## 二、QuantScout-Skill 可以做什么？

### 工作原理

```text
用户以自然语言描述需求
          ↓
Agent 读取 SKILL.md，判断需要哪些检查
          ↓
调用本地 Python 工具读取 ONNX 与已有分析结果
          ↓
核对当前用户 Profile、X5 BPU 规则和 Tensor/Shape 事实
          ↓
有必要时查询可信官方资料或本地离线手册
          ↓
生成中文说明及可复核的 JSON / Markdown 报告
```

**Agent 负责理解任务、选择步骤和解释已有证据；Python 负责解析模型、计算数值及执行确定性规则。** 规则库和参考文档并不因为更换 Agent 而改变。

### 当前 V1.4 的主要功能

| 功能 | 能做什么 | 使用价值 |
| --- | --- | --- |
| **ONNX 基础检查** | 读取 OpSet、输入输出、节点及计算图，执行格式检查 | 区分模型文件问题与硬件约束问题 |
| **当前工具链版本预检** | 将模型 OpSet 与用户提供的 Profile 比较，返回 `MATCH` / `MISMATCH` / `UNKNOWN` | 提前识别版本配置不匹配 |
| **BPU 静态约束检查** | 检查已覆盖算子的可确认条件，并记录节点 ID、实际参数、期望范围及依据 | 定位具体约束冲突 |
| **官方资料查证** | 对未覆盖算子按需查官方文档或固定版本的离线资料 | 避免根据 AI 印象猜测算子支持情况 |
| **Shape 分析** | 跟踪已知和未知的维度、有限范围内的静态证明及冲突 | 理解模型结构与动态维度情况 |
| **Tensor 理论资源统计** | 依据已知 Shape 和 dtype 计算理论原始载荷 | 了解可静态确定的数据规模 |
| **计算图追踪** | 查询异常节点的前后继及其可达输出 | 了解节点在模型中的位置与依赖关系 |
| **结构优化候选识别** | 标注部分恒等变换、逆 Transpose、无操作 Reshape、Conv-BN 融合审查等候选 | 为人工复核提供结构证据，不直接修改模型 |
| **诊断报告** | 生成 `analysis.json`、`report.md` 等，并保留独立的版本检查与知识查询记录 | 便于查看、留档、复核 |

当前已发布基线规则库覆盖 **14 类算子、66 条自动检查或复核条目**。条目数量不等于全 ONNX 算子的覆盖率，也不代表符合静态规则的节点就一定运行在 BPU 上。

#### 一个小例子

仓库中的 `examples/demo.onnx` 包含一个专门设置的 Conv 节点，其卷积核高度为 **32**，而当前已收录的对应 X5 BPU 规则范围是 **1～31**。

因此，工具可以明确指出：**模型结构检查可以通过，但该 Conv 节点触发了已收录的 BPU 静态约束冲突。** 这并不等于已经执行过地平线量化工具链，也不直接预测编译器的最终处理结果。

## 三、跨 Agent 复用：不只限于 Codex

QuantScout-Skill 按 [Agent Skills](https://agentskills.io/) 的通用结构设计。核心入口是 `SKILL.md`，搭配独立的 Python 命令行分析器、规则 YAML 和参考资料。

项目级 Skill 位于：

```text
.agents/skills/rdk-x5-onnx-doctor/
├── SKILL.md
├── references/
└── scripts/
```

多种 AI Agent 已在官方文档中支持读取这种 Skill 结构。**下表表示各 Agent 的 Skill 发现机制，并不表示 QuantScout 已在全部 Agent 中完成端到端验证。**

| Agent | 官方支持的相关项目目录 | QuantScout 使用说明 |
| --- | --- | --- |
| **OpenAI Codex** | `.agents/skills/` | 当前项目的主要开发与测试环境 |
| **Cursor Agent** | `.agents/skills/` | 可尝试直接发现本仓库 Skill；需实际验证运行流程 |
| **GitHub Copilot Agent** | `.agents/skills/` | 支持项目级 Skill，包括 VS Code 的 Agent 模式；需实际验证 |
| **Gemini CLI** | `.agents/skills/` | 可尝试直接发现 Skill，通过 `/skills list` 查看 |
| **OpenCode** | `.agents/skills/` | 可尝试直接发现并按需加载 Skill |
| **Claude Code** | `.claude/skills/` | 需要额外映射或复制 Skill 入口，并核对参考资料的相对路径 |

这些客户端的 Skill 功能和加载路径可能随版本变化，实际使用以各自当前文档和运行结果为准。

**跨 Agent 复用为什么可行？** 因为决定性的数据读取、规则检查、Shape 分析及报告事实来自同一个 Python 模块 `rdkx5_doctor`；不同 Agent 主要负责解释用户需求、组织调用以及中文表达。

但跨 Agent 运行仍有共同前提：

- Agent 必须能够读取完整仓库，并获得运行本地终端命令的权限。
- 所在 Python 环境需要安装 `rdkx5-onnx-doctor` 包及依赖。
- 不能只复制 `SKILL.md`：本项目还需要对应的 Skill references、仓库 `references/` 及 Python 分析器。
- 在线官方资料检索能力依赖具体 Agent；没有联网能力时，仍可使用项目保留的离线手册，但必须正确标注为缓存证据。
- 语言模型的推理、工具调用权限和路径处理能力不同，**跨平台完整运行效果尚需逐个平台测试**。

相关官方说明：[Codex Skills](https://developers.openai.com/blog/skills-agents-sdk)、[Cursor Skills](https://cursor.com/docs/skills)、[GitHub Copilot Skills](https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/add-skills)、[Gemini CLI Skills](https://geminicli.com/docs/cli/skills/)、[OpenCode Skills](https://docs.opencode.ai/docs/skills/)、[Claude Code Skills](https://code.claude.com/docs/en/skills)。

> **名称说明：** `QuantScout-Skill` 是 GitHub 仓库名和对外展示名称；当前 Codex Skill 的实际内部名称仍为 `rdk-x5-onnx-doctor`，Python 模块仍为 `rdkx5_doctor`。仓库改名没有重命名这些内部接口。

## 四、安装与使用

### 1. 准备环境

推荐：**Windows VS Code + WSL Ubuntu + Codex 插件**。也可使用原生 Ubuntu，或具备同等文件与终端访问能力的其他 Agent。

需要 Git、Python **3.10 或更新版本**及 `venv`。只进行静态预检时，**不需要 GPU、RDK X5 开发板、Docker 或已安装的 OpenExplorer 工具链**。

下面所有命令都在 **Ubuntu / WSL Linux 终端**执行，而不是 Windows PowerShell。

### 2. 克隆仓库与安装 Python 依赖

```bash
git clone https://github.com/Stradlin1/QuantScout-Skill.git
cd QuantScout-Skill

python3 --version
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

若 Ubuntu 提示缺少虚拟环境组件，可安装对应的 `python3-venv` 包。需要运行开发测试时，可额外安装开发依赖：

```bash
.venv/bin/python -m pip install -e '.[dev]'
```

检查是否安装成功：

```bash
.venv/bin/python -m rdkx5_doctor --help
.venv/bin/python -m rdkx5_doctor rules validate
```

### 3. 在 Windows VS Code 的 WSL 工作区调用 Codex

在仓库根目录执行：

```bash
code .
```

确认 VS Code 当前连接的是 **WSL Ubuntu 工作区**，并打开 Codex 插件。Codex 可以根据自然语言任务发现适用的 Skill，也可以在 Skill 选择器中显式选择 `rdk-x5-onnx-doctor`。

首次使用建议在终端生成仓库自带的演示模型：

```bash
.venv/bin/python examples/generate_demo.py
```

然后在 Codex 中输入：

> 使用当前仓库的 `rdk-x5-onnx-doctor` Skill 检查 `examples/demo.onnx`。请核对当前 RDK X5 工具链 Profile、OpSet 与已收录的 BPU 静态规则，说明真实节点和规则依据，并把报告保存在本地 `reports/`。不要改写 ONNX，不执行量化。

检查自己的模型时，将路径换成 WSL 可以访问的 `.onnx` 文件即可。

使用 Cursor、GitHub Copilot、Gemini CLI 或 OpenCode 时，也可以尝试在其 Agent 会话中打开**同一仓库**并提出同样的任务；这些客户端的 Skill 发现和批准执行命令方式可能不同，不能直接把 Codex 的 `$skill-name` 调用语法套用到其他 Agent。

### 4. 不使用 Agent：通过终端直接检查

项目提供独立的 Python 命令行工具，适合复核和调试。

```bash
# 分析示例模型，生成 report.md 和 analysis.json
.venv/bin/python -m rdkx5_doctor analyze \
  --model examples/demo.onnx \
  --out reports/demo

# 核对当前用户配置的 OpSet 11 准入要求
.venv/bin/python -m rdkx5_doctor preflight \
  --analysis reports/demo/analysis.json \
  --profile .agents/skills/rdk-x5-onnx-doctor/references/toolchain_profile_opset11.yaml \
  --json > reports/demo/preflight.json

# 列出触发静态约束冲突的节点
.venv/bin/python -m rdkx5_doctor nodes \
  --analysis reports/demo/analysis.json \
  --status VIOLATION

# 查看一个节点的逐规则证据
.venv/bin/python -m rdkx5_doctor inspect \
  --analysis reports/demo/analysis.json \
  --node oversized_kernel

# 追踪对应节点到模型输出的依赖关系
.venv/bin/python -m rdkx5_doctor trace \
  --analysis reports/demo/analysis.json \
  --node main/node_000000
```

其他已实现的查询命令包括 `shapes` / `shape`、`tensors` / `tensor`、`candidates` / `candidate` 和 `rules list`，可通过 `--help` 查看参数。

如果需要查询官方资料的离线快照：

```bash
.venv/bin/python references/offline_manual/query.py --verify --json
.venv/bin/python references/offline_manual/query.py --operator Shape --json
```

离线资料有固定上游 revision、原始文件校验和许可记录；读取缓存不等于本次已访问官网，也不能据此认定算子在实际编译中由 BPU 执行。

### 5. V1.5-S1：只总结检查事实

自然语言示例：“总结一下 reports/demo/analysis.json 的检查结果，只需要事实，不要分析和建议。”Skill 的 `summary_only` 分支生成独立 `summary.md`；只有 ONNX 时先使用现有 `analyze` 生成事实。完整诊断 `report.md` 保留原有行为，事实摘要不包含根因分析、风险排序或修改建议。

```bash
.venv/bin/python -m rdkx5_doctor summary --analysis reports/demo/analysis.json --limit 3
```

仅支持已验证的 Schema 1.3；计数不一致会拒绝生成，已有 `summary.md` 不覆盖。可通过 `--preflight` 与 `--profile` 核对同次报告的既有 Profile 记录；未提供时明确标注未执行。`--official-lookup` 仅摘录已经确认属于当前运行的记录，不联网。

### 6. 输出文件

通常保存在 `reports/<本次运行目录>/` 下：

```text
reports/demo/
├── analysis.json       # 完整结构化静态证据
├── summary.md          # 可选：独立事实摘要（无分析和建议）
├── report.md           # 中文完整分析报告
└── preflight.json      # 可选：当前用户 Profile 比对记录
```

若 Skill 实际执行了官方知识查询，还可能生成独立的知识查询记录；未执行时不应伪造文件或宣称已经查询。

`reports/` 默认由 Git 忽略，以免将本地模型诊断产物上传到 GitHub。

## 五、如何理解检查结果？

### 四种节点状态

| 状态 | 含义 |
| --- | --- |
| `VIOLATION` | 已执行的规则中，至少一条存在确定的静态 FAIL。 |
| `NO_VIOLATION_FOUND` | 已执行规则未发现确定违规，**不是**完整的 BPU 兼容保证。 |
| `NEEDS_VERIFICATION` | 某些关键信息或条件仍需验证。 |
| `NOT_COVERED` | 没有适用的已注册规则；不能当作检查通过。 |

单条规则还可能返回 `PASS`、`FAIL`、`UNKNOWN` 和 `NOT_APPLICABLE`；它们与节点级状态不是同一层级。

### 不应作出的推断

- `OpSet MATCH` **只表示满足当前用户 Profile 的版本条件**，不能证明 `hb_mapper` 一定编译成功。
- 未发现静态违规，不代表所有算子都有规则覆盖，也不代表全部由 BPU 运行。
- 某个 Tensor 的理论原始字节数，不等于实际 DDR / SRAM 分配，也不等于峰值内存。
- 优化候选不代表编译器已经完成融合或删除了节点。
- 本项目**不执行 Docker、`hb_mapper`、量化校准、模型编译或板端推理**，不能给出实际 FPS、量化精度或部署成功保证。

## 六、计划完善的功能

项目范围固定为 **RDK X5 ONNX 量化前诊断**，不会扩展为自动量化或自动部署系统。V1.5-S1 已实现，后两项仍为后续规划：

| 功能 | 规划内容 | 边界 |
| --- | --- | --- |
| **AI 诊断事实总结（V1.5-S1，已实现）** | 从现有 `analysis.json` 和实际存在的 `preflight.json` 中提取统计，生成简洁的中文事实摘要 | **只总结，不增加原因分析、风险评价和修改建议** |
| **Tensor 资源分类分析（V1.5-S2）** | 区分中间特征图、模型权重、Shape 参数、输出等，分别统计已知理论载荷 | 不将理论载荷称为真实硬件内存占用 |
| **量化敏感结构提示（V1.5-S3）** | 基于可验证的计算图特征提示值得关注的结构 | 不预测量化误差、精度或编译结果 |

以上是开发方向；**实际完成情况以仓库代码、版本记录和运行结果为准**。

## 七、项目结构与文档

```text
QuantScout-Skill/
├── .agents/
│   └── skills/
│       └── rdk-x5-onnx-doctor/
│           ├── SKILL.md
│           ├── references/
│           └── scripts/
├── src/rdkx5_doctor/       # Python 静态诊断器与规则资源
├── references/            # 官方资料、规则依据、离线手册
├── examples/              # 演示 ONNX 与报告生成脚本
├── docs/                  # Schema、研发记录和详细规范
├── tests/                 # 自动化测试
├── AGENTS.md
└── pyproject.toml
```

更多技术资料：

- [Codex Skill 入口](.agents/skills/rdk-x5-onnx-doctor/SKILL.md)
- [V1.4 开发记录](docs/RDK_X5_ONNX_Doctor_V1_4_Development_Log.md)
- [Analysis Schema 1.3](docs/ANALYSIS_SCHEMA_V1_3.md)
- [官方资料离线检索说明](references/offline_manual/README.md)
- [原始官方来源与规则依据](references/official_sources.md)

---

**QuantScout-Skill 的目标是让 ONNX 量化前检查更容易启动、结果更容易理解、依据更容易复核，并尽可能复用于不同 AI Agent。**