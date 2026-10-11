# QuantScout-Skill

**面向地平线 RDK X5（Bayes-e）的 ONNX 量化前静态诊断 Agent Skill**

> AIGC 通识课程个人项目 · 当前交付版本 **V1.5-S1.1** · Python 包 **0.5.0** · Analysis Schema **1.3**

QuantScout-Skill 将模型部署前反复进行的 **ONNX 结构分析、BPU 静态约束检查、问题节点定位与结果整理**，组织为一套能够通过自然语言调用的工作流。用户提出检查需求后，Agent 根据 Skill 指令选择工具，Python 提供确定性检查结果，Agent 再根据真实证据组织中文说明。

项目的核心是 **Agent 智能编排 + RDK X5 专项规则 + 可追溯的事实校验**。这不仅是一段提示词，也不仅是一份固定格式的 Python 检查报告，而是将 AI 的任务理解能力与可靠的程序分析能力结合起来。

| **14 类算子** | **66 条规则** | **9 个真实 ONNX** | **528 项测试** |
| :---: | :---: | :---: | :---: |
| 当前规则库收录 | 静态检查及复核条目 | V1.3 阶段历史回归，9/9 分析成功 | V1.5-S1.1 开发日志记录 528 passed |

**查看成果：** [完整诊断报告](examples/demo-report/report.md) · [Agent 事实摘要](examples/demo-report/summary.md) · [Skill 入口](.agents/skills/rdk-x5-onnx-doctor/SKILL.md) · [GitHub Actions](https://github.com/Stradlin1/QuantScout-Skill/actions/runs/38022125687)

## 01 · 项目介绍

在模型从训练框架导出为 ONNX、进入 RDK X5 工具链量化之前，开发者通常需要检查模型的 OpSet、输入输出、算子参数、Tensor 连接和 BPU 约束。**ONNX 格式正确，并不意味着模型的全部节点都符合目标硬件的要求。**

传统方式需要人工翻阅算子资料、检查计算图并整理错误原因。QuantScout 将这些步骤拆分为可以重复执行的静态分析和 Agent 工作流，让使用者不必记忆全部命令，也能针对某个异常继续追问其依据。

```text
自然语言需求
     ↓
Agent 读取 SKILL.md，识别任务并选择工具
     ↓
Python 静态诊断 + RDK X5 规则库 + 按需参考资料查询
     ↓
结构化证据（analysis.json）与完整报告（report.md）
     ↓
Agent 按用户需求组织中文说明 / 事实摘要
```

**两部分各司其职：** Python 负责读取真实参数、计算数值、核对规则和验证事实；Agent 负责理解意图、安排检查顺序、调用必要的查询并解释已有证据。明确的未知条件会保留为未知，而不是被补写成“兼容”。

本项目采用 `SKILL.md` 和独立 Python CLI，不依赖 Codex 专用 API。具备 **Agent Skills 加载能力、文件访问权限和终端调用能力**的其他 Agent 可以适配使用；目前真实端到端验收使用的是 **OpenAI Codex**，其他客户端尚未逐一实测。

## 02 · 核心功能与技术特色

| 功能 | 已实现的内容 |
| --- | --- |
| **ONNX 结构审计** | 模型格式校验，提取主图节点、OpSet、输入输出、算子类型和 Tensor 连接 |
| **BPU 静态规则诊断** | 核对已收录的 RDK X5 算子参数约束，输出具体节点、规则 ID、实际值、允许条件和来源 |
| **工具链版本预检** | 对比模型与指定工具链 Profile 的标准 ONNX 主域 OpSet 条件，不冒充真实编译结论 |
| **Shape / Tensor 资源检查** | 有限的静态 Shape 推断、未知原因与证据追踪、根据已知 dtype/Shape 计算理论 Tensor 载荷 |
| **计算图查询与结构候选** | 节点详情、上下游依赖、可达输出，以及 Identity、Cast、Transpose、Reshape、Conv-BN 等结构复核候选 |
| **Agent 诊断与资料查证** | 根据问题选择完整诊断、局部查询或按需官方资料查询；来源、版本和未查范围分别记录 |
| **受事实约束的 AI 总结** | Agent 组织摘要，Python 重新核对统计、来源哈希及事实引用，再发布 Markdown |

### AI 如何参与诊断，而不是仅运行固定脚本？

例如，用户询问“这个模型有哪些已确定的硬件约束冲突”，Agent 可以从已有报告中定位 `VIOLATION` 节点，调用 `inspect` 查阅具体参数；询问“这个节点会影响哪些模型输出”，则按需执行 `trace`，不必重新分析整个模型。

对于已经得到的检查结果，V1.5-S1.1 还支持三种**自然语言事实总结**：

| 用户需求 | 模式 | 摘要内容 |
| --- | --- | --- |
| “简短总结一下，只要事实。” | `overview` | 模型概况、检查状态、代表冲突和检查范围 |
| “只列出异常，不要分析或建议。” | `anomalies` | 已确认违规、待验证与未覆盖内容分开列示 |
| “只看模型输入和输出。” | `io` | Graph I/O 名称、Shape、dtype 与省略数量 |

此过程不是调用旧 `summary` 命令冒充 AI 生成：**`summary-facts` 导出可信事实 → Agent 选择和组织事实 → `summary-publish` 复核后安全发布**。摘要中的事实采用 `{{FACT.ID}}` 引用，避免 Agent 自行编造数字；语言组织采用受限句式，不声称能对任意中文做到绝对无幻觉。

## 03 · 实际运行效果

仓库提供了一个**可复现的微型 ONNX 演示模型**：`examples/demo.onnx`（标准域 OpSet 11）。检查器在其 **5 个已解析主图节点**上得到以下结果：

| 节点最终状态 | 数量 | 解释 |
| --- | ---: | --- |
| `NO_VIOLATION_FOUND` | 2 | 已执行规则未发现确定冲突 |
| `VIOLATION` | 1 | 存在已确认的静态规则冲突 |
| `NEEDS_VERIFICATION` | 0 | 当前示例中没有此类最终节点 |
| `NOT_COVERED` | 2 | 未被当前规则覆盖，不能视为通过 |
| **合计** | **5** | 仅统计已解析的主图节点 |

**一个被准确定位的静态冲突：**

```text
节点 ID      main/node_000000
原始名称     oversized_kernel
算子         Conv
规则 ID      X5-CONV2D-KERNEL-H
实际参数     kernel_h = 32
规则范围     kernel_h = 1～31
结果         VIOLATION（已确认静态规则冲突）
```

这条记录明确说明了**哪个节点、哪个参数、违反哪条已收录规则**，而不只是输出“模型存在问题”。它是静态规则命中的演示，**不是已经运行 hb_mapper 后得到的编译失败记录**。

同一个示例提供了从机器事实到中文摘要的公开产物：

[`analysis.json`：结构化诊断](examples/demo-report/analysis.json) → [`report.md`：完整报告](examples/demo-report/report.md) → [`summary.md`：独立 Codex 会话生成的事实摘要](examples/demo-report/summary.md)

此外，项目曾使用 **9 个真实 ONNX 模型**完成 V1.3 阶段回归，包括 MobileNetV2、ResNet18、TinyBERT、ViT 和 YOLO26n 等，记录为 **9/9 格式校验与分析成功，原始模型哈希保持不变**。该次回归使用较早的 **10 类算子、49 条规则**，不能等同于对当前 66 条规则的重新全量验证。[查看多模型回归记录](docs/MULTI_MODEL_VALIDATION.md)

## 04 · 安装与使用

在 **Ubuntu / WSL，Python 3.10+** 环境下可直接运行。静态诊断无需 GPU、开发板或 OpenExplorer Docker。

```bash
git clone https://github.com/Stradlin1/QuantScout-Skill.git
cd QuantScout-Skill

python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m rdkx5_doctor rules validate

# 分析示例模型；输出目录需尚不存在
.venv/bin/python -m rdkx5_doctor analyze \
  --model examples/demo.onnx --out reports/quickstart

# 查看确定的静态规则冲突
.venv/bin/python -m rdkx5_doctor nodes \
  --analysis reports/quickstart/analysis.json --status VIOLATION
```

分析产物保存在 `reports/quickstart/` 中。要进一步查看示例节点的规则证据，可运行：

```bash
.venv/bin/python -m rdkx5_doctor inspect \
  --analysis reports/quickstart/analysis.json --node oversized_kernel
```

**使用 Agent 时**，在能够读取整个仓库、执行终端命令的工作区中提出自然语言请求即可，例如：

> 使用当前仓库的 `rdk-x5-onnx-doctor` Skill，检查 `examples/demo.onnx` 的 RDK X5 量化前静态约束。将报告保存到一个新的 `reports/` 子目录；说明已确认的异常和规则依据，不修改模型，也不执行量化。

也可以针对已经生成的报告提出：

> 读取 `reports/quickstart/analysis.json`，仅总结已有检查事实，不分析原因，不给修改建议。保存为同目录中尚不存在的 `summary.md`，不要重新分析模型。

详细的安装、其他 CLI 命令和多模式摘要用法见 [快速上手](docs/GETTING_STARTED.md)。

## 05 · 测试与验证

项目除了单元测试，还保留了真实模型回归和**独立 Agent 会话**的验收记录：

| 验证范围 | 已有结果 | 可查看证据 |
| --- | --- | --- |
| **开发阶段自动化测试** | V1.5-S1.1 开发日志记录 **528 passed / 0 failed** | [开发日志](docs/QuantScout_V1_5_S1_1_Development_Log.md) |
| **GitHub Actions** | Python **3.10 / 3.12** 两个 CI 作业通过，执行规则校验、测试与构建 | [当前代码基线的 CI](https://github.com/Stradlin1/QuantScout-Skill/actions/runs/38022125687) |
| **真实 ONNX 回归** | V1.3 阶段 **9/9** 模型分析成功，未经改写 | [多模型验证记录](docs/MULTI_MODEL_VALIDATION.md) |
| **真实 Agent 事实总结** | 累计 **11 次独立 Codex 会话：10 PASS、1 FAIL**；失败场景修正后另起会话复验通过 | [Agent E2E 记录](docs/validations/V1_5_S1_1_Agent_E2E.md) |
| **公开演示** | 提交 Schema 1.3 静态报告与 Agent 真实生成的摘要 | [公开演示核对](docs/validations/V1_5_S1_1_Public_Demo.md) |

Agent 测试中曾有一次**绕过发布器、覆盖已有摘要文件**的失败案例。随后加强了 Skill 的目标文件检查与禁止绕过约束，独立复验通过；保留原失败记录，不将其描述为从未失败，也不声称具备操作系统级文件权限隔离。

GitHub Actions 验证的是确定性程序部分，真实 Agent 的自然语言执行另行测试。**目前 Codex 有实际 E2E 记录；其他 Agent 的完整兼容性尚未实测，也尚未开展正式的无 Skill / 有 Skill 对照实验。**

## 06 · 项目结构与说明

```text
QuantScout-Skill/
├── .agents/skills/rdk-x5-onnx-doctor/  # Skill 入口、任务路由和参考契约
├── src/rdkx5_doctor/                   # Python CLI、规则检查与事实校验
├── references/                         # 官方资料与规则审核依据
├── examples/                           # 可运行的 ONNX 示例及公开报告
├── tests/                              # 自动化测试
├── docs/                               # 使用教程、技术文档、开发与验收记录
├── .github/workflows/ci.yml            # Python 3.10 / 3.12 持续集成
└── pyproject.toml                       # Python 项目及依赖配置
```

**名称与版本：** 仓库名为 `QuantScout-Skill`；Skill 内部名称是 `rdk-x5-onnx-doctor`；Python 包名是 `rdkx5-onnx-doctor`，导入与 CLI 模块为 `rdkx5_doctor`。当前课程交付版本为 **V1.5-S1.1**。

**适用范围：** 项目只执行 ONNX 量化前静态分析；节点级统计限于已解析主图，不递归检查嵌套子图。它**不执行 Docker 量化、编译、ONNX 改写或板端测试**，也不保证 CPU/BPU 实际分配、量化精度、FPS 或真实硬件内存占用。`NOT_COVERED` 不代表硬件不支持，`NO_VIOLATION_FOUND` 也不等于完整兼容。

更多资料：[项目技术架构](docs/PROJECT_OVERVIEW.md) · [完整使用教程](docs/GETTING_STARTED.md) · [验证边界](docs/VALIDATION_STATUS.md) · [文档索引](docs/README.md)

---

**QuantScout-Skill 的目标是让模型量化前的静态检查更容易执行、异常更容易定位、结论更容易复核，并让 Agent 在明确的事实和规则约束下完成实际工程任务。**