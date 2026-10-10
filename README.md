# QuantScout-Skill

**AIGC 通识课程个人结课作业｜面向 RDK X5 的 ONNX 量化前诊断 Agent Skill**

这是我结合实际模型部署需求开发的 AI Skill。用户只需用自然语言提出“检查这个 ONNX 模型”或“只总结检查事实”，Agent 就能按任务需要调用静态分析工具、核对规则与证据，并输出中文诊断结果。

**本作业想展示的不是“用 AI 写一个 Python 脚本”，而是让 Agent 依据 Skill 指令，完成任务识别、工具编排、事实引用和受约束的总结。** 确定性检查由 Python 负责，Agent 不凭空判断硬件兼容性。

| 项目信息 | 内容 |
| --- | --- |
| 作业类型 | **AIGC 通识课程个人 Skill 开发** |
| 项目名称 | **QuantScout-Skill** |
| 实际应用 | 地平线 **RDK X5（Bayes-e）** 的 ONNX 模型量化前静态检查 |
| 当前版本 | **V1.5-S1.1**；Python 包 `0.5.0`；Analysis Schema `1.3` |
| 使用方式 | **自然语言 + 终端**，不需要网页界面 |
| 运行环境 | Ubuntu / WSL，Python 3.10+；静态检查**无需 GPU、Docker 或开发板** |

## 一、教师验收入口

**不安装任何环境，也可以先查看已经提交的完整成果。**

| 验收内容 | 查看位置 |
| --- | --- |
| **Skill 怎样引导 AI 工作** | [Skill 入口 `SKILL.md`](.agents/skills/rdk-x5-onnx-doctor/SKILL.md) |
| **是否真的检查了 ONNX 模型** | [完整诊断报告](examples/demo-report/report.md) · [结构化分析数据](examples/demo-report/analysis.json) |
| **AI 是否参与了事实总结** | [独立 Agent 生成的摘要](examples/demo-report/summary.md) · [已校验的事实包](examples/demo-report/summary_facts.json) |
| **有没有真实 Agent 测试** | [Codex 独立会话验收记录](docs/validations/V1_5_S1_1_Agent_E2E.md) |
| **有没有自动化测试** | [GitHub Actions 实际运行](https://github.com/Stradlin1/QuantScout-Skill/actions/runs/38021532036) · [代表性测试代码](tests/test_summary_publish.py) |
| **如何自己复现** | [第五节：教师复现](#五教师复现步骤) · [完整安装教程](docs/GETTING_STARTED.md) |

## 二、我为什么选择这个题目？

在将神经网络部署到 RDK X5 时，需要先把训练模型导出为 **ONNX**（神经网络交换格式），再交给地平线工具链进行量化与编译。即使 ONNX 文件本身符合格式规范，也不代表所有计算节点都满足目标 **BPU**（神经网络加速单元）的静态约束。

以卷积算子为例：模型的卷积核尺寸可能符合 ONNX 规范，却超出当前已收录的硬件规则范围。人工核查往往需要反复翻阅模型节点、Tensor、算子文档和不同版本的工具链配置。

我希望用 Skill 解决三个具体问题：

1. **把重复的静态检查自动化。** 提取 OpSet、算子、节点、输入输出、Shape、Tensor 信息，并核对已有规则。
2. **让结论可追溯。** 当发现确定冲突时，保留具体节点 ID、规则 ID、实际值、期望范围和来源，而不只回答“模型有问题”。
3. **约束 AI 的不确定判断。** 未覆盖的算子不直接说成“不支持”，静态检查通过也不说成“已经成功部署”。

因此，它是一个**量化前诊断助手**，不是自动量化器、ONNX 编辑器或编译器。

## 三、项目实现了哪些功能？

| 已实现功能 | 能完成的工作 |
| --- | --- |
| ONNX 结构检查 | 读取并校验模型，提取主图节点、计算图连接、OpSet 和输入输出 |
| BPU 静态规则检查 | 对已注册算子核对具体约束，记录实际参数与规则来源 |
| 工具链 Profile 预检 | 比较模型 OpSet 是否匹配当前配置的版本条件 |
| Shape 与 Tensor 检查 | 跟踪已知/未知维度、有限静态推断、Tensor 理论原始字节数 |
| 节点查询与图追踪 | 定位异常节点，查询规则细节、输入输出依赖与 Tensor 信息 |
| 结构候选检测 | 为部分已有计算图模式提供人工复核线索，不修改模型 |
| 官方资料查证 | 必要时由 Agent 查阅官方文档或已保存的离线参考资料 |
| **Agent 事实总结** | 按自然语言要求生成“简短总结”“只列异常”“只看输入输出”三类结果 |

目前规则库包含 **14 类算子、66 条检查或复核条目**。规则条目数量不代表全部 ONNX 算子均已覆盖，也不代表真实 BPU 编译、运行结果。

### 核心问题：AI 到底做了什么？

项目分成两部分：

**Python 静态诊断器**负责解析模型、计算数字、执行规则、核对状态和输出可复核的 JSON。这一部分即使不使用大模型也可以独立运行。

**Agent Skill** 负责根据用户的自然语言选择任务、决定调用哪些工具、读取证据，以及组织合适的中文回答。例如：

```text
用户：“帮我检查这个 ONNX 模型”
    → Agent 选择完整诊断流程
    → 调用 analyze / preflight / inspect / 必要的资料查询
    → 使用真实节点和规则证据组织中文结论

用户：“只总结检查结果，不要分析和建议”
    → Agent 识别 summary_only 意图
    → Python 导出可信事实包（summary-facts）
    → Agent 选择事实、标题、顺序和受限连接语
    → Python 复核并发布摘要（summary-publish）
```

**V1.5-S1.1 的重点是“AI 参与总结，但不能擅自编造事实”。**

它支持三种意图：

| 用户说的话 | 模式 | 输出 |
| --- | --- | --- |
| “简短总结，只要事实。” | `overview` | 模型概况、四种检查状态、代表冲突、范围说明 |
| “只列出异常，不要建议。” | `anomalies` | 已确定的 FAIL、待验证与未覆盖，分类呈现 |
| “只看模型输入输出。” | `io` | Graph I/O 的名称、Shape、dtype 和省略数量 |

Agent 撰写草稿时必须引用 `{{FACT.ID}}`，不能直接手写不受检验的数字；`summary-publish` 会重新提取事实并核对来源 SHA256、引用和必要内容后才创建 Markdown 文件。当前只允许有限的中文连接语，**不是任意自由改写，也不声称从根本上消除了大模型幻觉**。

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

老师可以顺序查看：[`analysis.json` 中的原始事实](examples/demo-report/analysis.json) → [`report.md` 完整报告](examples/demo-report/report.md) → [Agent 生成的 `summary.md`](examples/demo-report/summary.md)。

## 五、教师复现步骤

### 方式 A：无需运行，直接查看证据

优先查看第一节的链接，即可核对 Skill 入口、静态报告、AI 摘要和独立测试记录。公开演示中，Agent 实际执行了 `summary-facts → 草稿 → summary-publish`，不是把 Python 的固定模板直接称为 AI 输出。

### 方式 B：终端复现静态检查

请在 Ubuntu / WSL 中执行以下命令（需要 Python 3.10+）。仓库已经提供模型，不需要启动 Docker。

```bash
git clone https://github.com/Stradlin1/QuantScout-Skill.git
cd QuantScout-Skill

python3 -m venv .venv
.venv/bin/python -m pip install -e .

# 检查 CLI 和规则库
.venv/bin/python -m rdkx5_doctor --help
.venv/bin/python -m rdkx5_doctor rules validate

# 新建一次诊断：如果输出目录已经存在，请换一个目录名
.venv/bin/python -m rdkx5_doctor analyze \
  --model examples/demo.onnx \
  --out reports/teacher-review

# 查看明确触发静态冲突的节点
.venv/bin/python -m rdkx5_doctor nodes \
  --analysis reports/teacher-review/analysis.json \
  --status VIOLATION

# 查看对应规则与参数
.venv/bin/python -m rdkx5_doctor inspect \
  --analysis reports/teacher-review/analysis.json \
  --node oversized_kernel
```

结果将保存在 `reports/teacher-review/`，主要包括 `analysis.json` 和 `report.md`。

### 方式 C：用 Codex 验收 Agent Skill

在 VS Code 的 Ubuntu / WSL 工作区打开本仓库，使用有本地终端权限的 Codex Agent，发送以下指令：

> 使用当前仓库的 `rdk-x5-onnx-doctor` Skill。读取 `reports/teacher-review/analysis.json`，简短总结检查结果，只要事实，不要原因分析和修改建议。将结果保存为同目录**新的** `summary.md`。已有 JSON 不要重新 analyze。

随后可以单独提出“只列出异常”和“只看输入输出”，分别保存为不同的**新文件名**，观察 Agent 是否选择正确模式。

验收时重点检查 Agent 是否真的读取 Skill、调用 `summary-facts` 和 `summary-publish`，以及输出是否与已有 JSON 的统计一致。

**如果没有 Codex，也可以检查 CLI 的确定性摘要，但这不等同于 AI 功能验收：**

```bash
.venv/bin/python -m rdkx5_doctor summary \
  --analysis reports/teacher-review/analysis.json
```

这条命令也会创建 `summary.md`；请在**还没有**同名文件时使用。默认拒绝覆盖已有摘要。完整说明见 [快速上手](docs/GETTING_STARTED.md)。

## 六、测试结果与实际边界

| 测试项目 | 已有记录 | 查看证据 |
| --- | --- | --- |
| Python 本地自动化测试 | 开发日志记录 **528 passed / 0 failed** | [V1.5-S1.1 实施日志](docs/QuantScout_V1_5_S1_1_Development_Log.md) |
| GitHub Actions | **Python 3.10 和 3.12 均成功**，执行安装、规则校验、pytest 和构建 | [在线 CI 运行记录](https://github.com/Stradlin1/QuantScout-Skill/actions/runs/38021532036) |
| 独立 Codex Agent 端到端测试 | **累计 11 次：10 PASS、1 FAIL** | [E01～E10 和复验记录](docs/validations/V1_5_S1_1_Agent_E2E.md) |
| 公开示例 | 已提交 Schema 1.3 报告与独立 Agent 摘要 | [公开演示核对](docs/validations/V1_5_S1_1_Public_Demo.md) |

**为什么会有一次 FAIL？** 第一次 E10 中，Agent 绕过专门的发布器，通过文件操作覆盖了已有摘要。之后加强了 Skill 的“目标已存在就停止”要求，并使用新的独立会话复验通过。失败被如实保留，说明**发布器本身的防覆盖检查不等于对 Agent 所有文件权限的绝对保护**。

GitHub Actions 只验证确定性程序逻辑；独立 Codex 测试用于验证真实自然语言工作流。项目尚未开展正式的“普通 Prompt 与 Skill”对照实验，不应声称当前 Skill 必然优于所有其他工作方式。

老师也可本地运行：

```bash
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest -q
```

## 七、已知限制和后续工作

**本项目刻意不做：** Docker 量化、`hb_mapper` 编译、ONNX 模型改写、板端运行，以及实际 FPS、精度或 BPU 分配预测。它只负责进入量化流程之前的静态诊断。

目前只统计 ONNX **主图节点**，未递归展开 If / Loop 等嵌套子图；`NOT_COVERED` 不代表硬件不支持，`NO_VIOLATION_FOUND` 也不是可部署保证。当前有 Codex 的真实验收记录，其他 Agent 尚未完成相同级别的验证。

**尚未实现、不属于当前交付的功能：** V1.5-S2 Tensor 资源来源细分类、V1.5-S3 量化敏感结构候选提示，以及正式的无 Skill / 有 Skill 对照实验。

## 八、项目文件与开发记录

```text
QuantScout-Skill/
├── .agents/skills/rdk-x5-onnx-doctor/  # Skill 入口、意图分支与参考契约
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

**项目总结：** QuantScout-Skill 将一个真实的硬件部署前检查任务组织成“Agent 根据自然语言编排工作 + Python 给出可复核事实”的工作流。通过可运行的代码、公开示例、独立 Agent 会话和 CI 测试，展示了 AI Skill 在实际任务中的使用方式，而非仅提供一个提示词模板。