# RDK X5 ONNX Doctor

**AIGC 通识课程个人结课作业 · 面向 RDK X5 的 ONNX 量化前诊断 Skill**

这是我为 AIGC 通识课程制作的一个 **AI Skill**。项目来自实际使用 RDK X5 进行模型量化时遇到的一个问题：把 ONNX 交给量化工具链之前，往往需要先弄清楚模型的版本、算子、张量形状以及哪些地方可能不符合硬件约束。

我希望把这些重复的排查工作交给 AI：**只要告诉 Codex“帮我检查这个 ONNX”，Skill 就能按步骤调用检测工具，必要时查询地平线官方手册，并用中文解释检查结果。**

本项目**专门面向地平线 RDK X5（Bayes-e）的 ONNX 量化流程**，不是通用的 ONNX 编辑器，也不是一键量化软件。它更像是量化前的“体检助手”：在模型进入地平线量化工具链之前，先发现和解释可能影响转换的静态问题，帮助使用者决定下一步应该核实什么。

- **交互方式：** Windows VS Code + WSL Ubuntu + Codex 插件；全程使用对话与终端，无网页界面。
- **当前版本：** Skill V1.4；Python 辅助包 `rdkx5-onnx-doctor` 0.5.0。
- **硬件与工具链目标：** 地平线 RDK X5（Bayes-e）；本人使用 **OpenExplorer v1.2.8**，当前项目工具链 Profile 按 **ONNX Opset 11** 检查。
- **基本原则：** 只读 ONNX、有据可查、不把静态预检等同于量化或部署成功。

**我使用的地平线量化工具链：** OpenExplorer v1.2.8，目录/发行包名称为：

```text
horizon_x5_open_explorer_v1.2.8/
└── horizon_x5_open_explorer_v1.2.8-py310_20240926/
```

这是本项目实际使用的量化环境版本信息，不表示 Skill 会直接调用该目录中的量化程序。**OpSet 11 是我当前工具链配置的目标版本**，不代表所有 RDK X5 量化工具链都只支持 Opset 11。

## 一、这个 Skill 能解决什么问题？

### 先认识三个概念

| 名词 | 通俗解释 |
| --- | --- |
| **ONNX** | 一种常用的神经网络模型交换格式。模型导出后，里面记录了计算节点、运算关系、参数与输入输出信息。 |
| **量化** | 把模型中的部分计算和数据表示转换为更适合目标硬件的形式，例如使用 INT8。并非所有模型都能直接转换成功。 |
| **BPU** | RDK X5 上的神经网络计算加速单元。ONNX 本身合法，并不代表所有算子都满足 X5 BPU 的具体限制。 |

例如，一个卷积层在 ONNX 规范中可能完全合法，但它的卷积核大小、步长或其他条件不一定符合目标 BPU 的限制。过去需要逐项翻查文档，现在可以让 Skill 先做静态检查，再解释依据和不确定的地方。

### 工作原理：AI 负责判断步骤，脚本负责可靠计算

~~~text
用户用自然语言提出问题
          ↓
Codex 读取 SKILL.md，确定本次检查目标
          ↓
调用本地 Python 工具读取 ONNX / 检查 Opset / 分析计算图
          ↓
对照经过审核的 X5 BPU 规则，定位需要关注的节点
          ↓
遇到未覆盖算子 → 按需查询官方资料或离线手册
          ↓
AI 整理证据、解释问题，生成中文结论及本地报告
~~~

**Skill 是整个流程的主控**；Python、YAML 规则库和官方文档是它使用的辅助资源。这样既能利用 AI 理解自然语言、组织调查的能力，又能让具体的数值判断具有可复现的依据。

## 二、目前已经实现的功能

| 功能 | Skill 会做什么 | 为什么有用 |
| --- | --- | --- |
| **模型基础体检** | 验证 ONNX 结构、读取输入输出、Opset 和节点信息 | 先区分“模型文件本身有问题”与“目标硬件可能不兼容” |
| **当前工具链准入检查** | 对照用户提供的 Profile 检查是否为 Opset 11，返回 MATCH / MISMATCH / UNKNOWN | 尽早发现不符合当前项目配置的版本 |
| **BPU 算子约束检查** | 用审核过的规则检查 Conv、Mul、Add、Pool 等节点的可知条件 | 找到具体节点、实际参数、限制值和出处 |
| **官方知识查询** | 对未覆盖算子按需查阅 X5 官方手册；官网不可用时可查询有版本记录的离线资料 | 避免仅凭 AI 印象猜测算子是否受支持 |
| **Shape 与 Tensor 分析** | 查看中间张量形状、未知维度原因和可计算的理论字节数 | 帮助理解模型的数据流和资源规模 |
| **计算图追踪** | 查看问题节点的上下游依赖与可到达的模型输出 | 知道问题发生在模型哪里、与哪些输出相关 |
| **结构优化候选** | 识别部分无效变换、可疑重复操作及 Conv-BN 融合审查候选 | 为回到原训练/导出工程调整模型提供线索 |
| **报告与解释** | 输出 `report.md`、`analysis.json`，必要时另存版本预检与官方查询证据 | 结果可查看、可复核，而不只有一句“能/不能量化” |

目前规则库覆盖 **14 类算子、66 条自动检查或人工复核条目**（包括非阻塞审查项）。这不代表已经检查了 ONNX 中所有算子，也不代表通过规则的节点一定会被实际放到 BPU 上运行。对于 Relu、Transpose 等没有新增专属数值限制的算子，Skill 可以提供文档解释，但不会凭空制造“全部通过”的规则。

官方资料会区分 **X5 ONNX BPU 条款、CPU 条款、编译器转换说明**，并标注本次是否真正联网获取，还是使用已缓存的文档。仓库还保留了一个固定 Git 版本的[官方算子表离线快照](references/offline_manual/README.md)，供网络不可用时查阅。

### 一个简单的诊断例子

仓库里的 `examples/demo.onnx` 是专门准备的演示模型：它的 Conv 使用了高度为 **32** 的卷积核，而现有 X5 Conv2D 规则规定相应范围为 **1～31**。

因此，Skill 能指出：

> ONNX 结构检查可以通过，但某个 Conv 节点的卷积核高度超出了已收录的 X5 BPU 静态约束。需要回到模型设计或导出阶段检查该结构；尚未实际运行量化工具链，不能直接断言编译器最终怎样处理它。

这个例子可以看出：**“ONNX 合法”与“满足目标 BPU 规则”是两件不同的事。**

## 三、安装与使用

### 1. 推荐环境

本项目的主要使用方式是：

**Windows 上运行 VS Code → 通过 WSL 扩展打开 Ubuntu 工作区 → 使用 VS Code 的 Codex 插件调用 Skill。**

**为什么使用 Ubuntu 或 WSL？** 因为我的完整 RDK X5 量化工作流需要在 **Ubuntu/Linux 环境中配置并使用地平线 OpenExplorer 的 Docker 量化环境**。我在 Windows 上使用 WSL Ubuntu，是为了同时保留 Windows VS Code + Codex 的操作方式，并让 ONNX 文件、检查脚本与后续 Docker 量化流程处于衔接方便的 Linux 工作环境。使用原生 Ubuntu 也可以。

这里要区分两个阶段：

- **本仓库目前实现的 Skill：** 在 Docker 量化之前读取 ONNX、检查算子和模型结构；**不要求启动 Docker，也不会执行量化**。
- **后续实际量化阶段：** 在配置好的 Ubuntu Docker / OpenExplorer v1.2.8 环境内执行校准、模型转换等工作；**不属于本 Skill 目前的执行范围**。

因此，选择 Ubuntu/WSL 主要是为了对接实际量化工具链，**不是说 ONNX 静态分析本身必须依赖 Docker**。

需要准备：

- Windows VS Code，能够连接 WSL Ubuntu，并已安装、配置 Codex 插件。
- WSL Ubuntu 内的 Python **3.10 或更新版本**、Git 和 Python 虚拟环境工具。
- 一个准备检查的 `.onnx` 文件。**仅运行本 Skill 的静态预检时**，不需要 GPU、RDK X5 开发板、Docker 或已安装的地平线量化工具链。

以下命令均在 **WSL Ubuntu 终端**执行，不是在 Windows PowerShell 中执行。

### 2. 克隆仓库并安装 Python 辅助工具

~~~bash
git clone https://github.com/Stradlin1/skillzuoye.git
cd skillzuoye

python3 --version
python3 -m venv .venv
.venv/bin/python -m pip install -e .
~~~

如果提示缺少 `venv`，可在 Ubuntu 中先安装 `python3-venv`；如需运行开发测试，可额外执行 `.venv/bin/python -m pip install -e '.[dev]'`。

检查安装结果：

~~~bash
.venv/bin/python -m rdkx5_doctor --help
.venv/bin/python -m rdkx5_doctor rules validate
~~~

项目使用的 Codex Skill 位于：

~~~text
.agents/skills/rdk-x5-onnx-doctor/
├── SKILL.md
├── references/
└── scripts/
~~~

**不需要单独创建 Skill 文件**，克隆仓库后它已经在正确的目录中。

### 3. 在 Windows VS Code 中打开 WSL 项目

在 WSL 的仓库根目录运行：

~~~bash
code .
~~~

确认 VS Code 正在使用 **WSL 工作区**，再打开 Codex 插件。Codex 需要能够读取当前仓库并调用 WSL 终端中的 Python 工具。

### 4. 用自然语言调用 Skill（推荐）

首次演示可先生成仓库自带的测试模型：

~~~bash
.venv/bin/python examples/generate_demo.py
~~~

然后在 **VS Code 的 Codex 对话框**输入：

> 帮我检查 `examples/demo.onnx` 是否适合当前 RDK X5 量化工具链。请使用仓库中的 ONNX Doctor Skill，检查 Opset、BPU 静态约束，解释主要问题，并把报告保存在本地 `reports/`。不要修改模型或执行量化。

也可以通过 Codex 的 Skill 选择器**显式选择** `rdk-x5-onnx-doctor`，再输入同样的任务。正常使用时不必记住每一条 Python 子命令，Skill 会根据任务选择所需步骤。

检查自己的模型时，把演示路径改为 WSL 可以访问的模型路径，例如：

> 用 RDK X5 ONNX Doctor 检查 `/home/你的用户名/models/my_model.onnx`。先确认是否满足我目前 Opset 11 的工具链配置，再说明哪些算子存在明确的静态约束冲突、哪些需要查询官方资料。不要改写 ONNX。

还可以提出更具体的问题，例如“这个 Mul 为什么被标记为异常？”、“哪个中间 Tensor 比较大？”、“帮我查 Shape 算子在 X5 官方手册中的处理说明”。

### 5. 不使用 Codex 时，也可以直接通过终端检查

Skill 的底层工具可以独立运行，方便复现检查结果：

~~~bash
# 对演示 ONNX 做只读分析
.venv/bin/python -m rdkx5_doctor analyze \
  --model examples/demo.onnx \
  --out reports/demo

# 按当前工具链 Profile 检查 Opset
.venv/bin/python -m rdkx5_doctor preflight \
  --analysis reports/demo/analysis.json \
  --profile .agents/skills/rdk-x5-onnx-doctor/references/toolchain_profile_opset11.yaml \
  --json

# 查看有静态约束冲突的节点
.venv/bin/python -m rdkx5_doctor nodes \
  --analysis reports/demo/analysis.json \
  --status VIOLATION

# 追踪一个节点与模型输出的关系
.venv/bin/python -m rdkx5_doctor trace \
  --analysis reports/demo/analysis.json \
  --node main/node_000000
~~~

如果官网不可访问，仍可从仓库保留的官方原文快照查询算子条目：

~~~bash
.venv/bin/python references/offline_manual/query.py \
  --operator Shape --json
~~~

离线查询只是查文档，不代表自动证明该算子在真实模型中由 BPU 执行。

### 6. 结果保存在哪里？

默认在指定的 `reports/<运行目录>/` 下保存 `report.md` 和 `analysis.json`；Skill 进行完整预检时还可能保存 `preflight.json`、`official_lookup.json` 等独立证据。

`reports/` 默认被 Git 忽略，个人模型与诊断报告不会因为普通提交而自动进入仓库。历史版本、规则来源与实现细节可以进一步阅读：

- [Skill 主控文件](.agents/skills/rdk-x5-onnx-doctor/SKILL.md)
- [V1.4 开发记录](docs/RDK_X5_ONNX_Doctor_V1_4_Development_Log.md)
- [官方资料来源说明](references/official_sources.md)
- [多模型回归验收总结](docs/MULTI_MODEL_VALIDATION.md)

### 使用前需要知道的边界

当前的 **OpSet 11 是本项目用户 Profile 的要求**，不代表所有版本的 RDK X5 工具链只能使用 Opset 11。`MATCH` 只表示版本条件匹配，`MISMATCH` 时仍可进行通用 ONNX 结构分析。

本 Skill **不执行 hb_mapper、量化校准、模型编译或板端推理**，也不修改 ONNX。因此，它不能保证模型最终成功部署，不能预测真实 FPS、BPU/DDR 内存占用或量化精度。检测结果中的 `UNKNOWN`、`NOT_COVERED` 和“需要人工验证”都是正常而重要的结论，不能当作自动通过。

## 四、接下来准备完善的功能

以下属于 **后续计划，当前 V1.4 尚未实现**：

1. **更直观的资源分类：** 将中间特征图、模型权重与 Shape 参数分开统计。对于动态维度，明确说明“数据量尚无法确定”，避免把已知的 32 B Shape 参数误解为模型中最大的特征图。
2. **更适合初学者的智能诊断摘要：** 不只是列规则和节点，还能把相同原因的问题归纳起来，用更容易理解的语言说明“发现了什么、依据是什么、建议先核实什么”。
3. **继续完善 Opset 11 的基础算子知识：** 结合实际模型中尚未覆盖的算子，优先补充经过官方核实的规则；不为了增加数量而编造限制。
4. **优化课程演示与使用体验：** 准备更清晰的真实模型示例、简洁的诊断结果和必要的使用说明，让第一次接触 RDK X5 的同学也能理解这个 Skill 在做什么。

长期来看，可以在独立工作流中接入真实量化工具链，比较静态诊断与编译器结果；但这不属于当前的 ONNX 量化前检查 Skill，也**不是已经完成的功能**。
