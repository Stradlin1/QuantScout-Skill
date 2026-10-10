# 快速上手：安装、诊断与事实总结

适用版本：**V1.5-S1.1** · Python **3.10+** · 建议使用 Ubuntu 或 Windows VS Code + WSL Ubuntu。本文所有命令都在**仓库根目录的 Linux/WSL 终端**执行。

## 1. 安装

```bash
git clone https://github.com/Stradlin1/QuantScout-Skill.git
cd QuantScout-Skill
python3 -m venv .venv
.venv/bin/python -m pip install -e .

.venv/bin/python -m rdkx5_doctor --help
.venv/bin/python -m rdkx5_doctor rules validate
```

开发者额外运行自动化测试：

```bash
.venv/bin/python -m pip install -e '.[dev]'
env -u PYTHONPATH .venv/bin/python -m pytest -q
```

`env -u PYTHONPATH` 只清除当前命令的环境变量，适用于 ROS Python 路径影响依赖加载的情况，并非所有环境都必须使用。

无需 GPU、RDK X5 板卡、Docker 或 OpenExplorer 才能进行静态检查。

## 2. 运行第一份静态报告

仓库已经包含可用于测试的 `examples/demo.onnx`。`--out` 应指向**新目录**，避免覆盖已有报告。

```bash
.venv/bin/python -m rdkx5_doctor analyze \
  --model examples/demo.onnx \
  --out reports/my-first-run
```

主要输出：

```text
reports/my-first-run/
├── analysis.json        # 完整结构化事实，Schema 1.3
└── report.md            # 完整报告
```

当前只计数已解析的主图节点。嵌套子图没有被逐节点展开。

### 核对工具链 Profile（可选）

```bash
.venv/bin/python -m rdkx5_doctor preflight \
  --analysis reports/my-first-run/analysis.json \
  --profile .agents/skills/rdk-x5-onnx-doctor/references/toolchain_profile_opset11.yaml \
  --json > reports/my-first-run/preflight.json
```

`MATCH` 仅表示满足提供的 Profile 的标准域 OpSet 条件；不表示已通过实际编译。该 Profile 是作者针对 OpenExplorer v1.2.8 使用的配置，不能推广为所有工具链的版本要求。

### 查看具体证据

```bash
.venv/bin/python -m rdkx5_doctor nodes \
  --analysis reports/my-first-run/analysis.json --status VIOLATION

.venv/bin/python -m rdkx5_doctor inspect \
  --analysis reports/my-first-run/analysis.json --node oversized_kernel

.venv/bin/python -m rdkx5_doctor trace \
  --analysis reports/my-first-run/analysis.json --node main/node_000000
```

更多子命令：`shapes` / `shape`、`tensors` / `tensor`、`candidates` / `candidate`、`rules list`。

## 3. 事实总结：Agent 与纯 Python 的区别

### A. 让 Agent 撰写事实摘要（推荐演示）

在当前仓库打开 VS Code 的 Codex Agent，会话能读到 `.agents/skills/rdk-x5-onnx-doctor/SKILL.md` 和 Python 工具。

可直接提出：

> 使用 QuantScout 的事实总结模式。读取 `reports/my-first-run/analysis.json`，简短总结检查结果，只要事实、不要原因分析和建议。把结果安全保存为同目录**新的** `summary_overview.md`。不要重新 analyze。

或：

> 只列出当前报告中已确定的静态冲突、待验证与未覆盖记录。不要给优化建议。保存为同目录 `summary_anomalies.md`。

或：

> 只总结 ONNX graph 的输入与输出名称、Shape、dtype；未知项照实保留，不推断 NCHW 等布局。保存为同目录 `summary_io.md`。

Agent 应根据指令分别选择 `overview`、`anomalies`、`io`。实际执行链是：

```text
analysis.json
   └── summary-facts --mode overview/anomalies/io
           └── summary_facts.json（可信事实 ID 和来源）
                   └── Agent 组织 summary_draft.json
                           └── summary-publish
                                   └── 新的 summary_*.md
```

发布器会重算当前事实，检查事实包的来源哈希、模式、必需引用和允许句式。**不能用人工伪造的数字或额外建议绕过发布校验。** 但这种受限检查不是对任意中文内容的形式化语义证明。

事实包导出可独立通过 CLI 查看：

```bash
.venv/bin/python -m rdkx5_doctor summary-facts \
  --analysis reports/my-first-run/analysis.json \
  --mode io --limit 2 --json
```

需要真实保存 Agent 摘要时，按照 [summary_only 契约](../.agents/skills/rdk-x5-onnx-doctor/references/summary_only_contract.md) 组织草稿并调用 `summary-publish`；不要用旧 `summary` 命令冒充 Agent 生成。`summary-facts` 和 `summary-publish` 只读已有分析数据，不会重新分析 ONNX。

### B. 不使用 Agent 的确定性摘要

```bash
.venv/bin/python -m rdkx5_doctor summary \
  --analysis reports/my-first-run/analysis.json
```

会生成 `reports/my-first-run/summary.md`，这是**Python 固定结构的简版事实输出，不是 AI 撰写**。

`--detailed` 可显式选择历史七节详细格式，但须使用**新的**分析输出目录（已有 `summary.md` 不覆盖）。普通 `summary` 同样只支持经验证的 Schema 1.3。

## 4. 输出目录与保护规则

```text
reports/my-first-run/
├── analysis.json
├── report.md
├── preflight.json           # 仅实际执行 preflight 后存在
├── summary_facts.json       # 仅实际请求导出事实包时存在
├── summary_draft.json       # 仅 Agent 实际编写草稿时存在
├── summary_overview.md      # 示例：由 Agent 事实发布器生成
├── summary_anomalies.md     # 示例：由 Agent 事实发布器生成
└── summary_io.md            # 示例：由 Agent 事实发布器生成
```

以上仅展示**可能出现的文件**，不会一次自动生成全部文件。`reports/` 由 Git 忽略，不应上传个人模型或私有检查报告。

文件已存在或是符号链接时，事实发布器拒绝覆盖。曾发生真实 E10 Agent 绕开发布器替换文件的失败案例，现已增加“检查目标后早停”的 Skill 契约和复验；通用 Agent 文件权限仍不能提供 OS 级的绝对保护。

## 5. 常见问题

**已有 `analysis.json`，还需要模型文件吗？** 事实总结不需要。当前事实包只接受 Schema 1.3。旧版快照不能通过伪造字段直接升级。

**`NOT_COVERED` 是否代表 X5 不支持？** 不代表。它仅说明当前规则库没有适用的规则。

**`NO_VIOLATION_FOUND` 是否代表可部署？** 不代表。没有覆盖或没有执行的编译、量化步骤仍未知。

**理论 Tensor 字节数是否等于真实硬件内存占用？** 不等于。它是根据已知 Shape 与 dtype 计算的原始载荷。

**为什么不能随意修改摘要句子？** Agent 仅能组合已验证的事实引用和受限连接语；自由扩写可能引入未经验证的结论，因此发布器会拒绝。

**其他 Agent 能用吗？** 结构遵循 Agent Skills 约定，但目前仅有本项目的 Codex 端到端验收。迁移需同时安装 Python 包和引用资料，不能只复制 `SKILL.md`。

详见 [README](../README.md)、[项目概览](PROJECT_OVERVIEW.md) 和 [验证现状](VALIDATION_STATUS.md)。
