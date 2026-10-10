# 项目介绍：QuantScout-Skill

> AIGC 通识课程个人结课作业 · RDK X5（Bayes-e）ONNX 量化前静态诊断 Skill  
> 当前实现：**V1.5-S1.1**（2026-10-10）；Python 包 `rdkx5-onnx-doctor` **0.5.0**，Analysis Schema **1.3**。

## 1. 选题背景

从 PyTorch 等训练框架导出 ONNX 后，开发者通常需要检查模型格式、OpSet、输入输出、算子参数和计算图结构。对于 RDK X5，ONNX 语义合法并不足以证明某个节点满足当前 BPU 静态约束，更不能直接推断量化精度或实际运行位置。

QuantScout-Skill 针对**进入量化工具链之前**的这段工作：提供可重复的离线静态检查，再让 Agent 根据自然语言任务组织工具调用和证据表达。开发者仍自行在 OpenExplorer 环境中完成正式量化、编译和板端验证。

## 2. 产品定位

**输入**：本地 ONNX 模型，或已有的 `analysis.json`（事实摘要要求经验证的 Schema 1.3）。

**输出**：`analysis.json`（结构化诊断证据）、`report.md`（完整分析报告）、可选的 `preflight.json`、按需要生成的 `summary.md` 或其他明确授权的新 Markdown 文件。

**交互方式**：面向具备终端能力的 Agent 的自然语言交互，辅以可独立执行的 Python CLI；无需网页 UI、独立云 API Key、Docker 或板端设备。

**重点边界**：只读 ONNX；当前不执行模型改写、量化、编译或部署。读取的是**主图节点**；嵌套子图尚未逐节点展开。

## 3. 技术架构

```text
自然语言
  ├─ 完整预检 ─> Agent 选择 analyze / preflight / inspect / trace / 官方查询
  │                └─ Python 静态检查 + 规则来源 ─> JSON / 完整报告
  └─ 事实总结 ─> Agent 选择 overview / anomalies / io
                   └─ Python 导出已验证的有界事实（summary-facts）
                         └─ Agent 选择事实、标题、顺序与受限连接语
                               └─ Python 重新校验并发布（summary-publish）
```

### Python 和 Agent 的分工

| Python 负责 | Agent 负责 |
| --- | --- |
| 读取 ONNX、验证 Schema、计算节点与 Tensor 统计 | 理解“检查模型”“只列异常”“仅看输入输出”等意图 |
| 根据版本化 YAML 检查有限 BPU 条款 | 按需选择诊断/查询命令，引用真实规则证据 |
| 对 `summary_facts.json` 绑定数据源 SHA 和固定事实 ID | 在 `summary_draft.json` 中组织事实引用和中文表达 |
| 发布前重算事实、检查必须项与禁用自由扩写 | 审读结果是否符合用户“只要事实”的要求 |

V1.5-S1.1 为控制幻觉，采用 `{{FACT.ID}}` 引用和受限句式。**这不是通用中文生成器，也不保证任意自由文本的语义正确性。** 纯 Python `summary` 是确定性 fallback，与 Agent 撰写的工作流需要区分。

## 4. 已完成阶段

| 阶段 | 主要交付 |
| --- | --- |
| V1 ～ V1.3 | 主图解析、Conv 与多算子静态规则、Shape 证明、Tensor 理论字节数、计算图节点/资源查询、部分结构候选 |
| V1.4 | Agent 工作流、当前 Profile 的标准域 OpSet 比对、按需查阅官方资料和离线手册 |
| V1.5-S1 | 从 Schema 1.3 导出确定性的事实摘要 |
| **V1.5-S1.1** | **三种自然语言摘要模式、Agent 草稿与机器核验发布、精简演示、真实 Agent 验收、CI** |

当前规则库包含 **14 类算子、66 条检查或复核条目**，并不代表全算子覆盖。V1.5-S2 的 Tensor 来源细分类和 V1.5-S3 的量化敏感结构审查尚处于规划阶段。

## 5. 示例与证据

公开 `demo.onnx` 是 5 个主图节点的微型模型。Schema 1.3 的静态结果为：

- `NO_VIOLATION_FOUND`：2 个节点；
- `VIOLATION`：1 个节点；
- `NEEDS_VERIFICATION`：0 个节点；
- `NOT_COVERED`：2 个节点。

其中一个 Conv 节点的 `kernel_h=32` 超出当前规则记录的 1～31 范围。这个案例只证明检查器能够按规则发现静态冲突，**不证明地平线编译器已经拒绝该模型**。

演示入口：[模型与报告](../examples/demo-report/analysis.json) · [Agent 事实摘要](../examples/demo-report/summary.md) · [演示核对记录](validations/V1_5_S1_1_Public_Demo.md)。

## 6. 测试与公开披露

- 开发日志记录：**528 个本地 pytest 通过**。
- 对提交 `60b81f5` 的 [GitHub Actions 运行](https://github.com/Stradlin1/QuantScout-Skill/actions/runs/38021157135) 已完成：Python 3.10、3.12 两个作业均成功。
- 累计独立 Codex 验收 **11 次会话：10 PASS、1 FAIL**。首次 E10 确实覆盖了已有摘要，修正契约后使用新会话复验通过。必须保留失败记录；不能把该结果宣传成绝对文件安全。
- 未有正式的无 Skill/有 Skill 对照实验；尚未验证其他 Agent 的完整流程；未进行 hb_mapper、量化或板端实测。

完整验证快照：[VALIDATION_STATUS.md](VALIDATION_STATUS.md)。

## 7. 作为 AIGC 作业的展示方式

一次完整演示可以分为两步：首先通过自然语言要求 Agent 检查 `examples/demo.onnx` 并生成结构化报告；接着要求它**只利用已有的 `analysis.json`** 生成 `overview`、`anomalies` 或 `io` 的事实摘要，检查是否正确避免重复运行 analyze，并确认发布器拒绝错误事实和无依据扩写。

课程评价应区分**确定性 Python 静态诊断的工程能力**与**Agent 自主选择工具、事实绑定和语言组织的 Skill 能力**。目前已有独立 Agent 工作流验证，但尚未通过受控对照实验证明它必然优于普通 Prompt。

进一步说明：[快速上手](GETTING_STARTED.md) · [Skill 入口](../.agents/skills/rdk-x5-onnx-doctor/SKILL.md) · [文档目录](README.md)。
