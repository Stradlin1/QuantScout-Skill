# 当前验证状态与能力边界

**核验时间：2026-10-10（Asia/Singapore，UTC+08:00）**  
**被核验的代码基线：** [`60b81f5`](https://github.com/Stradlin1/QuantScout-Skill/commit/60b81f55edce40c461e44e43e41af727dfa7990b)  
**功能版本：** V1.5-S1.1 · Python 包 0.5.0 · Analysis Schema 1.3

本文是**状态快照**，不是持续自动更新的 CI 看板。后续文档提交或代码提交应以新的工作流运行状态为准。历史开发日志可能保留“尚未触发 CI”，表示**开发完成当时**的状态，而非本快照。

## 1. 可核验的测试记录

| 范围 | 结果 | 证据和限定 |
| --- | --- | --- |
| 开发环境本地 pytest | **528 passed / 0 failed** | [实施日志](QuantScout_V1_5_S1_1_Development_Log.md) 记录；此处并未独立重跑 |
| GitHub Actions · Python 3.10 | **SUCCESS** | 对提交 `60b81f5` 的 [运行 #1](https://github.com/Stradlin1/QuantScout-Skill/actions/runs/38021157135)，包含安装、CLI help、规则验证、测试、构建 |
| GitHub Actions · Python 3.12 | **SUCCESS** | 同一次 GitHub Actions，独立矩阵作业通过 |
| 真正的独立 Codex 会话 | **累计 11 次：10 PASS、1 FAIL** | [逐案验收记录](validations/V1_5_S1_1_Agent_E2E.md) |
| 公开微型模型 | **Schema 1.3**，可复现 | [演示核验](validations/V1_5_S1_1_Public_Demo.md) 与 [analysis.json](../examples/demo-report/analysis.json) |
| 非 Codex Agent 跨客户端验收 | **未开展** | 声称支持 Skill 发现不等于本项目完整验证 |
| OpenExplorer / hb_mapper / Docker / 板端验证 | **未执行** | 本工具的实际范围只有静态检查 |

GitHub Actions 只运行离线确定性 Python 检查及构建，不调用外部 LLM 来测试摘要生成。**CI 通过不代表真实 Agent 的自然语言路由必然稳定。**

## 2. 真实 Agent 事实摘要的测试情况

V1.5-S1.1 公开了 10 个不同场景的独立 Codex 测试说明，覆盖：

- 三种意图 `overview` / `anomalies` / `io`；
- 已有 JSON 的复用，以及仅给 ONNX 时执行一次 analyze；
- 无确定 FAIL 时不能宣传“完全兼容”；
- 用户要求禁联网、输入包含恶意模型/节点名称；
- 已有 `summary.md` 的防覆盖约束。

首次 E01～E10 为 **9 PASS / 1 FAIL**。E10 实际绕过 `summary-publish`，用临时文件替换了受保护的摘要。修改 Skill 指令和发布前检查后，新增独立 E10_retry 为 **PASS**，因此累计为 **11 次：10 PASS / 1 FAIL**。

**仍然存在的边界**：Python 发布器可拒绝自己执行的覆盖，但 Agent 仍可能持有其他文件写入权限；复验通过不是操作系统级权限隔离或“永远不会覆盖”的证明。脱敏后的逐案说明已公开；原始会话日志仍在被 Git 忽略的本地 `reports/`，目前不能从仓库直接复核全部原始工具事件。

## 3. 事实总结的校验内容

`summary-facts` 校验 Schema 1.3、节点/规则/统计一致性、输入输出元信息，输出包含来源 SHA、模式与允许事实 ID 的有界事实包。

Agent 选择事实、标题、顺序和有限连接语；`summary-publish` 重新提取事实并逐项核对，校验 `{{FACT.ID}}` 的引用、必需事实和输出路径，再创建新的 Markdown。

其限制包括：

- **受限句式而非自由自然语言生成**，不能把机器校验宣传成任意中文全语义无幻觉证明。
- `summary-facts` 新流程当前不读取官方查询 sidecar；原来的 `summary --official-lookup` 接口仍保留。
- 当前只分析主图节点，**嵌套子图未逐节点展开**，不能推断已检查全部子图内部算子。
- Tensor 载荷计算基于已知 Shape 与 dtype，不是真实 BPU/DDR/SRAM 或峰值内存。
- Profile `MATCH` 只表示当前用户配置下的 OpSet 条件匹配，不是实际编译证明。
- `NOT_COVERED` / `NEEDS_VERIFICATION` / `UNKNOWN` 不能混同为 `FAIL` 或 `PASS`。

## 4. 公开示例的实际数字

`examples/demo.onnx` 的 **5 个已解析主图节点**分别为：

| 最终状态 | 节点数 |
| --- | ---: |
| `NO_VIOLATION_FOUND` | 2 |
| `VIOLATION` | 1 |
| `NEEDS_VERIFICATION` | 0 |
| `NOT_COVERED` | 2 |

`main/node_000000`（`Conv`）触发 `X5-CONV2D-KERNEL-H`：`kernel_h=32`，规则记录范围为 1～31。唯一违规节点 **1**，FAIL 规则记录 **1**。这是人工构造的小模型，不代表真实大模型上的误报率、精度或硬件部署能力。

## 5. 接下来应补充的验证

**尚未完成、不得写成既有结论：** 在相同模型、相同 Prompt 与环境下开展“无 Skill / 仅 CLI / 使用 Skill”对照实验，记录事实准确率、工具调用与任务完成情况；对若干真实 ONNX 与手工获取的官方工具链检查记录进行一致性比对；以及其他 Agent 的端到端兼容性测试。

建议从 [快速上手](GETTING_STARTED.md) 复现公开微型模型，再阅读 [Codex E2E](validations/V1_5_S1_1_Agent_E2E.md) 与 [V1.5-S1.1 实施日志](QuantScout_V1_5_S1_1_Development_Log.md)。

**本项目不执行 Docker 量化、`hb_mapper` 编译或板端运行，不预测模型实际量化精度、FPS、延迟或 BPU 分配。**
