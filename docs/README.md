# QuantScout-Skill 文档导航

从仓库 [README](../README.md) 开始，按照自己的需求进入相应文档。**当前功能快照：V1.5-S1.1；Python 0.5.0；Analysis Schema 1.3（2026-10-10）。**

## 面向使用者与课程评审

| 文档 | 内容 |
| --- | --- |
| [项目介绍与技术架构](PROJECT_OVERVIEW.md) | 选题背景、Skill 与 Python 的职责、已实现内容和 AIGC 展示方式 |
| [快速上手](GETTING_STARTED.md) | 安装、终端静态检查、Codex 三模式事实总结、安全输出与常见问题 |
| [当前验证状态](VALIDATION_STATUS.md) | 本地测试、GitHub Actions、真实 Agent E2E、公开示例、尚未验证的能力 |
| [Skill 工作流入口](../.agents/skills/rdk-x5-onnx-doctor/SKILL.md) | Agent 实际按此识别意图并组织工具 |
| [事实总结契约](../.agents/skills/rdk-x5-onnx-doctor/references/summary_only_contract.md) | `overview` / `anomalies` / `io`、草稿格式、复核与防覆盖要求 |
| [公开摘要](../examples/demo-report/summary.md) | 独立 Codex 生成、可在浏览器中阅读的微型模型事实摘要 |

## 真实验证与研发记录

| 文档 | 内容 |
| --- | --- |
| [V1.5-S1.1 Codex 端到端验收](validations/V1_5_S1_1_Agent_E2E.md) | 首次 9 PASS / 1 FAIL、E10 复验与限制 |
| [公开演示核验](validations/V1_5_S1_1_Public_Demo.md) | 实际模型、结构化数字、Agent 调用与演示复现 |
| [V1.5-S1.1 实施日志](QuantScout_V1_5_S1_1_Development_Log.md) | 开发阶段完成项和本地测试；历史状态按开发当时记录 |
| [V1.5-S1.1 正式开发规范](QuantScout_V1_5_S1_1_Codex_Development_Spec.md) | 曾用于指导 Codex 的需求与验收条款 |
| [多模型验证记录](MULTI_MODEL_VALIDATION.md) | 旧版本静态覆盖实验；不能当作实际编译验证 |

## 技术格式与规则证据

- [Analysis Schema 1.3](ANALYSIS_SCHEMA_V1_3.md)；历史 [1.2](ANALYSIS_SCHEMA_V1_2.md) / [1.1](ANALYSIS_SCHEMA_V1_1.md)。
- [报告策略 V1.3](REPORT_POLICY_V1_3.md)、[终端版规范](TERMINAL_V1_SPEC.md)。
- [官方资料来源](../references/official_sources.md)、[离线手册说明](../references/offline_manual/README.md)、[基础算子审核记录](../references/x5_opset11_basic_ops_review.md)。

## 历史开发规范

- [V1 初版规范](RDK_X5_ONNX_Doctor_V1_Development_Spec.md)
- [V1.1 规范](RDK_X5_ONNX_Doctor_V1_1_Development_Spec.md)
- [V1.2 多算子规范](RDK_X5_ONNX_Doctor_V1_2_Multi_Op_BPU_Development_Spec.md)
- [V1.3 Shape 与注意力算子规范](RDK_X5_ONNX_Doctor_V1_3_Shape_Inference_And_Attention_BPU_Development_Spec.md)
- [V1.4 Agent 编排与官方证据规范](RDK_X5_ONNX_Doctor_V1_4_Skill_Orchestration_Opset11_And_Official_Knowledge_Development_Spec.md)
- [V1.4 开发日志](RDK_X5_ONNX_Doctor_V1_4_Development_Log.md)

**关于状态说明：** 历史规范是开发要求，历史日志是当时结果。例如 V1.5-S1.1 日志中的远程 CI `NOT_RUN` 是提交前的实际情况；随后对代码基线 `60b81f5` 的 GitHub Actions [运行 #1](https://github.com/Stradlin1/QuantScout-Skill/actions/runs/38021157135) 已通过 Python 3.10 / 3.12。当前核验以 [验证状态快照](VALIDATION_STATUS.md) 为准。
