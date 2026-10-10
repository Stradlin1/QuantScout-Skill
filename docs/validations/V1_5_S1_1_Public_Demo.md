# V1.5-S1.1 公开 demo 核对

公开输入为仓库原有 `examples/demo.onnx`（微型 Conv/分支演示，标准域 OpSet 11）。没有重写模型。
默认 `examples/demo-report/analysis.json` 已升级为 Schema 1.3：5 个已解析主图节点，四状态依次为 2/1/0/2，唯一违规节点 1 个、FAIL 记录 1 条。
真实证据：`main/node_000000`、`oversized_kernel`、`X5-CONV2D-KERNEL-H`，实际 kernel_h=32，允许 1～31。

## 实际 Agent 请求与产物

独立、全新、ephemeral Codex CLI 会话 E01 收到的真实请求：

> 总结一下这个 ONNX 的检查结果，只需要事实，不要分析和建议。已有检查数据在 reports/agent_e2e/E01/analysis.json，结果写入同目录的 summary.md。当前只处理本次总结请求，不开发代码、不改模型、不联网、不 commit 或 push。

该会话自行读取 Skill 和摘要契约，选择 overview，然后实际调用：

```bash
.venv/bin/python -m rdkx5_doctor summary-facts \
  --analysis reports/agent_e2e/E01/analysis.json --mode overview --limit 2 --json \
  --out reports/agent_e2e/E01/summary_facts.json
# Agent 读取 allowed_claims，组织四节中文草稿；未使用旧 summary 模板
.venv/bin/python -m rdkx5_doctor summary-publish \
  --analysis reports/agent_e2e/E01/analysis.json \
  --facts reports/agent_e2e/E01/summary_facts.json \
  --draft reports/agent_e2e/E01/summary_draft.json \
  --out reports/agent_e2e/E01/summary.md
```

上述两个命令实际退出码均为 0。检查事件确认未重新 analyze，未检索官方资料；最终回复只提供文件与事实范围。
经实施 Agent 审读，计数、FAIL 证据、Profile 未提供、主图边界与未执行工具链声明一致，无新增分析和建议。
公开 [summary.md](../../examples/demo-report/summary.md) 采用这个独立 E01 会话的真实发布产物；公开 [summary_facts.json](../../examples/demo-report/summary_facts.json) 与同次事实包一致。
机器事实校验不能证明任意中文全语义；本例的受限句式和语义审读分别有记录。完整原始会话、草稿和日志只保留在忽略的 `reports/agent_e2e/E01/`。

## 可复现与历史兼容

```bash
.venv/bin/python examples/regenerate_report.py
```

实际连续执行两次，退出码均为 0，analysis.json/report.md 逐字节相同；既有模型及 Agent 摘要不变。
示例脚本仅在模型不存在时创建微型示例，已有 ONNX 从不重写。公开 JSON 固定键排序，避免现有 extractor 集合遍历使字段顺序跨进程改变；仅公开序列化归一化，不改变分析算法或 Schema。
历史 Schema 1.0 快照保留在 `examples/legacy-demo-report/`；历史测试仅改用该快照，断言保留。

没有执行 Docker、hb_mapper、量化、编译、板端测试或 ONNX 优化。未新增规则、S2/S3、外部 LLM SDK/API。
范围仅为主图：嵌套子图未展开，不计入节点诊断统计。Profile 未提供，不能推断 MATCH；未覆盖节点不当作 PASS 或不支持。
独立自然语言其他案例见 [Agent E2E 状态](V1_5_S1_1_Agent_E2E.md)。
