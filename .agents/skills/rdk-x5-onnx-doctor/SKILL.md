---
name: rdk-x5-onnx-doctor
description: 通过终端检查 ONNX 的 RDK X5 BPU 静态约束、分析 Conv2D 及 Mul/Sigmoid/Add/Concat/Slice/Gemm、定位异常并追踪输出依赖。用户说“分析这个 ONNX 是否适合 RDK X5”“检查 Conv”“查看风险路径”“分析输出 Tensor 大小和大特征图”“找冗余节点、逆 Transpose、no-op Reshape”“量化前优化候选”时使用。
---

# RDK X5 ONNX Doctor

1. 在本仓库根目录执行。确认模型路径存在；未提供路径时请用户给出本机模型路径。默认只读模型，所有交互在终端完成。
2. 检查 `.venv/bin/python -m rdkx5_doctor --help`。没有项目环境时执行 `python3 -m venv .venv` 和 `.venv/bin/python -m pip install -e '.[dev]'`，不向系统 Python 安装依赖。
3. 规则集默认 `rulesets/x5-bayes-e`，也可使用用户指定目录。先执行 `.venv/bin/python -m rdkx5_doctor rules validate --ruleset <规则集>`。缺失、schema 失败或来源不确定时明示，不填未经核实的限制。
4. 选择新的、Git 默认忽略的 `reports/<模型名>-<日期时间>`，执行 `.venv/bin/python -m rdkx5_doctor analyze --model <路径> --ruleset <规则集> --out <目录>`。正确引用独立路径参数，不把文件名作为 shell 代码。
5. 确认退出码与 analysis.json/report.md；读取报告，结合 references/official_sources.md 和 conv_rule_notes.md 解读。JSON 是静态事实来源。检查规则包版本及七类算子覆盖矩阵；其余算子未检查。UNKNOWN 必须说明缺少什么事实。新增来源见 references/x5_multiop_sources.md 与 x5_multiop_rule_review.md。
6. 通过 `nodes --analysis <JSON> --status VIOLATION` 查看异常，`nodes --analysis <JSON> --search <关键词>` 搜索；用 `inspect --analysis <JSON> --node <内部 ID>` 查看真实参数、逐规则证据和建议；用 `trace --analysis <JSON> --node <内部 ID>` 查看可达输出和 Tensor 路径。三个查询都支持 --json。
7. 向用户报告原始名称/内部 ID、规则 ID、实际/允许值、官方来源和依赖路径。原始名称重复时用内部 ID。依赖路径不等于下游违规、CPU 回退或精度下降。
8. 用户询问输出/资源时，调用 `tensors --analysis <JSON> --kind output --json` 和 `tensors --analysis <JSON> --kind intermediate --sort bytes --limit 10 --json`；对关键 Tensor 用 `tensor --analysis <JSON> --name <名称> --json` 查看精确 B、未知原因与 fanout，按实际 producer/consumer ID 选择 trace。不要把逻辑载荷之和叫峰值内存；假设 INT8 场景不是量化结果。
9. 用户询问优化时，调用 `candidates --analysis <JSON> --json`，对值得审查的模式用 `candidate --analysis <JSON> --id <候选 ID> --json` 查看证据、接口、分支、重叠、按需输出路径及验证步骤，再用 nodes/inspect/trace 核对源节点。INSUFFICIENT_INFORMATION 不是已确认冗余；REVIEW_REQUIRED 不是可自动删除。不要宣称工具链已经或尚未融合。旧报告缺新字段时重新 analyze 到新目录，不能编造数据。
10. 根据覆盖矩阵中的确定违规、关键 UNKNOWN 和重要输出路径选择节点，再根据具体发现决定下一项确定性查询，用节点证据解释工程取舍；这一步是 Skill 对 Python 工具的编排作用，不要为所有算子粘贴同一段建议。
11. 建议绑定 node_id/rule_id/evidence，说明需要导出检查、重训或工具链验证，不承诺精度/FPS。无违规时只说“已检查规则未发现违规”。给出报告路径与终端查询命令。

禁止自动执行 Docker、hb_mapper、PTQ、板端推理、原始 ONNX 改写或删除/替换节点；不评分、不生成可视化或打开浏览器。始终声明未经过 OpenExplorer/hb_mapper 实测。规则已缓存，不需每次重读全部官方网页。

结构或导出方式变更只能回到原始训练/导出工程实施、重新导出并复检；是否重训取决于计算语义变化。用户未提供训练仓库时不得自行搜索其它工程。仅将精选摘要保存到 docs/validations/，完整 JSON 和日志留在 reports/。规则包版本、文档版本、ONNX opset 和未验证工具链必须分别说明。

本轮用户偏好优先：模型检测报告不上传；开发期间 Markdown 可供线上审查，开发结束后的模型验收 Markdown 留在 reports/，不进入 Git。
