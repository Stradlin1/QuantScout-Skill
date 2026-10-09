# Analysis schema 1.2

包 0.3.0 / Registry 0.2.0 / Conv 子规则包 0.1.0。保留 1.1 全部顶层键及资源、候选结构。新报告每个节点恰有一条 diagnostics，node_id 保留稳定图路径；不依靠显示顺序或原名唯一性。

ruleset 新增 operator_rule_versions、operator_opsets、toolchain_verified=false，完整保留 source_version 和 toolchain_version=unverified。规则结果保留旧 rule_id/field/actual/expected/status/reason/source_url/source_section/checkability，增量提供 node_id/operator/observed/reason_code/blocking/source/evidence/unverified。source 包含文档、版本、章节、URL、X5 BPU 栏和取证时间；evidence 绑定属性或输入输出 Tensor。operator_facts 记录派生字段、字段依据、独立 ONNX 语义问题与未验证条件。

节点 status：确定 FAIL 优先 VIOLATION；缺少阻塞事实或发现 ONNX 语义问题为 NEEDS_VERIFICATION；具体自动检查已通过且无阻塞未知为 NO_VIOLATION_FOUND；无适用规则为 NOT_COVERED。仅 UNKNOWN、仅 NOT_APPLICABLE 或没有规则不能算通过。非阻塞 review 提醒即使与通过共存，也仍保留逐条 UNKNOWN。

coverage_type 为 AUTO_CHECKED / PARTIAL_OR_CONDITIONAL / NOT_COVERED，表示检查覆盖，不表示 BPU 实测。summary.operator_coverage 按实际算子包含节点数、四状态计数及来源版本；所有行节点数及 all_status_counts 之和等于 nodes 数，covered_node_count + uncovered_node_count 同样守恒。旧 conv_status_counts 保持仅 Conv。

nodes/inspect/trace 只读接受 1.0/1.1/1.2。旧报告缺失新来源字段时明确显示缺失，不猜补。tensors/tensor/candidates/candidate 接受 1.1/1.2；1.0 缺资源字段时清晰报错，须重新 analyze。Conv 原规则和边界不变；资源理论载荷及候选从未代表设备内存或自动模型改写。
