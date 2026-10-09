# 中文回复契约

先说明格式检查和当前用户 profile MATCH/MISMATCH/UNKNOWN，若请求只涉及资源/节点可省略无关 profile。
异常优先：已证明静态 FAIL（node_id/原名、Tensor/Shape、rule_id、actual/expected、source URL/版本/列），然后条件未知和未覆盖。
把同原因的多个节点作为一组解释，但每个节点证据必须保留在 JSON/查询日志。

知识结论与机器结论分别表述：
- “官方 X5 ONNX BPU 栏列出”不等于“节点运行于 BPU”。
- “无专属约束”不把 NOT_COVERED 变为 PASS。
- “折叠/转成其他算子”不证明当前模型已折叠或融合。
- “CPU 文档条目”不证明实际 fallback。
- “未查/未找到/访问失败/缓存/版本冲突”分别标状态。
- 空查询记录不声称本次已检索；旧 JSON 没有字段不猜补，说明重跑需求。

资源原始 dtype 理论载荷与假设 INT8 分列；真实内存、任务精度、性能未测量。
最后给本地报告/检索路径和需要的下一步验证。所有结构变化回到独立原训练/导出工程；不提供具体源码定位或补丁。
禁止主观评分、编译通过或全图 BPU 保证。toolchain_version=unverified，compiler_checked=false。
