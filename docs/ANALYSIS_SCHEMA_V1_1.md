# analysis.json 1.1

V1.1 分析输出 schema_version="1.1"，包版本 0.2.0；Conv2D 规则包仍为 0.1.0。V1 的 model/ruleset/nodes/tensors/edges/diagnostics/traces/summary/unverified_assumptions/limitations 键保留，新增 resource_analysis 与 optimization_candidates。

旧查询 nodes/inspect/trace 接受 1.0/1.1。四个新查询读取对应新 section；旧报告缺 section 时返回 2，提示重跑 analyze。查询不重新加载 ONNX。独立运行的 analyze 仍只有 analysis.json/report.md 两个文件。

## resource_analysis（内部 schema 1.0）

| 字段 | 含义 |
|---|---|
| units | B、MiB divisor 1048576、MB divisor 1000000 |
| tensor_records | 每个 Tensor 名称一条记录，不重复统计 |
| summary | 每类 known_bytes_sum/unknown_tensor_count/completeness，前 10 大已知输出/中间项和 fanout 数量 |
| hypothetical_int8_label | 明确声明假设 INT8 原始载荷场景 |

记录保留 shape（整数/符号/null）、dtype、原始边界/initializer flags、producer_node_id/consumer_node_ids/consumer_count。resource_category 优先级：initializer > model_output > model_input > constant > intermediate_activation > other_unknown。导出 initializer 使用 initializer 主类并给出 classification_note，不再计入 output 主类。

element_count 仅对全部非负 Python int（排除 bool）计算；标量 [] 为 1，静态零维为 0，0 与未知维并存仍未知。raw_bytes 为元素数乘 dtype 明确字节数，整数为权威值，不提前舍入。mib/mb 仅显示辅助，超出浮点可表示范围时为 null，但 exact B 保留。

size_status 是 KNOWN、UNKNOWN_SHAPE 或 UNKNOWN_OR_UNSUPPORTED_DTYPE；shape_status 与 dtype_status 分别说明不确定性，unknown_reason 保留具体原因。支持 float16/bfloat16/float32/float64、signed/unsigned 8/16/32/64-bit integers、bool（假定逻辑 1 B）。string、complex、packed 2/4-bit、sparse、sequence/map/optional 未支持宽度计算。

hypothetical_int8_bytes 只对已知形状、受支持的稠密数值类型计算，bool 和 unsupported 类型不计算。这不是真实量化、BPU 分配或模型整体压缩预测。所有 known_bytes_sum 不是峰值内存；未知成员存在则 PARTIAL，否则 COMPLETE。

## optimization_candidates（内部 schema 1.0）

| 字段 | 含义 |
|---|---|
| candidates | 按原图顺序和模式确定排序，ID 为 OPT-0001 等 |
| summary | candidate_count、counts_by_pattern、counts_by_classification、confirmed_local_redundancy_count、insufficient_information_count |
| limitations | 结构语义分析、非硬件规则、未改写/未测量、版本覆盖限制 |

每条 candidate 保存 candidate_id/pattern/node_ids/node_names/tensor_names/classification/evidence/explanation/conditions/blockers/benefit_hint/verification_needed/overlap_with/source_urls/limitations。

分类 SEMANTICALLY_REDUNDANT 表示在当前已验证局部语义下无变化，仍不自动改写；REVIEW_REQUIRED 表示融合、公开输出或共享分支等需审查；INSUFFICIENT_INFORMATION 仅是未确认观察，不可报告为已证明冗余。

模式：IDENTITY、TRANSPOSE_INVERSE_PAIR、CAST_SAME_DTYPE、RESHAPE_NOOP、CONV_BN_FUSION_REVIEW。evidence 包含真实 Tensor 接口/消费节点，复合模式额外包含真实连边。重叠候选写入 overlap_with，不累加收益。

analyze 不为每条候选执行全图追踪：reachable_outputs=null，reachability_status=ON_DEMAND。`candidate` 查询仅追踪所选候选最后一个节点，在返回对象中添加 trace/reachable_outputs 和 COMPUTED_ON_DEMAND 状态；不修改保存的 JSON。

## 有界形状常量

GraphIR.small_constants 和涉及的 Tensor.small_constant 保留状态、来源、shape、dtype、values 或 unknown reason。仅解析 Reshape 请求的内联 signed INT32/INT64 initializer 或 Constant 值；检测 Reshape 仍要求一维 INT64 shape vector。

每常量最多 64 个元素、编码最多 4096 B。直接按小端整数解码，不调用 NumPy 加载权重。拒绝 external data、超限、错误编码、非整数；initializer 也是图输入时可被覆盖，值视为未知。不计算动态 Shape、算术折叠或通用常量传播。

## 语义覆盖限制

仅对标准 ONNX 域、导入 opset 1..23 中已审查的 operator schema 使用明确语义；Cast 从 schema 6、Reshape 从 schema 5 起。未审查版本返回信息不足观察，custom-domain 同名算子不套用规则。实际版本通过本机 ONNX schema 查找，与源码 verified schema 集合比对。

Reshape 支持默认零拷贝、单个 -1、schema>=14 的 allowzero；元素数量相等并不代表 no-op，必须 resolved target 与输入逐维完全一致。BN schema<=6 按 is_test、7/9 按实际输出数、14/15 按 training_mode 判断推理；训练模式不作为普通推理融合候选。参数仅核对元信息，不读取或折叠数值。
