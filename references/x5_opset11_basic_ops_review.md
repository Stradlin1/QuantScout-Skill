# Opset 11 基础算子：X5 ONNX BPU 审核台账

审核日期 2026-10-09（Asia/Singapore）；实际公开网络读取并交叉核对手册 1.1.2 与英文 2.0.0。主 RDK 页面两次 open 失败（UnexpectedStatusCode），未把失败当“未找到”。运行明细与 schema 属性快照只留 reports/v1_4_baseline；本文件提供离线 CACHED 知识，不意味着每次调用都会实时查证。

来源：
- [X5 芯片用户手册 1.1.2](https://developer.d-robotics.cc/x5_sdk_doc/toolchain_development/intermediate/supported_op_list.html)：6.3.3.3 **X5支持的ONNX算子列表**，**X5 BPU支持约束**列；转换补充为6.3.3.1使用限制。
- [X5 Chip User Manual 2.0.0](https://developer.d-robotics.cc/x5_sdk_doc_v2.0.0/en/toolchain_development/intermediate/supported_op_list.html)：对应 ONNX / X5 BPU 列。本次四个新包数值条款无版本差异。
- [RDK 主表](https://developer.d-robotics.cc/rdk_x_doc/Advanced_development/toolchain_development/intermediate/supported_op_list)：本次不可访问，不作为新增机器约束的独立已查来源。
- [ONNX 官方 schema](https://onnx.ai/onnx/operators/) 与本地 onnx.defs.get_schema(op,11,'')：仅解释 ONNX 语义，不能提供 BPU 限制。

新包仅限 imported Opset 11、标准域，子版本0.1.0。现有49条和原子包不改。

| 算子 | Opset11实际 schema | X5 ONNX BPU 条款（两手册一致） | 本版处理 / 状态 |
|---|---:|---|---|
| Reshape | 5 | 输入输出 rank 1～10；转换/折叠需核实 | 2条 rank 自动 + 1条非阻塞转换 review；X5_BPU_DOCUMENTED_WITH_CONSTRAINTS，同时 FOLDED_OR_LOWERED_CONDITIONALLY |
| Split | 11 | 不沿N分割；输入长度为每块长度倍数；分块数应整除，非四维可支持 | 2条自动规则 + 1条非阻塞整除措辞审查，N需要Conv2D/显式布局证据；X5_BPU_DOCUMENTED_WITH_CONSTRAINTS |
| MaxPool | 11 | kernel/stride/padding 各≤256；不支持dilation | 2D rank4：4条自动，dilation默认/显式单位=无膨胀；非单位FAIL；X5_BPU_DOCUMENTED_WITH_CONSTRAINTS |
| AveragePool | 11 | H/W kernel 1～256，面积>1且≤8192；stride1～256；pad0～255 | 2D rank4：7条数值；X5_BPU_DOCUMENTED_WITH_CONSTRAINTS |
| Transpose | 1 | 任意输入维度；int16能力描述 | 仅知识，NOT_COVERED 保留；X5_BPU_DOCUMENTED_NO_EXTRA_CONSTRAINTS（未把能力宣称作为已执行条件） |
| Relu | 6 | 无专属约束 | 仅知识，NOT_COVERED 保留；X5_BPU_DOCUMENTED_NO_EXTRA_CONSTRAINTS |

## 解释和排除

- Split Opset11 split为属性，省略时按输出数等分；负轴按已知rank规范化。显式大小先检查输出数、正值与总和（ONNX语义），再核对各块整除。文档“split数应可以整除”未明确指输出数或每块长度。L % output_count 只记录观察值，COUNT-DIVISIBILITY 为非阻塞 review_only，不产生自动 FAIL；真实不等分块揭示这种歧义，须未来官方澄清。未知长度返回UNKNOWN，不物化特征数据。N不是任意布局axis0；未知布局保留UNKNOWN。
- Pool仅在已知 rank4 的标准ONNX N,C,H,W语义范围确定数值。未知rank保留UNKNOWN，其他rank NOT_COVERED；CPU表的4D/5D、auto_pad/storage_order描述不搬入BPU。auto_pad非NOTSET的有效pad本版UNKNOWN；kernel/stride仍可单独核验。MaxPool单位dilation代表没有膨胀，不因属性出现而FAIL。额外输出、ceil_mode、通用BPU限制与执行位置不由这些数值规则证明。
- AveragePool面积用整数H*W，不使用X3 Kernel[1,7]限制；64×128=8192是边界，65×127=8255越界。ONNX非法属性和BPU越界分层。
- Reshape目标tiny值复用已有有界缓存，动态目标未知不是语义冲突。rank匹配只证明专属rank条件，常量折叠/运行时转换仍非阻塞UNKNOWN。
- Transpose CPU列的两个布局perm不是BPU白名单；合法任意perm不虚构FAIL。Relu“无限制”不生成假通用数值限制，不登记可执行YAML。

## 未覆盖节点的离线知识（同一1.1.2/2.0.0 ONNX章节来源）

下面第二梯队仅缓存文档地位和方向，不是完整条款抄录；尤其 Gather/Div/ArgMax 的数值/条件需要按节点按需查原行，不能用此简表判自动 PASS。

| 算子 | knowledge_status | 查证解释 / 仍需验证 |
|---|---|---|
| Shape、Constant | FOLDED_OR_LOWERED_CONDITIONALLY | 表中说明折叠为数值存储；原图节点仍存在，元信息可求值不证明独立BPU执行 |
| Unsqueeze、Squeeze | FOLDED_OR_LOWERED_CONDITIONALLY | 转成Reshape，转换与约束仍需工具链验证 |
| Flatten | FOLDED_OR_LOWERED_CONDITIONALLY | CPU条目提到部分场景融合；不假定已融合 |
| Cast | X5_CPU_DOCUMENTED | CPU文档条目；不据原图断言真实fallback |
| Gather | X5_BPU_DOCUMENTED_WITH_CONSTRAINTS | 条件应按实际行进一步查证，本版不生成自动规则 |
| Div、Tanh | X5_BPU_DOCUMENTED_WITH_CONSTRAINTS | 有各自专属rank/广播等条款，不套用未经审核的其它规则 |
| ArgMax | X5_BPU_DOCUMENTED_WITH_CONSTRAINTS | 需结合实际axis/channel与来源判断，本版未实现硬件规则 |

资料层与机器层分开：知识不改diagnostics；没找到只有成功查阅指定版本才可记录NOT_FOUND，访问失败为SOURCE_UNAVAILABLE。没有hb_mapper/板端结果，compiler_checked=false、runtime_placement_verified=false。
