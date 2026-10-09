# V1 终端版开发与验收记录

日期：2026-10-09。仓库起始为空。遵循原附件的静态分析链路及用户最新的纯终端要求，最终范围见 TERMINAL_V1_SPEC.md。没有可用的真实 YOLO，使用有效微型 ONNX 完成验收；未运行 Docker、OpenExplorer、hb_mapper 或量化。

| 阶段 | 实现 | 实际验证 |
|---|---|---|
| P0 | src 包、pyproject、.venv、CLI、pytest、README、AGENTS | editable 安装和 CLI --help 成功 |
| P1 | onnx_reader / graph_ir：checker、形状推断、稳定 ID、生产消费关系、符号维度、external data 警告 | reader/extractor 子集 18 passed |
| P2/P3 | Conv 参数、官方来源、版本化 YAML、严格 schema、字段白名单与逐规则证据 | rules validate 成功，19 条；kernel/volume 边界、Group/Depthwise、UNKNOWN 条件均通过 |
| P4 | BFS 可达输出、每输出最短代表路径、Tensor 标签、两层邻居、死分支、省略标记 | 双输出与 751 节点菱形图通过；不枚举指数路径 |
| P5 | 纯终端 analyze/nodes/inspect/trace、JSON/Markdown、节点建议 | 摘要、搜索、过滤、参数/来源、任意节点追踪、重名报错与 --json 测试通过 |
| P6 | 仓库级 Codex Skill、小模型、示例两份报告、打包 | 完整回归和独立 wheel 安装验证 |

## 官方约束核对

实际阅读 RDK X5 ONNX Conv 行，交叉核对 X5 手册 1.1.2 / 英文 2.0.0，以及 ONNX Conv 规范。规则包 0.1.0 收录 15 条可执行静态检查和 4 条条件提示；工具链版本 unverified。来源、版本冲突和排除条款分别在 references/official_sources.md、conv_rule_notes.md、manifest.yaml 中。

CPU 列条款没有当作 BPU 限制。shortcut 措辞冲突不自动套用。量化 dtype、超常规通道/末端例外返回 UNKNOWN。字段缺失不伪造数值。规则 schema 拒绝未知字段、错误界限、重复 ID、非法条件类型及不安全来源。

## 完整实例

执行 `examples/generate_demo.py`、CLI analyze、nodes、inspect、trace，以及 `examples/regenerate_report.py`。demo.onnx 仅 933 字节，kernel_h=32 对 [1,31] 明确 FAIL；经过分支/汇合，到达 prediction 和 auxiliary。示例存于 examples/demo-report/analysis.json 与 report.md，无前端资源。原始 ONNX SHA256 前后相同。

终端查询支持内部 ID 和唯一原名，重复原名要求内部 ID；控制字符在模型名称显示时转义，防止 ANSI 注入。

## 验证结果

| 命令/检查 | 结果 |
|---|---|
| `.venv/bin/python -m rdkx5_doctor --help` | 成功，analyze/rules/nodes/inspect/trace |
| `.venv/bin/python -m rdkx5_doctor rules validate` | 成功，19 条 |
| `.venv/bin/pytest -q` | **47 passed** |
| `.venv/bin/python -m build` | wheel / sdist 成功 |
| 新 venv 安装 wheel，从 /tmp 运行 validate/analyze/inspect/trace | 成功，内置规则可脱离源码目录运行 |
| 非法模型、畸形规则、畸形报告、重名/缺失节点 | 明确错误、退出码 2 |
| 模型内容不变、输出软链保护、双输出路径 | 通过 |

测试使用本环境 Python 3.12；声明支持 Python ≥3.10，但未在每个 Python 小版本执行。实测依赖快照见 requirements-dev-tested.txt。

## 未验证/未覆盖

- 用户本机 Ubuntu VS Code Codex 的自然语言发现行为尚未验证；Skill 中的完整命令工作流已实际执行。
- 未获得真实 YOLO ONNX，未做真实模型压力测试；大型微型图测试已通过。
- 其他算子、嵌套子图、通用 BPU 大小限制、实际分配/量化精度/性能未覆盖。
- 通道例外、shortcut 版本差异和量化后输出精度仍需未来工具链核验。

# V1.1 增量开发记录

## P7 基线（2026-10-09）

分支 main，HEAD 38cb99b；工作树干净，无预先存在的用户修改。Python 3.12.14，依赖沿用 requirements-dev-tested.txt。实际执行 --help、rules validate、pytest -q：19 条 V1 规则校验成功，**47 passed in 0.80s**。这是重新执行得到的基线，非引用历史结果。

兼容契约：新分析输出 schema_version 1.1；所有 V1 顶层键保持。nodes/inspect/trace 接受 1.0 和 1.1。资源/候选查询遇到旧报告缺新增部分时返回 2，并提示重跑 analyze。现有 Conv2D 规则不修改。CLI 成功仍为 0，错误仍为 2。

## P8 Tensor 理论资源

新增 tensor_resource.py；只做元信息整数运算，没有数组分配。分类去重、输出/中间/initializer 分开汇总，标量、零维、符号/未知维、dtype 宽度、稀疏表示、fanout、Top 10、PARTIAL 均保留证据。假设 INT8 与实际量化明确分开。

首次执行 `pytest -q tests/test_tensor_resource.py`：35 passed；后续加巨大整数浮点显示溢出与有效 sparse initializer 的测试，共 37 项。FP32 [1,64,320,320] oracle 为 26,214,400 B=25 MiB，假设 INT8 为 6,553,600 B=6.25 MiB。byte 数不取 NumPy/序列化文件大小。

## P9 五种模式

新增 optimization_candidates.py、small_constants.py、dtype_utils.py。复用 GraphIR，缓存形状小常量；模型权重不二次物化。只读取 Reshape 所需的内联 signed INT32/64，最多 64 元素、4096 编码 B；拒绝 external、超限、错误编码和可覆盖 initializer。

已查 ONNX 官方 operator 页，并使用本机 `onnx.defs.get_schema`核对实际导入版本。Identity/Transpose/Cast/Reshape/推理 BN 的证据和 public output、fanout、未知条件、重叠候选明确记录。Conv-BN 永远为融合审查，BN 参数未读数值。未来版本>23 不默认为已理解语义；custom-domain 不套用。

首次候选测试发现一条测试 oracle 错误：零输入 [-1,0] 在 allowzero=0 时 0 复制非零第二维，-1 可唯一解为零；修正 oracle，并另测真正 0/0 不确定情形。候选初轮 36 passed，增加未知 opset、缺 metadata Identity、可覆盖常量、allowzero、BN dtype/channel 后为 41 项。

## P10 查询、报告与 Skill

新增 resource_queries.py；tensors/tensor/candidates/candidate 均支持 --json；读取保存报告，不加载模型。candidate 只追踪所选候选最后节点，无全候选最短路径开销。补充 dtype/稀疏/容器元信息；Constant 重复值属性只保留元信息，大常量列表不物化。text_utils.py 保护终端 C0/DEL/C1 控制字符及 JSON 的正确转义。

报告保留 V1 八个章节并增加资源与候选，默认只列资源前 10 和未知前 10；完整事实在 JSON。更新已有 Skill，按资源/候选结果选择进一步 inspect/trace。schema 文档说明 1.0/1.1 兼容。

CLI V1.1 子集 13 passed：新/旧协议、过滤/排序、未知数、四个 --json、重名/缺失错误、删除原模型后保存 JSON 查询、终端控制字符、安全只读与可复现事实。原有 V1 47 项未删减；全量执行 **138 passed**（新增 91 项）。

## P11 回归、包装与交付

实际执行 --help、rules validate、`.venv/bin/pytest -q`、`.venv/bin/python -m build`：九个终端子命令正常，19 条原规则通过，**138 passed in 1.11s**，wheel 和 sdist 均成功构建为版本 0.2.0。原有 47 项 + 资源 37 项 + 候选 41 项 + CLI 13 项 = 138；新增 91 项。原 Conv2D YAML、V1 测试和历史 schema 1.0 示例报告保持不变。

在全新临时 venv 安装构建的 wheel，工作目录切到 /tmp，逐项执行 --help、rules validate、analyze、tensors、tensor --json、candidates、candidate --json，全部退出 0。独立安装实际解析 ONNX 1.23.2 / NetworkX 3.7 / Pydantic 2.14.0 / PyYAML 6.0.3，也验证了内置规则随 wheel 安装，无 editable/source 目录依赖；全量单元测试环境仍为 requirements-dev-tested.txt 中的版本。

生成七节点微型模型 examples/v1_1_demo.onnx，并提供 examples/v1_1_demo-report/analysis.json 与 report.md。实际分析：
- prediction：float32 [1,2,4,4]，32 元素，**128 B**（0.0001220703125 MiB）；假设 INT8 为 32 B，不是实际量化。
- 中间激活已知理论载荷之和 768 B，完整度 COMPLETE；不是峰值内存。
- OPT-0002：main/node_000001 to_nhwc → main/node_000002 back_nchw，perm [0,2,3,1] / [0,3,1,2]，复合 [0,1,2,3]，SEMANTICALLY_REDUNDANT。
- 总计 4 SEMANTICALLY_REDUNDANT、1 REVIEW_REQUIRED；后者为 Conv→推理 BN，参数未读数值，也未执行融合。

对原 examples/demo.onnx 重新 analyze 到 reports/v1_1-original，执行 tensors/candidates/nodes/inspect/trace（oversized_kernel），全部退出 0。两模型分析前后 SHA256 相同：
- demo.onnx：9f1da4be1ddb5e3b3873daffc4d5734014bf593120e7812d7a4736b316b2b116
- v1_1_demo.onnx：fe9d960d5264b8167a8e8b20cb309dfd160eb5bf302ed975de024fbda5215cba

未做真实 YOLO/大型权重压力测试、用户本机 Codex Skill 自动发现、ONNX Runtime 数值等价、Docker/hb_mapper/量化/板端验证；未声称性能、实际 BPU/DDR 分配或部署结论。嵌套子图、复杂/压缩 dtype 的载荷、未知形状仍按限制说明处理。候选识别保守限制到已核对的 imported opset ≤23，未来/自定义版本返回信息不足或不套用。按需追踪为规范允许的选项；规格无功能范围偏离。仓库内规格副本仅将 Markdown 行尾硬换行转换为 <br>，便于 git whitespace 校验。

## 真实 YOLO26 ONNX 验收（2026-10-09）

在 main / c10ac2ac7d672716e95c4b8b6e264048b5d6b4ec 的干净工作区实际执行终端验收。模型为用户指定的 yolo26_lane_robot.onnx（104,290,060 B），没有复制权重进仓库或修改源模型。完整本地结果位于 `reports/yolo26_lane_robot-20261009-181829/REAL_ONNX_VALIDATION.md`；reports 按已有策略忽略，含最终 analysis.json/report.md、before_fix 快照、命令日志及独立审计。

环境为 Ubuntu 22.04.5 / WSL2、Python 3.10.12、ONNX 1.23.2、NumPy 2.2.6、NetworkX 3.4.2、Pydantic 2.14.0、PyYAML 6.0.3、pytest 8.4.2。初始没有 .venv，系统 ensurepip 缺失，sudo 安装需要密码；通过官方 get-pip 引导仅安装项目环境，再完成 editable dev 安装。首次 pytest 因 ROS PYTHONPATH 引入 launch_testing、缺 lark 在收集前失败；清除外部 PYTHONPATH 后完整运行，未跳过项目测试。

- CLI --help 和原 19 条规则校验成功；隔离环境基线 **138 passed in 0.96s**。
- 真实模型 opset 11，281 节点、23 种算子、1 输入和 2 输出；原始节点/名称/接口/连边逐项对照一致，checker 成功，无 shape inference 异常、自定义节点或外部权重。
- 47 个 Conv2D 全为 NO_VIOLATION_FOUND；517 PASS、376 NOT_APPLICABLE、0 FAIL/UNKNOWN。234 个非 Conv 节点 / 22 种算子 NOT_COVERED，没有完整 BPU 兼容结论。
- images 为 float32 [1,3,640,640]；cls_logits 为 [1,161,56,4] / 144,256 B，offset 为 [1,1,56,4] / 896 B。
- 共 403 条 Tensor，231 个大小已知、172 个符号 shape 未知。首层最大已知中间项 [1,32,320,320] 为 13,107,200 B = 12.5 MiB。已知中间载荷之和 93,475,488 B（PARTIAL），不是峰值或部署内存。71 个多消费者中间项。
- 首个未知 shape 起于 Shape/Gather/Add/Div/Mul 计算 Slice 边界的链。原图无中间 value_info；默认推断及只读 data_prop=True 对照均留 172 个符号 shape。记录能力限制，不猜测后续资源。

发现一个候选筛选缺陷：Reshape 输入存在符号轴时，检测器直接返回信息不足，忽略已知 rank/轴已能否定 no-op 的证据。真实模型初始 9 个 INSUFFICIENT_INFORMATION 观察中，8 个 rank2→rank3，另一个已知轴512→4。增加已审查语义、一维INT64正目标下的否定判据；未改 BPU YAML、未猜符号维度、未重写模型。新增3个 tiny fixture 回归：修复前 **2 failed / 1 passed**；修复后 **3 passed**；完整回归 **141 passed in 0.93s**，保留原138项。

重新真实 analyze 成功（0.922秒，耗时仅本机观测），五种候选模式最终均为0。除 optimization_candidates 外全部 JSON 顶层事实与修复前相同。所有重要节点实际执行 nodes/inspect/trace/tensor；初始9个观察全部执行 candidate 查询。注意力两个 Transpose 间隔 MatMul/Mul/Softmax，不能抵消；11个 Reshape 有实际shape变化证据。候选查询历史不表示最终仍有候选。

模型前后 SHA256 相同：`a8ea6ccf474c77614f33512c4a118390f3b14c28e4871615670888a6ca292de4`。未执行 Docker/hb_mapper/量化、图优化、评分、GUI、数值推理或 git commit/push。工具链分配、真实 BPU/DDR/SRAM、峰值、性能和任务精度仍未验证。V1.2 建议优先有界 shape 算术传播、未知来源诊断，再按官方版本证据扩展 elementwise/Slice/Concat/Resize/MatMul/Softmax/Gemm 等真实算子规则；本次没有编造或新增硬件限制。

## V1.2 — P12–P18（2026-10-09）

本轮按 `RDK_X5_ONNX_Doctor_V1_2_Multi_Op_BPU_Development_Spec.md` 渐进实现。使用项目 .venv、隔离 ROS 的 PYTHONPATH；全程只读模型，不执行 Docker、hb_mapper 或量化，不 commit/push。用户原有规范及 Zone.Identifier 保留。

- P12：读取规范、已有源码/协议/Skill/真实验收基线，执行 `--help`、`rules validate`、`env -u PYTHONPATH .venv/bin/python -m pytest -q` 和旧版真实分析。基线 141 passed；Conv 19 条。在线核对主站 X5 ONNX BPU 栏、手册 1.1.2 与英文 2.0.0，六类条款一致。记录网页和实际工具链版本的区别。
- P13：引入安全 RuleRegistry、算子字段与谓词白名单、注册顺序及版本/domain/opset 校验；保持原 Conv YAML 字节不变、直接接口兼容。拒绝路径逃逸、软链逃逸、重复 ID/文件/YAML key、未知字段/谓词/额外配置及危险 YAML。阶段 159 passed。
- P14：依次实现 Sigmoid、Concat、Slice、Add、Mul、Gemm extractor 和 YAML；每类运行定向测试及真实子集检查。定向结果分别 6/7/5/15/38/9 项通过；全量阶段最终 223 passed。Add/Mul 使用独立 NumPy 广播 oracle；固定常量区分 initializer 可覆盖性；Slice 仅有界读取整数参数；Gemm 保留逻辑矩阵参数，不冒充 Conv 布局。早期测试选择器有一次未选中和一次误选尚未接入 Gemm 测试，改用明确 node selectors 后通过；未 skip 测试。旧测试的总规则版本断言迁移到 Registry 0.2.0，保留 Conv 0.1.0。
- P15：生成 schema 1.2，保留全部旧顶层事实，添加逐规则节点、Tensor、实际值/允许值、文档版本/X5栏证据；每节点恰一诊断，四状态与覆盖数守恒。CLI 保持旧节点/路径/资源/候选查询；历史 1.0/1.1 只读兼容。Concat 增加布局证明：仅 rank4 不足以认定 N 轴，加入回归。阶段 228 passed。
- P16：报告分为版本/摘要/覆盖/违规/未知/资源/候选/建议/来源，去除正常节点批量建议；违规前50、未知每组10、候选10，提示省略且 JSON 全量保留。增加转义、防注入和截断/守恒测试，新增协议与报告策略文档。阶段 232 passed。
- P17：增加 reports/、缓存、环境文件及外部模型数据忽略，保留 examples/*.onnx 例外；对 818 个已跟踪历史产物保存 SHA256 清单，`git rm -r --cached --quiet -- reports` 后逐个检查本地文件及 hash 不变。更新 README/AGENTS/Skill 的多算子选择、证据溯源与训练/导出回源原则；增加 Git hygiene 测试。阶段 233 passed。用户随后明确：模型检测报告不上传，开发 Markdown 可在周期内用于线上审查，但开发结束的模型验收 Markdown 不同步；因此覆盖规范中的 tracked docs/validations 模型摘要建议，最终仅保留于忽略的 reports/。
- P18：独立 ONNX API/checker 验证真实图、原 Conv 逐字段规则结果、Tensor 载荷算术及资源/候选不回归；逐异常及各重要 UNKNOWN 执行 inspect/trace，输出和 Top10 使用 tensors/tensor 查询。完整证据和模型 SHA256 前后记录位于本地独立 reports/ 目录，不进入 Git。临时验收脚本先误用 rule_results/bytes_per_element，改为实际 results/element_size_bytes，重新独立核对成功；未改变诊断条件。

最终审查补充“已知谓词但算子/字段错配”的最小反例及加载阶段拒绝，避免运行时访问不适用字段。清理 MANIFEST.in 对不存在网页/脚本文件的声明；不增加可视化。总规则 35：Conv19、Sigmoid2、Concat1、Slice1、Add6、Mul5、Gemm1。Slice/Gemm 仅 review_only，不伪造硬件 PASS/FAIL；Add shortcut 是非阻塞 review。FP32 与 int16 能力、ONNX 自身合法性与 BPU 限制分别呈现。

打包命令 `.venv/bin/python -m build --no-isolation`（利用已安装构建依赖）成功构建 wheel/sdist。创建 `/tmp/rdkx5-v12-wheel-validation` 新环境，下载依赖后仅从本地 wheelhouse 安装，在 `/tmp` 工作目录执行 --help/rules validate/rules list/真实 analyze；包来源明确为独立环境 site-packages。下载初次被沙箱禁止本地代理连接，按权限流程获批后重试成功；没有绕过限制。完整命令、退出码、stdout/stderr 保留在忽略的 reports/。

P18 最终全量测试：234 passed in 6.32s。最终 wheel/sdist 再构建成功，独立环境重装最终 wheel 后 rules validate 和真实 analyze 均成功，分析 JSON 与源码环境完全一致。git diff --check 通过。模型详细结果按用户要求仅保留本地 reports/REAL_ONNX_V1_2_VALIDATION.md（实际目录为独立带时间戳子目录）；未建立 tracked 模型验收摘要。

## V1.3

P19–P27 bounded Shape provenance and MatMul/Softmax/Resize development is recorded in `RDK_X5_ONNX_Doctor_V1_3_Development_Log.md`. The source model and model-specific reports remain local; this log contains implementation/testing information only. Prior V1/V1.1/V1.2 history is preserved.
