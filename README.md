# RDK X5 ONNX Doctor

**纯终端、只读**的 ONNX 静态诊断工具（V1.2）。根据版本化官方规则定位异常节点，沿 Tensor 依赖追踪到模型输出，提供终端搜索、过滤、节点详情和路径查询；保存 `analysis.json` 与 `report.md`。V1.1 新增输出/中间 Tensor 理论原始载荷统计与五种结构优化候选。

V1 未经过 OpenExplorer/hb_mapper 实测，不能保证 BPU 执行、量化精度或性能。不评分、不改写模型、不执行 Docker/量化。V1.2 检查 Conv、Mul、Sigmoid、Add、Concat、Slice、Gemm；其余算子解析展示但不检查。根据用户最新要求，**不生成 graph.html，不包含前端、浏览器交互或 Netron 依赖**。

## 安装（Ubuntu / Python ≥3.10）

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m rdkx5_doctor --help
python -m rdkx5_doctor rules validate --ruleset ./rulesets/x5-bayes-e
```

也可执行 `rdkx5-doctor`。核心依赖有明确版本范围；本次实测版本在 requirements-dev-tested.txt 中。无需 GPU、设备、Docker 或模型 API 密钥。

## 全流程终端使用

```bash
# 分析模型：直接显示异常摘要、规则证据、可达输出及代表路径
python -m rdkx5_doctor analyze --model /path/to/model.onnx --out reports/model

# 列出所有节点，或按状态过滤
python -m rdkx5_doctor nodes --analysis reports/model/analysis.json
python -m rdkx5_doctor nodes --analysis reports/model/analysis.json --status VIOLATION

# 按节点 ID、原始名称、算子、Tensor 搜索（忽略大小写）
python -m rdkx5_doctor nodes --analysis reports/model/analysis.json --search Conv

# 节点详情：Tensor/Shape、Conv 参数、逐规则实际值/允许值/官方来源和建议
python -m rdkx5_doctor inspect --analysis reports/model/analysis.json --node main/node_000000

# 追踪到输出：直接前后继、路径中的 Tensor、可达输出及其他路径是否省略
python -m rdkx5_doctor trace --analysis reports/model/analysis.json --node main/node_000000
```

`--node` 支持内部 ID 或唯一原始名称；原名重复时必须使用内部 ID。`nodes`、`inspect`、`trace` 均支持 `--json`，可接管道。查询使用已保存报告，不重新加载权重。可以追踪任意已解析节点，未覆盖算子的依赖也可查询。

不传 `--ruleset` 时使用安装包内置规则，可脱离仓库目录运行。输出只有 JSON 与 Markdown，无浏览器和静态资源。

## 先试小型示例

```bash
python examples/generate_demo.py
python -m rdkx5_doctor analyze --model examples/demo.onnx --out reports/demo
python -m rdkx5_doctor inspect --analysis reports/demo/analysis.json --node oversized_kernel
python -m rdkx5_doctor trace --analysis reports/demo/analysis.json --node main/node_000000
pytest -q
```

示例 Conv kernel_h=32，违反已收录规则 [1,31]；分叉后汇合，能到达 prediction 与 auxiliary 两个输出。已生成的小型模型和两份报告在 [examples](examples)。`python examples/regenerate_report.py` 可重建仓库内示例报告。

## V1.1：资源与优化候选（纯终端）

```bash
python examples/generate_v1_1_demo.py
python -m rdkx5_doctor analyze --model examples/v1_1_demo.onnx --out reports/demo-v1_1
python -m rdkx5_doctor tensors --analysis reports/demo-v1_1/analysis.json --kind intermediate --sort bytes --limit 10
python -m rdkx5_doctor tensors --analysis reports/demo-v1_1/analysis.json --kind output --json
python -m rdkx5_doctor tensor --analysis reports/demo-v1_1/analysis.json --name prediction --json
python -m rdkx5_doctor tensors --analysis reports/demo-v1_1/analysis.json --unknown --limit 0
python -m rdkx5_doctor candidates --analysis reports/demo-v1_1/analysis.json
python -m rdkx5_doctor candidates --analysis reports/demo-v1_1/analysis.json --pattern TRANSPOSE_INVERSE_PAIR --json
python -m rdkx5_doctor candidate --analysis reports/demo-v1_1/analysis.json --id OPT-0002
```

四个新查询均读取保存的 JSON，不再加载模型，均支持 --json。Tensor 列表按 raw B 排序，未知最后；--sort name 按名称排序；--limit 0 表示全部。--kind 可选 all/output/intermediate/input/initializer/constant/unknown，unknown 指未分类类别；`--unknown` 独立过滤任何类别的未知尺寸。

资源只使用 shape/dtype 的逻辑载荷。B 是精确整数；1 MiB=1,048,576 B，1 MB=1,000,000 B。静态 scalar/empty shape 支持；符号维度、未知维度、string/packed/sparse/container 类型保留未知及原因。initializer、输出、中间激活分别去重求已知字节和，存在未知成员则 PARTIAL。fanout 只表示消费者数量，不代表额外分配。

**Hypothetical INT8 raw-payload scenario** 是假设元素数乘 1 B，不是实际量化结果。资源统计不代表 BPU/DDR/SRAM 分配或峰值内存，不预测部署可行性、延迟、FPS 或精度。

| 模式 | 认定与限制 |
|---|---|
| IDENTITY | 标准域原值转发；公开输出或 fanout 需接口审查 |
| TRANSPOSE_INVERSE_PAIR | 真实 Tensor 连接，两个 perm 组合为 identity；共享中间分支不能全局删除第一节点 |
| CAST_SAME_DTYPE | 输入 dtype 与目标 `to` 明确相同且支持 |
| RESHAPE_NOOP | 有界常量目标，按版本正确解析 0/-1/allowzero，目标逐维等于静态输入；相同元素数不足以证明 |
| CONV_BN_FUSION_REVIEW | 直接相连且 BN 处于推理模式；参数/通道/精度/分支/接口需审查，不宣称编译器已经或尚未融合 |

局部无变化分类为 SEMANTICALLY_REDUNDANT；融合/接口/共享分支为 REVIEW_REQUIRED；无法证明的观察为 INSUFFICIENT_INFORMATION。候选含实际内部 ID/Tensor、证据、条件、阻碍、重叠关系与未来验证步骤；没有删除或重写节点。需未来 ONNX checker、输出接口/shape 和 ONNX Runtime 数值对比。候选最后节点的下游输出仅在 candidate 查询时计算。

新报告 schema=1.2，nodes/inspect/trace 仍接受 1.0/1.1；资源和候选查询接受 1.1/1.2。旧报告无法查询新资源/候选，需重跑 analyze。保持原有 Conv2D 规则不变。版本与字段见 [schema 文档](docs/ANALYSIS_SCHEMA_V1_1.md)，语义与支持范围见 [ONNX 来源](references/optimization_semantics.md)。

自带第二示例识别五种模式，prediction 原始载荷为 128 B；完整示例在 [examples/v1_1_demo-report](examples/v1_1_demo-report)。历史 V1 demo-report 保留 1.0 格式用于兼容性验证。

## Codex Skill

用 Ubuntu VS Code + Codex 打开该仓库，仓库级入口为 `.agents/skills/rdk-x5-onnx-doctor/SKILL.md`。示例请求：

> 使用 rdk-x5-onnx-doctor，分析 examples/demo.onnx 是否适合 RDK X5，通过终端查看异常节点参数和到两个输出的路径。

Skill 调用确定性 Python 工具、读取机器事实，再生成节点绑定的解释与建议。可以继续要求“分析输出 Tensor 大小和大特征图”或“找冗余节点、逆 Transpose、no-op Reshape、量化前优化候选”。Codex 应根据资源/候选证据选择进一步 inspect/trace，而不是粘贴泛化建议。CLI 可独立离线运行。这里是仓库级 Codex Skill，不是 ChatGPT Work 的插件安装包。用户本机 Codex 的自然语言发现流程仍需实际试用。

## 覆盖与状态

| 状态 | 含义 |
|---|---|
| VIOLATION | 至少一条已收录官方约束有确定 FAIL |
| NO_VIOLATION_FOUND | 已执行规则未发现违规，不是 BPU 兼容保证 |
| NEEDS_VERIFICATION | 关键元信息未知、不一致或量化条件无法确认 |
| NOT_COVERED | 非 Conv 或非 Conv2D，V1 未提供检查 |

逐规则返回 PASS / FAIL / UNKNOWN / NOT_APPLICABLE。自动检查 kernel H/W、每组体积、stride、dilation、有效 padding、膨胀 stride/整除条件。实际量化输出 int8、超常规通道条件保留 UNKNOWN。Conv→Add shortcut 措辞有版本差异，不自动套用。CPU 支持列条款不作为 BPU 条款。

来源与冲突见 [references/official_sources.md](references/official_sources.md)、[references/conv_rule_notes.md](references/conv_rule_notes.md)。规则缓存在 `rulesets/x5-bayes-e`，使用 schema 和白名单解释 YAML，无 eval。更新约束时升级版本并验证边界测试。

## 限制、错误与开发

- 0：分析或查询成功，即使存在违规或搜索无匹配；2：模型/规则/文件/查询失败，错误写到 stderr。
- 外部权重缺失或路径不安全：仅元信息结构校验并显式警告，不读取模型目录以外的权重。
- 符号维度不猜测；形状推断失败保留原始图；子图未覆盖时明确说明。
- SHA256 只覆盖 ONNX protobuf，不覆盖 external data。输出不得覆盖源模型，包括通过软链覆盖。
- 每个可达输出展示一条 BFS 最短代表路径，有其他路径时标记省略，避免指数枚举。
- 不覆盖通用 BPU 大小限制、其他算子支持情况、真实 CPU/BPU 分配、DDR、延迟或量化误差。

源码在 `src/rdkx5_doctor`，规则随 wheel 打包。根 rulesets 是指向包内资源的符号链接，避免双份维护。原附件与最新终端要求的优先关系见 [docs/TERMINAL_V1_SPEC.md](docs/TERMINAL_V1_SPEC.md)，执行结果见 [docs/DEVELOPMENT_LOG.md](docs/DEVELOPMENT_LOG.md)。

```bash
python -m build
pytest -q
```

## V1.2 多算子规则与报告

包版本 0.3.0，总规则包 0.2.0：Conv 保留原 0.1.0 的 19 条规则；新增六类共 16 条，共 35 条。规则只执行安全白名单比较或已审查谓词，按标准域、opset、文档版本匹配；“支持 int16”不会排除原始 FP32。Slice/Gemm 的工具链相关条件保守保留 UNKNOWN。

```bash
python -m rdkx5_doctor rules list --operator Mul --json
```

JSON 记录完整节点/Tensor/逐规则证据，Markdown 优先展示覆盖矩阵、异常和按原因合并的待验证事项；正常节点不重复生成建议。详见 [协议](docs/ANALYSIS_SCHEMA_V1_2.md)、[报告策略](docs/REPORT_POLICY_V1_2.md)、[官方来源](references/x5_multiop_sources.md)。

普通报告和日志保存在忽略的 `reports/`；历史生成报告仅解除 Git 跟踪，本地保留。结构、算子或导出方式建议必须回到训练/导出工程修改，再导出复检；本项目不修改 ONNX。静态规则通过不代表编译器/BPU 已验证，未知不能视为通过。

按本次用户要求，开发期间可同步 Markdown 供线上审查；开发结束后的模型检测报告和真实验收摘要仅保存在忽略的 reports/，不纳入 Git。
