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
