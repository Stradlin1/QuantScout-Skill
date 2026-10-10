# QuantScout V1.5-S1.1 实施记录

日期：2026-10-10。正式规范：`QuantScout_V1_5_S1_1_Codex_Development_Spec.md`，基线 HEAD `e5b64d5`。
用户新增的规范文件完整保留，未改写。包版本保留 0.5.0，展示版本为 V1.5-S1.1；analysis Schema 1.3、规则包 0.4.0 与所有 66 条规则保持不变。

## P0～P5 实际结果

| 阶段 | 完成工作 | 实际验证 |
| --- | --- | --- |
| P0 | 完整读取规范、保护用户修改、保存基线 | help/rules 退出 0；pytest 462 passed |
| P1 | 三模式有界事实包，稳定 ID/来源路径、原始 JSON SHA、范围与截断 | 26 passed |
| P2 | Agent 受限事实组织、重算事实发布、默认精简 fallback/显式详细 | 103 passed，旧摘要测试保留 |
| P3 | 公开 Schema 1.3、旧快照保留、主图范围、真实 Agent 摘要 | 首次 109 passed/1 failed（字段键序）；公开序列化归一化后 110 passed |
| P4 | CI、构建/独立 wheel、真实新会话 E01～E10 | 首次全量 520 passed/5 failed（历史 fixture 路径）；迁移到 legacy 后 526 passed。E10 首次真实验收失败，契约/早停修正后 47 项相关测试通过、新会话复验通过 |
| P5 | 全量、最终 wheel/sdist、源模型/规则/规范哈希与 Git 审查 | **528 passed、0 failed**；build 退出 0；diff --check 退出 0；18 个保护文件哈希一致，HEAD 未变化 |

pytest 使用 `env -u PYTHONPATH .venv/bin/python -m pytest -q`，只对当前进程消除 ROS 插件干扰。
构建最初因沙箱网络下载失败；获得权限后隔离构建成功。最终以已安装构建依赖执行 `python -m build --no-isolation`，重新得到 wheel 与 sdist。最终 wheel 在源码目录外以独立安装路径执行 help、rules validate、io 事实导出，均退出 0；包含所有 15 个规则 YAML。
完整阶段命令/日志/退出码及保护哈希在忽略的 `reports/v1_5_s1_1/`，最终摘要为 `final_acceptance.json`。

## 文件清单与目的

| 修改文件 | 目的 |
| --- | --- |
| `.agents/skills/rdk-x5-onnx-doctor/SKILL.md` | 三模式自然语言路由，Agent facts→draft→publish，已有目标先停 |
| `.agents/skills/rdk-x5-onnx-doctor/references/summary_only_contract.md` | 草稿 Schema、事实占位符、受限连接语、必须事实、禁止扩写/覆盖 |
| `AGENTS.md` | 正式规范优先级、主图边界、真实验收及覆盖保护 |
| `README.md` | 当前展示版本、CLI/Agent 区别、三模式、边界及真实失败记录链接 |
| `docs/ANALYSIS_SCHEMA_V1_3.md` | 只补主图统计范围说明，不变更 Schema |
| `src/rdkx5_doctor/summary_facts.py` | 复用旧计数验证，新增有界三模式导出、SHA/来源与安全新建 |
| `src/rdkx5_doctor/summary_report.py` | 保留 markdown() 七段模板，默认三节确定性 fallback 与字符串上限 |
| `src/rdkx5_doctor/cli.py` | 保留历史入口，新增 summary-facts/summary-publish 与 --detailed |
| `src/rdkx5_doctor/onnx_reader.py` | 仅修正文档字符串为主图读取，未改解析行为 |
| `src/rdkx5_doctor/reporting_sections.py` | 原完整报告保留章节，仅补主图范围文字 |
| `examples/regenerate_report.py` | 不重写已有模型；公开快照字典键排序，跨进程报表可复现 |
| `examples/demo-report/analysis.json`、`report.md` | 公开默认演示升级 Schema 1.3/current rules |
| `tests/test_cli_v1_1.py`、`test_cli_v1_2.py`、`test_v1_3_cli_and_report.py` | 历史兼容断言保留，只改为读取 legacy 快照 |

| 新增文件 | 目的 |
| --- | --- |
| `src/rdkx5_doctor/summary_publish.py` | 当前事实包逐项重算，受限草稿引用核对及独占发布 |
| `tests/test_summary_facts_modes.py` | 模式、类型/守恒、SHA、上限、I/O、If、Profile 与只读测试 |
| `tests/test_summary_publish.py` | 篡改、自由扩写、绑定、路径、覆盖/符号链接、早停与模板兼容 |
| `tests/test_public_demo.py` | 公开事实/报表/摘要一致、再生成与 CI 结构验证 |
| `.github/workflows/ci.yml` | Python 3.10/3.12 离线测试、规则校验、构建；只读 contents 权限 |
| `examples/demo-report/summary_facts.json`、`summary.md` | 真实 E01 Agent 的校验事实与四节公开摘要 |
| `examples/legacy-demo-report/analysis.json`、`report.md` | 原 Schema 1.0 历史快照完整保留 |
| `docs/validations/V1_5_S1_1_Public_Demo.md` | 真实中文请求、实际 CLI 与公开产物核对 |
| `docs/validations/V1_5_S1_1_Agent_E2E.md` | 真实会话结果、E10 失败/修正/复验与 CI 状态 |
| 本文件 | 阶段结果、文件用途、限制与交付清单 |

仅解除跟踪以下 Windows 下载元数据，本地文件及关联规范原文未删除或修改：

- `docs/RDK_X5_ONNX_Doctor_V1_2_Multi_Op_BPU_Development_Spec.md:Zone.Identifier`
- `docs/RDK_X5_ONNX_Doctor_V1_3_Shape_Inference_And_Attention_BPU_Development_Spec.md:Zone.Identifier`

用户提供的正式开发规范原先未跟踪，本轮不代替用户将它提交。没有自动 commit/push/PR。

## 架构与真实验收

Python 校验完整分析并提取有界事实，不重复 ONNX 解析或新建诊断算法。
Agent 选择 mode、事实、标题、顺序和连接短语；草稿使用 `{{FACT.ID}}`。发布器重新提取事实包并严格比对（包括类型），核对必需引用和来源 SHA，再替换为已核验事实并进行 Markdown 转义。
任意自由中文扩写返回 review_required 并拒绝。没有通过关键词黑名单声称全语义形式证明。

真实会话累计 **11 次：10 PASS、1 FAIL**。第一次十案例为 9 PASS/1 FAIL；E10 首次确实绕过发布器，用临时文件 replace 了测试摘要，完整失败证据保留。强化预先检查和禁止绕过后，全新 E10_retry 只检查存在并停止，原文件不变。不能宣称首次 10/10 或未来 Agent 文件权限绝对安全。
公开默认摘要采用独立 E01 新会话的真实产物，非 Python 七段模板，也非实施会话自问自答的路由验收。
详情见 [真实 Agent 验收](validations/V1_5_S1_1_Agent_E2E.md) 与 [公开演示](validations/V1_5_S1_1_Public_Demo.md)。

## 未执行事项与限制

- 远程 CI 未触发：NOT_RUN；本机 Python 3.12 未安装，该本地矩阵项 NOT_EXECUTED。
- 只支持经验证的 analysis Schema 1.3，不静默补造旧字段；三模式事实包与草稿各为 Schema 1.0。
- 新事实包首版不读取官方 sidecar。Profile 仅比较当前用户配置，不能证明编译支持。
- 主图统计保持不变；嵌套子图未展开、不计入诊断。I/O 是 ONNX graph inputs/outputs，可能包括 initializer。
- 受限句式允许 Agent 组织，尚不支持任意自由改写；有限 lint/绑定不等于全语义证明。
- 发布器不覆盖，但通用 Agent 的其他写文件工具不受发布器直接控制，E10 的真实失败体现此限制。
- 没有 Docker、hb_mapper、量化、编译、板端、ONNX 修改、S2/S3、规则扩展、GUI 或外部 LLM SDK/API 的引入。
- 网络活动仅包括构建依赖下载、已有 Codex 客户端的模型服务和插件初始化；没有任务官网检索。
