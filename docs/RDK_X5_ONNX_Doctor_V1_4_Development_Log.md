# V1.4 开发记录：Skill 自主工作流与 Opset 11

日期：2026-10-09（Asia/Singapore）。基线 HEAD `f02d8d5102ea3fd7e7b56a45b4f5f32df165585d`。
初始工作区仅有用户新增的 V1.4 规范及 Zone.Identifier，均保留，未改动。
本文件仅记录通用开发事实；实际模型路径、节点、SHA、在线检索与完整交互明细保存在忽略的 reports，不自动提交/推送。

## P28～P36 交付与最终阶段验收

| 阶段 | 实际工作及最终门槛 | 状态 |
|---|---|---|
| P28 | 重读仓库/规范/历史验收；CLI help、49条 validate、327项 pytest；实际扫描9模型并加入指定可读Opset11历史样本，10份旧版分析和SHA基线；提取两目标样本真实未覆盖清单 | PASS |
| P29 | Skill按完整预检、版本、节点、资源、知识、候选等意图分支；相对参考、frontmatter校验；用自然语言路径验证，不强制所有场景联网或运行整套命令 | PASS |
| P30 | 唯一用户profile YAML；preflight纯函数/CLI及独立JSON，11精确MATCH，其它版本MISMATCH，缺失/歧义/非法UNKNOWN，自定义域分开；旧schema1.0～1.3查询兼容 | PASS |
| P31 | 实际查阅X5手册1.1.2和英文2.0.0，区分ONNX/BPU/CPU；主RDK页面访问失败明确记录；检查真实schema11属性；官方审核台账与排除项 | PASS（主页面不可访问，使用已实读交叉手册，不冒称主页面已核实） |
| P32 | Reshape/Split/MaxPool/AveragePool四个Opset11-only包，新增17条（15自动、2非阻塞review）；旧10包/49条逐字不变；正常/边界/越界/动态/缺元信息/域版本/default属性回归 | PASS |
| P33 | Agent按实际未覆盖三元组去重、最多优先10类；知识sidecar契约/离线校验脚本；10类合成手册场景真实Agent离线执行，含失败、冲突、注入、X3/Caffe/CPU混淆和去重；无抓取服务 | PASS |
| P34 | Profile、本地规则、官方资料、实际运行分层；无额外数值限制不造PASS，review不硬判FAIL；缓存读取时间与原审核日期分开；analysis schema仍1.3 | PASS |
| P35 | 独立Codex实际执行S01～S08：8/8完成；真实官方web读取为S06，S01/S07/S08使用CACHED；禁联网S05/S08离线；中文回答、命令退出码、源资料和sidecar均保存 | PASS |
| P36 | 全量416 passed；rules validate/list：14类66条；10模型逐节点前后差异、SHA守恒；离开源码目录的独立venv、wheel和由sdist离线构建的wheel安装/查询/分析；diff与用户修改保护审计 | PASS |

阶段表表示修正后的最终门槛，而非声称开发中每次运行立即成功。过程中所有失败原始日志保留在baseline目录：CLI新分支误插入load_analysis导致NameError，修正回main；Split轴证据漏split_sizes后补齐；旧测试把Registry固定为0.3.0/10包/49条，改为核对旧49条仍保留及新包66条，V1 Conv历史结果断言未删。后续完整测试全部通过，没有隐藏失败或跳过。

## 改动范围

- `.agents/skills/rdk-x5-onnx-doctor/SKILL.md`：唯一主控工作流、意图分支、证据驱动查询、停止条件、官网查证与禁止事项。精简入口，细节分到references。
- Skill `references/toolchain_profile_opset11.yaml`：当前用户配置唯一机器真源；`diagnosis_decision_tree.md`、`official_lookup_workflow.md`、`answer_contract.md`：按需引用。
- `toolchain_profile.py`/CLI：轻量确定性preflight，不重建解析器；显式--profile可选择用户其它配置，绝不根据模型猜配置；无法确认标准主域时UNKNOWN。checker合法性、用户版本MATCH、真实编译支持互不等同。
- `basic_op_extractor.py`及既有字段/schema白名单：只读元信息、布局来源和有界参数缓存；新YAML仅标准域Opset11。Pool仅审核2D上下文，有效auto_pad值未推导时UNKNOWN；单位dilation不误判。
- Registry0.4.0、四个0.1.0子包；包0.5.0。原49条、版本/来源和10份原包字节未改。
- `official_knowledge.py`、Skill `scripts/validate_lookup.py`：只校验Agent写出的独立sidecar，严格来源allowlist、X5 ONNX/列、日期、状态组合、去重及runtime=false；不联网、不执行网页、不证明引文真假、不动态生成YAML。
- 官方审核台账、README、AGENTS、MANIFEST与回归测试。sdist含Skill YAML和脚本，wheel含全部规则，离仓使用preflight显式传入用户profile。

## 实际证据驱动的修正

1. 两本官方手册的Split“split数应可以整除”均未定义所指对象。初版将其解释为输出数，真实不等分块反例显示会产生不充分的FAIL；保留观察L%output_count，但变为非阻塞review_only，明确长度对每块倍数条件继续自动检查。不是修改旧规则，也没有宣称编译器接受不等分块。
2. 独立Skill调用发现`nodes --search Shape`是子串搜索，可能匹配Reshape和Tensor关联节点；入口要求用JSON的op_type精确筛选后统计，不改旧搜索语义。
3. 缓存读时刻与原资料审核日期此前只有notes区分；source新增original_reviewed_on，CACHED必填，checked_at保留实际缓存读取时刻。四组真实sidecar经新契约重新校验。
4. 第二梯队缓存仅提供文档地位，不足以完整判断ArgMax/Gather/Div数值条件；明确部分缓存边界，不填猜测条款或虚假PASS。

## 测试与真实回归

- 基线：327 passed。最终：**416 passed in 18.58s**，failed=0、skipped=0（实际pytest日志，无预计值）。新增89项，旧测试保留历史断言；原规则包逐字对比通过。
- 真实模型共10个：2个主域Opset11 MATCH，8个非目标版本MISMATCH。通用解析全部成功，全部原始SHA不变。
- 两目标样本未覆盖节点分别28→14、96→76；新变化只来自审核后的四包。节点ID/名字/输入输出/属性/边、Canonical Tensor、Shape证明和未知溯源、理论资源及候选完全守恒，旧规则诊断逐节点相同。
- 重要历史样本的5个Mul静态FAIL全部保留，已知载荷231→403、有界428轴证明及0冲突保持。新Split歧义保留review，不产生额外硬FAIL。
- S01～S08自然语言实际执行率8/8；另K01～K10是独立的MOCK_OFFLINE语义测试10/10，绝不计作真实联网。模拟注入指令未执行。
- 所有作出知识结论的真实记录保留URL/章节/版本/列/查询状态与日期，在线/缓存分别标记；空查询不声称已查官网。独立web证据在S06，开发来源实际访问失败/成功状态在baseline。
- 独立venv无法使用系统ensurepip，采用venv --without-pip及现有pip --python离线安装；未修改系统包或ROS配置。wheel及sdist构建成功，外部/tmp工作目录验证包导入、15份YAML（14包+manifest）、66条规则、analyze与preflight，未依赖editable导入。

## 验收清单与保留边界

A Skill自主工作流、B用户Profile准入、C有来源新规则与旧49条、D按需官方证据及失败退化、E测试/安装/只读与交付：均PASS。真实编译器、量化、BPU放置、设备内存、数值/任务指标/速度：**NOT_EXECUTED，按规范禁止执行**，不作为静态验收FAIL。

官方1.1.2/2.0.0为本次实际查阅版本；主RDK页面不可访问，当前仍有Split措辞歧义；非目标Opset14/20不扩硬件规则。Relu/Transpose只有独立知识记录，保持NOT_COVERED。Shape/Constant折叠陈述不证明原图节点独立执行。

当前目标版本仅源于用户profile，不宣传官方所有X5工具链只支持11。toolchain_version=unverified、compiler_checked=false、runtime_placement_verified=false。联网能力依赖Agent环境，缓存不代表实时网络核查。

## 本地产物

- `reports/v1_4_baseline/`：旧版分析、阶段失败/最终测试、源资料审核、schema、命令日志、旧包不变证据。
- `reports/v1_4_regression/`：10模型最终analysis/report/preflight、关键节点inspect/trace、前后差异及SHA记录。
- `reports/v1_4_skill_e2e/`：8个真实Codex自然语言场景的请求、实际命令/源资料/回答和自评。
- `reports/v1_4_skill_mock/`：10个明确合成离线知识场景，含注入拒绝证据。
- `reports/v1_4_dist/`：wheel/sdist、独立安装检查与日志。

这些目录被Git忽略，不上传；工作区新增/修改文件留待用户审查，未自动add、commit或push。用户原规范及其Zone.Identifier保持原状。完整工作区清单另存本地baseline，后续如用户更改工作区以实际git status为准。

## 用户授权提交前复核（2026-10-09）

用户随后明确要求“把1.4也提交了”，授权提交 V1.4 开发成果。此前离线官方手册已独立提交并同步远程；本次提交保留并引用该手册，模型及其检测报告仍不入 Git。

当前完整测试：**429 passed in 18.41s**，包含后续增加的 13 项离线手册回归；66 条规则校验、Skill 校验及离线原文/许可完整性校验通过。上文 416 项为 V1.4 原始验收结果，保留原始时间与范围。

本次同步整理 `.gitignore`，仅新增 Windows `*:Zone.Identifier` 下载元数据忽略项，保留原始附属文件。官方手册不在忽略范围内。本次授权执行本地提交，未执行远程 push。
