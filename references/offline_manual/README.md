# 官方算子手册离线调用

官网访问失败、用户要求离线，或需要稳定可追溯的原文时使用此目录。无需联网、浏览器、额外 Python 依赖或运行文档网站。下载范围为完整“模型算子支持列表”章节，不是全站镜像。

## 来源和许可

- 官方维护者：D-Robotics；[固定版本的原始 Markdown](https://github.com/D-Robotics/rdk_doc/blob/2ab42f3c295c1dd1fa94fbf0df24a81442cd642b/docs/07_Advanced_development/04_toolchain_development/intermediate/supported_op_list.md)。
- [supported_op_list.md](supported_op_list.md) 与 [LICENSE_UPSTREAM](LICENSE_UPSTREAM) 均按字节原样保存。上游 Apache-2.0 许可允许再分发；本地查询脚本与说明是新增文件。
- [manifest.json](manifest.json) 记录下载时间、上游 Git revision、来源 URL、原始文件大小与 SHA256。
- 这是官方源码快照，手册自身没有明确发布版本号。不能将它冒充 X5 SDK 1.1.2、2.0.0 或当前网页最新版本。在线网址仅用于交叉核对；尚未证明源码快照与现行网页逐字一致。
- 原文含 X3、X5、Ultra 和 Caffe/ONNX 多表。脚本只检索 **RDK X5支持的ONNX算子列表**，当前 159 条；使用限制上下文完整保留，Agent 必须分辨其中的芯片范围。

## 调用

从仓库根目录运行；脚本只使用 Python 标准库，也可以换成 `python3`：

```bash
.venv/bin/python references/offline_manual/query.py --verify --json
.venv/bin/python references/offline_manual/query.py --list
.venv/bin/python references/offline_manual/query.py --operator Transpose --json
.venv/bin/python references/offline_manual/query.py --operator Relu --json
.venv/bin/python references/offline_manual/query.py --operator Shape --json
.venv/bin/python references/offline_manual/query.py --usage
```

`--operator` 必须精确匹配官方表的大小写和完整名称；如 `GridSample（PyTorch）` 保留原标签，不能自动当成标准 ONNX domain 的 GridSample。每次调用核对手册与许可 SHA256、标题、列名、条目完整性，失败退出码 2 并输出 `FAILED / SOURCE_UNAVAILABLE`。缺失精确名称仅表示本快照没有该标签，不能推断硬件不支持。

查询输出是独立的原文检索 JSON，不是 `official_lookup.json` schema，也不是诊断结论。字段分别保留官方执行标签、X5 BPU 支持约束、CPU 支持约束、原始行与行号，并附使用限制上下文。`lookup_status=CACHED` 表示本次本地读取，`checked_at` 是读取时间；原文尚需人工/Agent 按实际节点、domain、opset 和工具链版本审查。

## Skill 使用约定

1. 从实际 `analysis.json` 的未覆盖节点选择算子，先核对目标 profile、domain 和 imported opset。
2. 使用以上命令读取本地条目及上下文；CPU 条件不能转为 BPU FAIL，执行标签不能当作实际部署位置。
3. 官网失败时保留本次网络失败记录，再明确改用这个固定源码缓存。缓存读取永远不标 FETCHED；下载日期不是全表语义审核日期。
4. 如需 V1.4 官方知识 sidecar，Agent 必须先审查引用原文、记录真正的审核日期和版本；当前 sidecar URL allowlist 不含 GitHub 固定源码，不得把本查询输出强行套入旧 schema 或伪装成网页原文。可将本查询 JSON 作为独立离线证据，报告准确的 `upstream_url`、revision、行号与 SHA256。
5. 保留与已审查 YAML、SDK 手册或现行网页的差异；不自动升级或生成规则，不改变离线 analyze，也不修改 ONNX。

更新时从可信上游固定 commit 获取原文及许可，审查差异，重新记录 revision/时间/哈希和实际行数并运行测试。禁止只修改哈希来掩盖异常内容。模型查询产生的 JSON/报告保存于 Git 忽略的 `reports/`；本目录仅同步官方资料和通用查询工具。

## 本次验证（2026-10-09）

完整当前开发工作区测试：429 passed（19.01 秒）；其中新增离线手册测试 13 项。CLI `--help`、66 条规则校验和 Skill 校验通过。实际离线查询 Transpose（456 行）、Relu（414 行）、Shape（431 行）和缺失标签通过。回归覆盖哈希损坏、缺失文件、错误芯片标题/列名、重复条目及仓库外调用。

注意 Shape 表内标“BPU加速”，BPU 列说明“会通过常量折叠将其优化为数值存储”，使用限制又说明它不能直接运行于 BPU；必须结合上下文解释。初次新增测试错误假定 Shape 标签为 CPU，并以错误空格构造重复行；已按真实源码修正测试，不改官方原文或既有 BPU 规则。

此记录基于本地 V1.4 开发工作区；离线手册可独立运行，不需要未提交的 V1.4 代码。本次远程同步仅包含本目录、通用来源入口及新增测试；不包含 V1.4 开发修改、模型文件或模型报告。
