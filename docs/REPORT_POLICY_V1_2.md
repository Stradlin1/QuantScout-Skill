# Markdown report policy 1.2

analysis.json 是完整机器证据；report.md 是异常优先的人类摘要，不删除 JSON 事实。

按实际版本生成九部分：版本与模型、摘要、按算子覆盖矩阵、确定违规、按算子和 reason_code 分组的未知、Tensor 资源、优化候选、必要工程建议、来源及未验证边界。模型只显示 basename 和 hash。正常节点不列“无需修改”。

确定违规保留总数，最多展示前 50 个节点；逐条保留节点 ID/原名/Tensor、规则实际允许值、官方版本 URL 和依赖输出。UNKNOWN 每组最多 10 个代表节点；候选最多 10 个；明确提示省略数量。所有记录可通过 CLI 在完整 JSON 中查询。同一节点可出现在多个未知原因组，组计数不可当作节点总数。

资源展示输出、已知中间载荷 Top 10、未知数量及原因；FP32/FP16 原始载荷与假设 INT8 分开，不能称作峰值或部署内存。候选区分 SEMANTICALLY_REDUNDANT/REVIEW_REQUIRED/INSUFFICIENT_INFORMATION。本模型零候选时如实呈现。

建议绑定实际异常/未知及其依据；涉及结构或导出变化时指向训练/导出工程重新导出复检，不能指导 ONNX 原地编辑。未覆盖与 UNKNOWN 不能推断 CPU 回退、量化精度或性能。用户名称和 Tensor 转义 Markdown/HTML/control 字符；所有真实节点身份保留于 JSON。
