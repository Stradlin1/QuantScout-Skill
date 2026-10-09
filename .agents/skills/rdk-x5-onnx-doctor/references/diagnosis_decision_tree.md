# 按证据选下一步

完整预检：路径/格式 → 用户 profile → 已证明 FAIL → 阻塞 UNKNOWN → 未覆盖类型 → 资源/候选。

- analyze exit=2：读取错误，区分 checker 与文件/参数问题；不输出兼容性肯定结论。
- preflight MISMATCH：明确是当前用户目标不匹配，不修改 opset 标签。通用图分析仍有效。
- 存在 FAIL：inspect 获取 actual/expected 和官方 BPU 来源。若问影响哪些输出才 trace；路径可达不是下游违规。
- UNKNOWN：先按字段/来源去重。缺 rank/尺寸查 shape；缺布局查真实 Conv 连接；缺工具链配置保持未知，不凭 rank 宣称 NCHW。
- NOT_COVERED：检查是否域/版本不适用。官方有条目、仅 CPU、折叠/转换、来源不可访问是不同情况；不能用知识记录覆盖机器诊断。
- 资源未知：列真实符号维度、unsupported transfer 或 missing metadata，不补猜。已知子集 Top 10 不代表全部特征图排名。
- 无 FAIL 但仍有未覆盖：不能说全图适合量化。输出已核验的范围和关键缺口。
- 单节点/单资源问题：完成用户所问即结束，不扩展成整套分析。

结束条件：每项主要结论有 node/Tensor/rule/proof 或 profile/source 证据；列未执行事项和产物。没有模型路径、没有旧报告所需字段、网络失败等只阻塞依赖步骤，不制造结果。
