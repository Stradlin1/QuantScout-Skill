---
name: rdk-x5-onnx-doctor
description: 通过终端检查 ONNX 的 RDK X5 BPU 静态约束、分析 Conv2D 及 Mul/Sigmoid/Add/Concat/Slice/Gemm/MatMul/Softmax/Resize 与 Shape 溯源、定位异常并追踪输出依赖。用户说“分析这个 ONNX 是否适合 RDK X5”“检查 Conv”“查看风险路径”“分析输出 Tensor 大小和大特征图”“找冗余节点、逆 Transpose、no-op Reshape”“量化前优化候选”时使用。
---

# RDK X5 ONNX Doctor V1.3

1. 在本仓库根目录确认模型可读及 .venv；执行 --help、rules validate，核对 Registry、子包、官方来源版本和模型 opset，保持工具链 unverified。内置规则可离线使用；来源说明在 references/。
2. 选择新的、Git 忽略的 reports/<模型名>-<时间>，只读 analyze --model <路径> --out <目录>；确认 JSON/Markdown 和退出码。参数独立引用，不把模型名称作为 shell 代码。
3. 从覆盖矩阵和静态 FAIL 选择重要节点；用 nodes、inspect、trace 核对实际 Tensor/属性、规则 actual/expected 和官方 X5 BPU 栏证据。依赖可达不表示下游违规或 CPU 回退。
4. 先查看 shapes --analysis <JSON> --summary；遇到符号维度或资源未知，调用 shape --analysis <JSON> --tensor <名称> --json，检查逐轴证明、冲突和首阻塞节点，按共同来源分组调查。unk__* 不等于动态外部输入；旧报告需重跑 analyze，不猜补。
5. MatMul/Softmax/Resize 重要时 inspect/trace 其真实节点及上下游；区别 ONNX 合法性、静态硬件条款、Softmax BPU 路径资格、run_on_bpu 配置和实际执行分配。不能从原名猜 PyTorch 转换来源。
6. 资源问题用 tensors --kind output 和 --kind intermediate --sort bytes --limit 10，再 tensor 查看生产者/消费者。理论原始载荷、假设 INT8 和实际设备内存分开，载荷之和不是峰值。
7. 优化问题用 candidates/candidate，再依证据检查节点和路径。保留五类模式与三种分类；信息不足不是已证明冗余，审查候选不等于可以自动删节点。
8. 解释绑定真实 node_id/Tensor/rule_id/proof_id 的事实、缺失条件及可验证步骤。需要模型方向时仅提供架构层建议，例如审查输出头与后处理分离；由模型所有者在独立训练/导出工程实施，再导出对比。不得搜索其它工程、猜源码位置、生成具体训练代码补丁或修改 ONNX。
9. 给出保存报告及查询路径，说明未做工具链/板端实测、真实内存/精度/性能不可确定。模型报告和日志仅留本地 reports/；开发 Markdown 可用于周期内审查，最终模型验收 Markdown 不入 Git。无自动 commit/push。

禁止 Docker、hb_mapper、PTQ、部署/板端推理、ONNX rewrite/simplifier、自动优化、评分、GUI/HTML/浏览器。详细语义见 docs/ANALYSIS_SCHEMA_V1_3.md、REPORT_POLICY_V1_3.md 与 references/x5_attention_resize_rule_review.md。
