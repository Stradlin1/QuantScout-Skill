# Conv2D 规则审查记录

规则版本 0.1.0；审查日期 2026-10-09；真实工具链版本 unverified。

## auto_check

- Kernel H/W：闭区间 [1,31]，支持从权重元信息推导省略属性。
- 每组 Kernel C×H×W：≤32767。C 为权重第二维，每组输入通道，不能使用总输入 C。
- Stride H/W：[1,256]；Dilation H/W：[1,16]。
- 有效 padding：[0,256]，顺序 top/left/bottom/right。ONNX auto_pad 按语义计算；动态空间维度不能确定 SAME padding 时 UNKNOWN。
- 任一 dilation >1 时，两个 stride 都须为 1；两个空间输入分别被相应 dilation 整除。符号化空间维度 UNKNOWN。

## conditional

- 膨胀卷积实际量化后输出必须是 int8。原浮点 ONNX dtype 不能证明量化精度，条件触发后一律 UNKNOWN。
- 常规输入/输出每组通道限制 8192、量化子图最后 Conv 放宽至 65536。检查到相关通道 >8192 时仅提示 UNKNOWN，不能仅凭图输出或下一节点认定末端量化子图。
- Kernel N/C 各维上限 8192 与通道放宽之间是否有独立适用条件存在解释空间。避免未经编译验证的 false FAIL，本版不自动应用这两维硬上限；kernel H/W 与体积上限仍独立检查。

## review_only（未伪装成已执行规则）

- Conv→Add：主 RDK 页的 shortcut stride 描述为 2；1.1.2/2.0.0 手册为 {1,2}。版本冲突保留在 manifest exclusions。不仅凭后继 Add 套用；尚未实现融合模式识别。
- auto_pad 不支持、pads 对称、float/int32/int8 在原表的 CPU 支持列。本版不把这些条款挪到 BPU 支持列。非对称 padding 不自动判 FAIL。
- 通用 shape 上限、张量字节限制、对齐后的实际存储、DDR、设备内存、所有算子合计算力和时延不在本版覆盖范围。
- Conv1D/Conv3D 记录为 NOT_COVERED，逐条 Conv2D 规则 NOT_APPLICABLE；非标准域 Conv 或缺 rank 时 NEEDS_VERIFICATION。
- 非 Conv 节点全部 NOT_COVERED。If/Loop 子图不解析，报告显式记录。
- 不自动推断原节点是否被融合、实际 CPU/BPU 划分或量化误差。

## 数据和解释限制

形状推断失败保留图与原始维度；缺 external data 时以权重元信息继续分析，并对替代输入的内存副本进行结构校验，不声称原模型完整 checker 成功。禁止加载越出模型目录的 external data 路径。文件摘要仅覆盖 ONNX protobuf，不包含外部权重内容。

NO_VIOLATION_FOUND 表示已执行本地规则没有违规；本次未检查的条件仍在 manifest/report 中列出。模型信息矛盾单独标为待验证，不假装有官方 BPU 违规证据。
