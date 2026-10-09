# V1.2 规则审查与排除

具体依据见 [来源记录](x5_multiop_sources.md)。新增规则采用手册 1.1.2 的 X5 ONNX BPU 栏，并与主站和英文 2.0.0 核对；本轮六类条款未发现版本冲突。原 Conv 子包 0.1.0 内容与判定不变，Registry 总版本升级至 0.2.0。

高维 Add/Mul 的算法先尾轴补 1；去除输出等于 1 的轴；其余轴分类为双方不广播、仅 A 广播、仅 B 广播。只合并相邻且类别相同的轴，类别切换形成独立段；段数≤4 才证明可合并。静态 ONNX 非法广播单列 extraction issue，不伪装硬件 FAIL；符号/零维无法完整证明时保持未知。

常量仅认定不可覆盖 initializer 或标准域 Constant 输出。graph input 包含 initializer 时可覆盖，固定性未知；其它算子产生的值作为 feature 值，不执行常量折叠。来源完全缺失保持未知。

Concat 的 N 轴结论只在明确 batch 布局下执行（显式 NCHW 元信息，或 Conv 数据边及可靠的保 rank 路径；仅 rank4 不够）；其它 rank 即使 axis=0 也不自动推定为 N。负轴必须在已知 rank 下归一化。轴非法/输入 rank 冲突属于 ONNX 问题。

Slice 仅审查来源/有界索引事实；没有凭空建立 starts、steps、dtype 或 rank 数值 FAIL。动态参数和实际转换均为待验证。Gemm 逻辑 M/K/N、alpha/beta/transA/transB 不代入四维 Conv；转换布局未知，不能构造可信硬件 FAIL/PASS。两类能力陈述采用 review_only，不能因没有硬约束就报告全条件通过。

原始 FP32、INT64 形状参数与量化后精度分开。Add shortcut、int16 能力、一般大小/对齐和设备分配仅记录未验证事项。任何结构或导出变更须回到训练/导出工程，重新导出后复检，本项目不修补 ONNX。

Mul 若使用 rank0 标量常量；三份 X5 来源均写输入/输出 rank1–10，应保留静态条款冲突。ONNX scalar 广播本身合法；编译器是否补维/折叠未验证，不据此断言部署失败。
