# ONNX 检查事实摘要

## 模型概况

模型名称：demo.onnx；OpSet 导入：&#91;&#123;"domain": "", "version": 11&#125;&#93;；已解析主图节点 5 个。

## 静态检查

NO&#95;VIOLATION&#95;FOUND 2、VIOLATION 1、NEEDS&#95;VERIFICATION 0、NOT&#95;COVERED 2；四项合计 5 个主图节点；唯一违规主图节点 1 个；FAIL 规则记录共 1 条。

## 已确定冲突

代表 FAIL：节点 main/node&#95;000000（原名 oversized&#95;kernel），算子 Conv，规则 X5-CONV2D-KERNEL-H，字段 kernel&#95;h，实际值 32，允许值 &#123;"max": 31, "min": 1&#125;；另外 0 条 FAIL 记录未展示。

## 检查范围

未提供 Profile 验证结果；本次总结未执行 Profile 检查。仅统计已解析主图节点；嵌套子图未逐节点展开、不计入诊断统计。本次静态检查未执行实际量化、编译或板端测试；未确认 CPU/BPU 分配；静态状态不等于完整兼容。
