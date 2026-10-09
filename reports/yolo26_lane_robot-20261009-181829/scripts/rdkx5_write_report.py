from pathlib import Path
import json, collections, hashlib, shutil, subprocess, platform
from datetime import datetime
from zoneinfo import ZoneInfo

root=Path('/home/xhm/lianghua_ws/skillzuoye')
out=Path(Path('/tmp/rdkx5-validation-dir').read_text())
d=json.loads((out/'analysis.json').read_text())
b=json.loads((out/'before_fix/analysis.json').read_text())
a=json.loads((out/'independent_audit.json').read_text())
e=json.loads((out/'graph_evidence.json').read_text())
meta=json.loads((out/'validation_metadata.json').read_text())
ts={t['name']:t for t in d['tensors']}
ns={n['id']:n for n in d['nodes']}
rs={r['name']:r for r in d['resource_analysis']['tensor_records']}
e['symbolic_conv_input_count']=sum(any(type(v)is str for v in ts[n['inputs'][0]]['shape']) for n in d['nodes'] if n['op_type']=='Conv')
(out/'graph_evidence.json').write_text(json.dumps(e,ensure_ascii=False,indent=2))
lines=[]
def add(s=''):lines.append(s)
def table(headers,rows):
    add('| '+' | '.join(headers)+' |');add('| '+' | '.join('---' for _ in headers)+' |')
    for row in rows:add('| '+' | '.join(str(x).replace('|','&#124;').replace('\n',' ') for x in row)+' |')
    add()
def code(v):return '`'+str(v)+'`'
def node(nid):return code(nid)+' / '+code(ns[nid]['original_name'])

add('# YOLO26 真实 ONNX 静态诊断验收报告')
add();add('完成时间：'+datetime.now(ZoneInfo('Asia/Singapore')).isoformat(timespec='seconds')+'。本报告是实际终端执行及节点级复核结果。')
add();add('真实模型分析成功。初始 138 项测试通过；修复 Reshape 候选筛选缺陷并新增 3 项回归后，141 项通过。最终 281 个节点全部解析、47 个 Conv2D 已收录规则未发现违规、五种模式均无实际优化候选。172 个中间 Tensor 的尺寸未知，已保留原因；不能据此声称模型完全兼容 BPU或已掌握全图最大的 Tensor。')
add();add('没有执行 Docker、hb_mapper、量化、结构优化、节点删除、板端推理、评分、GUI 或 Git commit/push。实际部署内存、数值等价、精度、CPU/BPU 划分和性能未测量。')
add();add('## 1. 测试环境');add()
table(['项目','实际值'],[
    ['系统','Ubuntu 22.04.5 LTS / WSL2 / x86_64'],['Python',a['python']],['解释器',str(root/'.venv/bin/python')],
    ['包版本','rdkx5-onnx-doctor 0.2.0 / analysis schema 1.1 / ruleset 0.1.0'],
    ['Git',meta['branch']+' / '+meta['head']],['初始工作区','干净；无预先存在用户修改'],
    ['模型',code(meta['model_path'])],['文件大小',str(meta['bytes'])+' B'],
    ['SHA256（分析前）',code(meta['sha256_before'])],['SHA256（分析后）',code(a['sha256_after'])],
    ['原始文件完整性','相同；仅只读加载，未移动、覆盖、删除或另存原始模型']])
add('依赖精确快照见 [environment.stdout](environment.stdout)。核心版本：ONNX 1.23.2、NumPy 2.2.6、NetworkX 3.4.2、Pydantic 2.14.0、PyYAML 6.0.3、protobuf 7.36.2、pytest 8.4.2、build 1.6.1。`pip check` 成功。')
add();add("环境配置过程：初始 `.venv` 不存在。`python3 -m venv .venv` 因系统缺少 ensurepip 失败；sudo 安装 python3.10-venv 因需要密码未执行成功。随后使用 `venv --without-pip`、官方 https://bootstrap.pypa.io/get-pip.py 引导项目 pip，并完成 `.venv/bin/python -m pip install -e '.[dev]'`。下载依赖需要获准的沙箱外网络访问；系统 Python 未安装项目依赖。")
add();add('首次 pytest 在收集前被系统 `PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages` 引入的 ROS launch_testing 插件阻塞，异常为 `ModuleNotFoundError: lark`。清除该外部 PYTHONPATH 后完整运行全部项目测试，未禁用项目测试或删减条件。首次失败的 stderr 也已保留。')
add();add('## 2. 测试结果');add()
table(['检查','结果','证据'],[
 ['CLI --help','退出 0，九个子命令','help.stdout'],['rules validate','退出 0，19 条原规则','rules_validate.stdout'],
 ['首次 pytest','收集前失败：ROS 插件环境污染','pytest_baseline.stderr'],['隔离后基线','138 passed in 0.96s','pytest_isolated.stdout'],
 ['最小案例，修复前','2 failed / 1 passed（预期复现）','regression_before_fix.stdout'],
 ['最小案例，修复后','3 passed / 41 deselected','regression_after_fix.stdout'],['最终完整回归','141 passed in 0.93s','pytest_final.stdout'],
 ['初始真实 analyze','退出 0，0.935 秒','before_fix/analyze.stdout'],['修复后真实 analyze','退出 0，0.922 秒','analyze.stdout'],
 ['节点 / Tensor / 连边逐项比对','PASS；与源 ONNX 一致，无遗漏节点','independent_audit.json'],
 ['尺寸整数运算核对','231 个已知记录全部一致；172 个未知保留原因','independent_audit.json'],
 ['修复影响范围核对','除 optimization_candidates 外全部 JSON 顶层段与修复前相同','graph_evidence.json'],
 ['原始模型 SHA256','前后相同','validation_metadata.json / independent_audit.json']])
add('模型 checker 成功、shape inference 无异常、CLI 分析无异常终止。全部保存的查询（含修复前 9 次 candidate 查询）均退出 0；首次环境失败及预期失败的回归案例明确区分。未重新做 wheel 打包或实际运行推理；这些不属于本次静态模型验收范围。')
add();add('复现命令（仓库根目录）：');add();add('```bash')
add("env -u PYTHONPATH .venv/bin/pytest -q")
add('.venv/bin/python -m rdkx5_doctor --help')
add('.venv/bin/python -m rdkx5_doctor rules validate')
add('.venv/bin/python -m rdkx5_doctor analyze --model '+"'"+meta['model_path']+"'"+' --out reports/yolo26-new-run')
add('```');add();add('更多实际参数、耗时和退出码见 [command_log.jsonl](command_log.jsonl)，其中 `unset_env` 记录隔离 PYTHONPATH。验证脚本已随报告保存，不依赖模型名称硬编码到产品源码。')
add();add('## 3. 真实模型结构');add()
add('标准域 opset 11；1 个输入、2 个输出、281 个节点、23 种算子、118 个 initializer。节点 ID 为原始图顺序 `main/node_000000` 至 `main/node_000280`。')
add();table(['边界','Tensor','Shape','dtype','元素','原始 B'],[(kind,t['name'],t['shape'],t['dtype'],rs[t['name']]['element_count'],rs[t['name']]['raw_bytes']) for kind,key in [('input','inputs'),('output','outputs')] for t in d['model'][key]])
table(['算子','节点数','全部节点占比','BPU 规则覆盖'],[(op,count,f'{count/281*100:.2f}%', 'Conv2D 已检查' if op=='Conv' else 'NOT_COVERED') for op,count in sorted(d['model']['operator_counts'].items(),key=lambda x:(-x[1],x[0]))])
add('非 Conv 共 22 种 / 234 个节点，全部为 NOT_COVERED，不能报告为兼容或通过。实际输出为 lane/task 分支的 cls_logits 与 offset；不是根据 YOLO 文件名套用常见检测输出格式。')
add();add('原模型无 `value_info`（0 条），仅输入/输出提供静态 shape；边界不存在动态维度，自定义域节点 0、外部 initializer 0、缺失外部权重 0、嵌套子图警告 0。默认 shape inference 增补 282 条 value_info，其中合并边界后 113 条元信息静态、172 条含符号维度。它们是推断未求解的 `unk__*`，不能据此断言原模型实际存在动态输入。')
add();add('只读、内存中的 ONNX `strict_mode=True,data_prop=True` 对照实验仍产生 172 条含符号维度的元信息，与默认推断相同；未保存或优化模型。缺失中间元信息 / 推断不完备是资源分析限制，不是 shape inference 抛出异常。详见 [independent_audit.json](independent_audit.json)。')
add();add('## 4. Tensor 资源分析');add()
add('全图去重 403 条 Tensor 记录：344 条 float32、59 条 int64；无 FP16 实际记录。已知 231 条、未知 172 条，均由独立整数 oracle 核对。字节数 = 所有静态维度乘积 × dtype 字节宽度；FP32 为 4 B、INT64 为 8 B，假设 INT8 为元素数 × 1 B。FP16 若相同 shape，则逻辑载荷为元素数 × 2 B，仅为类型算术说明。')
add();table(['类别','Tensor 数','已知原始 B 之和','未知数','完整性'],[(k,v['tensor_count'],v['known_bytes_sum'],v['unknown_tensor_count'],v['completeness']) for k,v in d['resource_analysis']['summary'].items() if isinstance(v,dict) and 'tensor_count' in v])
add('以上是按类别去重后的逻辑载荷之和；中间已知之和 93,475,488 B 是 PARTIAL。它不是峰值内存，也不能与权重求和推断实际 BPU、DDR 或 SRAM 占用。fanout 不表示复制次数。')
add();add('### 全部模型输出');add()
table(['Tensor','Shape / dtype','元素','原始 B / MiB','假设 INT8 B','生产节点 / 消费者'],[(r['name'],str(r['shape'])+' / '+r['dtype'],r['element_count'],str(r['raw_bytes'])+' / '+str(r['mib']),r['hypothetical_int8_bytes'],node(r['producer_node_id'])+' / '+str(r['consumer_node_ids'])) for r in rs.values() if r['is_graph_output']])
add('两个输出都无消费者，分别由末端 Concat 275 / 280 产生。输出总原始载荷 145,152 B，假设 INT8 总载荷 36,288 B；不是实际量化结果。')
add();add('### 已知原始载荷最大的 10 个中间 Tensor');add()
add('因 172 个中间项未知，该表只保证在已知集合中排序；不能保证是全图实际最大的十个。相同字节数按名称排序，注意力的其它等大 Tensor 因 limit=10 未入选。');add()
table(['Tensor','Shape / dtype','元素','原始 B / MiB','假设 INT8 B','生产节点','消费节点'],[(code(name),str(rs[name]['shape'])+' / '+rs[name]['dtype'],rs[name]['element_count'],str(rs[name]['raw_bytes'])+' / '+str(rs[name]['mib']),rs[name]['hypothetical_int8_bytes'],node(rs[name]['producer_node_id']),', '.join(node(x) for x in rs[name]['consumer_node_ids'])) for name in d['resource_analysis']['summary']['largest_intermediates']])
add('首层三个中间项均为 [1,32,320,320] FP32：3,276,800 元素、13,107,200 B = 12.5 MiB，假设 INT8 为 3,276,800 B = 3.125 MiB。它们分别为 Conv、Sigmoid、Mul 的逻辑输出；不能断言运行时同时分配三个独立同大小缓冲区。')
add();add('### 多消费者与未知资源');add()
add('71 个多消费者中间 Tensor：13 个大小已知，58 个未知。完整列表及全部 172 个未知记录、原因、producer/consumer 在 [graph_evidence.json](graph_evidence.json)；CLI 原始列表见 [tensors_unknown.stdout](tensors_unknown.stdout)。')
add();table(['例子','生产者','消费者','解释'],[(code(name),node(rs[name]['producer_node_id']),', '.join(node(x) for x in rs[name]['consumer_node_ids']),reason) for name,reason in [('/model/model.0/conv/Conv_output_0','Sigmoid 与 Mul 共享 Conv 值，属于 SiLU 结构'),('/model/model.2/cv1/act/Mul_output_0','Shape 与两个 Slice 共用特征图；未知 shape 的起点位于下游 Slice')]])
add('172 条未知全为 float32 的符号维度，非 dtype 宽度未知；0 条缺 rank、0 条 dtype 缺失、0 条 unsupported dtype。按符号数量分组：110 条 rank4/3 个符号轴、26 条 rank4/1 个符号轴、12 条 rank4/4 个符号轴、24 条 rank2/1 个符号轴。')
add();table(['未知项生产算子','数量'],sorted(e['unknowns_by_producer_op'].items(),key=lambda x:(-x[1],x[0])))
add('首个未知项为 node 19 的 `/model/model.2/Slice_output_0`，shape 为 [unk__0,unk__1,unk__2,unk__3]；node 22 的另一个 Slice 类似。node 8 的输入特征已知 [1,64,160,160]，node 9 Shape、node 11 Gather(axis=0) 读取通道轴，node 14 Add / 16 Div / 18 Mul 计算 Slice 终点。小型内联整数证据为 axis=1、start=0、Add=1、Div=2、Mul=1；另一路 Mul=2。因此这一局部数学链可算出终点 32 / 64，两个通道切片局部应为 [1,32,160,160]。这是手工证据解释，不写回工具 JSON，不将后续未逐项证明的尺寸当作已知。')
add();add('V1.1 未做通用形状算术传播，ONNX 默认/增强 data_prop 推断都在该链未恢复静态空间维度。不通过权重或模型输出反向猜测未知输入 shape；例如 node 23 权重 [16,32,3,3] 能独立检查 kernel，却不足以让工具编造其 batch/H/W。')
add();add('## 5. RDK X5 BPU 约束检查');add()
add('47 个 Conv 全部确认 rank4 Conv2D；结果为 NO_VIOLATION_FOUND=47、VIOLATION=0、NEEDS_VERIFICATION=0、Conv NOT_COVERED=0。逐规则 517 PASS、376 NOT_APPLICABLE、0 FAIL、0 UNKNOWN。NOT_APPLICABLE 是触发条件不成立，不能把它报告为数值 PASS。')
add();table(['规则 ID','状态计数'],[(rid,counts) for rid,counts in a['rule_counts'].items()])
add('实测字段范围：kernel H/W 1–3；每组 C×H×W 9–2304；stride H/W 1–2；dilation H/W 均 1；所有 padding 每侧 0–1；每组输入通道 1–1024、每组输出通道 1–512。所有 dilation 条件和超常规通道条件都未触发，故量化输出 dtype 条件没有在本模型产生 UNKNOWN。')
add();add('来源证据按安装的规则包引用，未擅自更改、扩展或声称重新审核官方网页：')
for s in d['ruleset']['sources']:add('- ['+s['title']+']('+s['url']+')；'+s['version']+'；'+s['section'])
add();add('仓库审查记录见 `references/official_sources.md`、`references/conv_rule_notes.md`；规则 package 0.1.0 / toolchain_version=unverified。基础约束分别为 kernel 1–31、每组体积 ≤32767、stride 1–256、dilation 1–16、padding 0–256。node 0 的原始名称 `/model/model.0/conv/Conv`、权重 [32,3,3,3]、输入 [1,3,640,640]、stride [2,2]、pads [1,1,1,1]，逐规则证据在 [inspect_node_000000.stdout](inspect_node_000000.stdout)。')
add();add(str(e['symbolic_conv_input_count'])+' 个 Conv 的输入存在推断符号轴，但本次触发的 kernel/stride/dilation/pad 规则可由静态权重或属性判定；它们的资源大小仍未知。没有确定违规源，故违规溯源不适用，`nodes --status VIOLATION` 返回空列表。检查到输出的依赖路径，不赋予下游违规或 CPU 回退含义。')
add();add('未覆盖：全部非 Conv 的硬件支持；shape 总大小/字节上限；通道量化子图末端例外；Conv→Add shortcut 版本冲突；编译器融合；实际量化精度与布局。未覆盖不等于失败，已检查未违规不等于完整 BPU 兼容保证。')
add();add('## 6. 五种优化候选检测');add()
table(['模式','真实结构','最终候选数量 / 分类','结论'],[
 ['IDENTITY','Identity 节点 0','0','无'],['TRANSPOSE_INVERSE_PAIR','Transpose 节点 2','0','中间隔着 MatMul、Mul、Softmax，不是直接逆序对'],
 ['CAST_SAME_DTYPE','Cast 节点 0','0','无'],['RESHAPE_NOOP','Reshape 节点 11','0','11 个均有已知 rank 或轴变化证据'],
 ['CONV_BN_FUSION_REVIEW','BatchNormalization 节点 0','0','无可检测的 Conv→BN；不能据此声称编译器已融合']])
add('最终 SEMANTICALLY_REDUNDANT=0、REVIEW_REQUIRED=0、INSUFFICIENT_INFORMATION=0；没有值得删除或融合的已确认候选。不是为了凑齐五类制造候选。')
add();add('### 修复前的 9 个信息不足观察及逐节点排除证据');add()
add('以下 ID 属于 [before_fix/analysis.json](before_fix/analysis.json)，最终报告中已排除。初始工具没有把它们误标成已确认冗余，但忽略可用反证导致无意义观察。每条均实际调用了 candidate、inspect 和 trace；candidate 输出在 before_fix/candidate_OPT-*.stdout，接口、消费者、输出路径均已核对。');add()
for c in b['optimization_candidates']['candidates']:
    n=ns[c['node_ids'][0]];r=rs[n['outputs'][0]];x=ts[n['inputs'][0]];target=c['evidence']['target_constant']['values']
    oldquery=json.loads((out/'before_fix'/('candidate_'+c['candidate_id']+'.stdout')).read_text())
    add('#### '+c['candidate_id']+' / '+n['id']);add()
    add('- 原名：'+code(n['original_name'])+'；原分类 INSUFFICIENT_INFORMATION。')
    add('- 输入值：'+code(n['inputs'][0])+'，shape '+code(x['shape'])+'；producer '+node(x['producer'])+'。')
    add('- shape 输入：'+code(n['inputs'][1])+'，inline INT64 常量 '+code(target)+'。')
    add('- 输出：'+code(n['outputs'][0])+'，shape '+code(ts[n['outputs'][0]]['shape'])+'；consumer '+', '.join(node(z) for z in r['consumer_node_ids'])+'。')
    why='输入 rank2、目标 rank3，确定改变接口 shape' if len(x['shape'])!=len(target) else '已知输入第二轴 512 与目标第二轴 4 不同，确定改变 shape'
    add('- 识别/排除：初始因符号轴无法计算完整输入元素数而列观察；人工反证为 '+why+'；修复后排除。')
    add('- 结构限制：输入值 fanout='+str(len(x['consumers']))+'，输出 fanout='+str(r['consumer_count'])+'，该 Reshape 输出非公开模型输出；最终可达 '+code(oldquery['reachable_outputs'])+'。'+('输入还被其它节点共享，任意改写必须保留其它消费支路。' if len(x['consumers'])>1 else '输入值没有其它消费支路。'))
    add('- 语义与研究价值：不能当作无变化转发删除。当前 no-op 优化不值得继续研究；如研究其它等价图改写，必须保留目标 rank/shape、所有消费者、公开输出名称及 dtype，在独立副本上做 checker、shape/interface 与 ONNX Runtime 代表输入数值及任务指标验证。本次未进行这些运行时验证。')
    add()
add('另外 node 187 / 188 的 Reshape：输入 [1,4,64,400]，目标 [1,256,20,20]，元素数虽相同，轴结构不同，初始与最终都没有列候选。两次 Transpose 的 perm 均 [0,1,3,2]，但 node178→179(MatMul)→181(Mul)→182(Softmax)→183，不能跨计算操作抵消；node183对 Softmax 结果做转置。')
add();add('## 7. Skill 自主深入诊断结果');add()
add('实际应用仓库 `.agents/skills/rdk-x5-onnx-doctor/SKILL.md`，按诊断事实选择节点，并未将所有算子套用相同建议。')
add();table(['触发证据','自主选取与核对','结果'],[
 ['最大已知载荷及 Conv fanout','tensor 查询首层三个输出，inspect/trace node0/1/2/3','Conv→Sigmoid→Mul，node0值被 node1/2共享；全部可达两个模型输出，不能按三个载荷直接相加推峰值'],
 ['第一个未知 Slice','node8/9/11/14/16/18/19/22 及内联 shape 标量','静态特征遇到 shape 算术 Slice 后产生 unk 符号；data_prop对照不消除'],
 ['Top10 中注意力矩阵','node176/177/178/179/181/182/183/184/187/188','[1,4,400,400] FP32 2,560,000 B；多次计算/transpose不能被误认逆 Transpose'],
 ['9 个信息不足观察','全部 candidate 查询，再查 producer/consumer 和 shape目标','已知轴或 rank 能排除 no-op；最小修复后 9→0'],
 ['高资源路径与候选关联','从 node2/8 到初始候选节点最短依赖路径','早期大特征可达注意力和四个任务分支；属于路径相关，未发现与确定违规或最终候选的关系'],
 ['Conv 状态全通过但多数资源未知','Conv 逐规则与 tensor shape交叉核对','BPU静态属性检查范围与资源shape完整性不同；不把未知资源解释为违规']])
add('路径证据见 [graph_evidence.json](graph_evidence.json) 的 selected_shortest_paths，每段包含真实 Tensor 标签。node2 为最大并列 Tensor 的生产者，直接上游 node0/1、下游 node3；trace 对 cls_logits 最短代表路径含 39 个节点，对 offset 含 41 个节点，其他分叉路径不枚举（paths_truncated=true）。')
add();add('注意力路径 node176→177(Split)→178(Transpose)→179(MatMul)→181(Mul)→182(Softmax)→183(Transpose)→184(MatMul)→187(Reshape)；此外 node177 输出 V 被 node184与node188共享。node179、181、182、183 的输出都为 [1,4,400,400] FP32，2,560,000 B = 2.44140625 MiB；假设 INT8 为 640,000 B。它们并非全都出现在限制十条的 Top10 表里。')
add();add('node222 的任务0分类路径是 node220(Gemm)→222(Reshape)→271(Unsqueeze)→275(Concat/cls_logits)；offset 支路 node224(Tanh)→226(Reshape)→228(Mul)→276(Unsqueeze)→280(Concat/offset)。因此 Reshape 负责终端 shape 组装，不能因 batch 符号未知就提出删除它。其余三任务分支有对应明确路径。')
add();add('实际发现：节点/连边/尺寸数学一致、shape推断限制、冗余观察筛选缺陷。局部理论推导：首个 Slice 的通道边界可由有界标量算出。未验证假设：编译器是否融合 SiLU/Conv、是否复用缓冲、attention 与其它算子是否落 BPU、真实 DDR/SRAM、延迟、量化误差。路径相关不能证明后者。')
add();add('## 8. Bug 根因、修复与回归');add()
add('Bug：Reshape 检测器先要求 input shape 所有维度静态；遇到符号轴即报告 INSUFFICIENT_INFORMATION，即使已知 rank或其它轴已证明改变 shape。这是候选筛选缺陷，不是已确认冗余的误报。真实模型 8 个 rank2→rank3 与 1 个已知512→4 暴露该问题。')
add();add('最小案例使用 tiny ONNX：输入 [batch,6] → Reshape 目标 [1,2,3]；输入 [batch,512,height,width] → [1,4,128,400]。修复前两例均产生无必要观察，回归断言均失败。对照 [batch,6] → [1,6] 仍不能证明no-op，必须保留信息不足。')
add();table(['修改文件','内容'],[
 ['src/rdkx5_doctor/optimization_candidates.py','12行：已审查标准语义+已知一维INT64正目标下，以rank/已知轴变化作为否定判据；不猜符号轴，不重写图'],
 ['tests/test_optimization_candidates.py','新增参数化回归3项，验证rank改变、已知轴改变、真正未知仍保留'],
 ['docs/ANALYSIS_SCHEMA_V1_1.md','说明符号shape下的已知反证筛选'],
 ['docs/DEVELOPMENT_LOG.md','记录真实模型验收、环境失败、修复与限制']])
add('原有 138 项测试全部保留，完整结果 141 passed。没有变更 Conv2D YAML、通道或其它官方BPU限制。修复前/后 analysis 对比除 optimization_candidates 外每个顶层段完全相同。源码差异见 [source_changes.diff](source_changes.diff)。')
add();add('## 9. V1.2 建议（仅建议，本次未开发新规则）');add()
table(['优先事项','真实证据','下一版应做什么'],[
 ['P0 有界 shape算术传播','node9→11→14→16→18→19 链使172个Tensor未知；data_prop无改善','仅为Shape/Gather及小整数Add/Div/Mul/Slice传播静态shape事实，限制元素数/编码/整数语义，保持动态与无法证明值未知；无需模型结构优化'],
 ['P0 逐算子shape完整性解释','原图没有value_info；42个Conv输出等未知','显示原始/推断元信息来源及不确定传播起点；Top10醒目标注仅已知集合，避免误解全图最大'],
 ['P1 elementwise规则覆盖','Mul51、Sigmoid38、Add14，大量SiLU/残差和广播链','核对对应工具链版本官方X5的dtype/广播/shape约束，再扩展逐规则证据；不凭结构猜CPU/BPU分配'],
 ['P1 Slice/Concat/Reshape/Transpose/Resize','Slice8、Concat11、Reshape11、Transpose2、Resize1','优先查官方轴/参数/版本/布局限制，特别Resize asymmetric/nearest/floor；保持支持条件UNKNOWN'],
 ['P1 MatMul/Softmax/Gemm','attention两个MatMul+Softmax，末端12个Gemm','核对官方rank、axis、transpose、dtype、常量参数条件；绑定node179/182/184和task分支'],
 ['P2 Pool/Flatten/Split等','MaxPool3、AveragePool4、Flatten4、Split2','补齐真实图中剩余算子规则，并分清shape解析成功与BPU支持证据'],
 ['P2 候选否定证据与解释','本次排除9个观察，两个等元素Reshape也不冗余','可单独提供排除理由，扩展0/-1部分shape否定判据时增加版本/allowzero边界回归，保持不自动改写']])
add('新硬件条款必须先按目标工具链版本核对官方X5 BPU列，再升级规则包及边界测试；不得将CPU列、其它芯片、Tensor原始B或推测当作新BPU限制。本次源路径含 OpenExplorer 1.2.8 字样，只说明文件所在目录，不说明已运行或已验证该工具链。')
add();add('## 10. 文件与验收');add()
add('- [analysis.json](analysis.json)：修复后的完整机器事实。')
add('- [report.md](report.md)：现有CLI生成的最终标准报告。')
add('- [before_fix/analysis.json](before_fix/analysis.json)：初始9个观察及候选ID的历史证据。')
add('- [independent_audit.json](independent_audit.json)：源ONNX逐节点/边界对照、尺寸oracle、规则计数与SHA256。')
add('- [graph_evidence.json](graph_evidence.json)：完整未知/多消费者列表、局部常量和跨节点Tensor依赖路径。')
add('- [command_log.jsonl](command_log.jsonl)：实际CLI参数、退出码、耗时；同名stdout/stderr文件保存内容。')
add('- `scripts/`：本次验证脚本快照，运行前设置验证目录；不写模型、不运行部署工具。')
add();add('验收：真实ONNX解析、Tensor资源分析、五种候选检查、重要节点查询、原有功能回归、完整报告和原始SHA256不变全部完成。未知资源、完整BPU覆盖、运行时数值/硬件/内存/性能明确无法在本次静态验收验证。`reports/` 按现有 .gitignore 被忽略，文件已保存本地；未执行git add/commit/push。')
text='\n'.join(lines)+'\n'
text=text.replace("`.venv/bin/python -m pip install -e \\' .[dev]\\'`（实际参数为 `-e \\\".[dev]\\\"`，无前导空格）", "`.venv/bin/python -m pip install -e '.[dev]'`")
(out/'REAL_ONNX_VALIDATION.md').write_text(text)
scripts=out/'scripts';scripts.mkdir(exist_ok=True)
for name in ('rdkx5_run_validation.py','rdkx5_deep_validation.py','rdkx5_graph_evidence.py','rdkx5_write_report.py'):
    shutil.copy2(Path('/tmp')/name,scripts/name)
print(out/'REAL_ONNX_VALIDATION.md')
