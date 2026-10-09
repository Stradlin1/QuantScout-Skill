"""Small declarative checker. YAML never executes code."""
from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, ConfigDict, model_validator, field_validator
from .conv_extractor import integer
from .operator_fields import ALL_FIELDS as FIELDS, FIELDS_BY_OPERATOR
from .rule_predicates import PREDICATES
from dataclasses import dataclass

class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

class Condition(Strict):
    field: str
    op: Literal['equals', 'greater_than']
    value: int | bool | str

    @model_validator(mode='after')
    def condition_type(self):
        if self.op == 'greater_than' and not integer(self.value):
            raise ValueError('greater_than 条件需要整数 value')
        return self

    @field_validator('field')
    @classmethod
    def known_field(cls, value):
        if value not in FIELDS:
            raise ValueError(f'未知派生字段：{value}')
        return value

class Rule(Strict):
    id: str
    check_type: Literal['range', 'max_value', 'equals', 'one_of', 'predicate']
    field: str
    min: int | None = None
    max: int | None = None
    value: int | bool | str | None = None
    values: list[int | str | bool] | None = None
    checkability: Literal['auto_check', 'conditional', 'review_only']
    when: Condition | None = None
    source_url: str
    source_section: str
    message: str
    operator: str | None = None
    predicate: str | None = None
    source_document: str | None = None
    source_version: str | None = None
    source_column: Literal['X5 BPU 支持约束'] | None = None
    reason_code: str | None = None
    blocking: bool = True

    @field_validator('field')
    @classmethod
    def known_field(cls, value):
        if value not in FIELDS:
            raise ValueError(f'未知派生字段：{value}')
        return value

    @field_validator('source_url')
    @classmethod
    def safe_url(cls, value):
        if not value.startswith('https://'):
            raise ValueError('来源必须为 https URL')
        return value

    @model_validator(mode='after')
    def bounds(self):
        if self.check_type == 'range' and (self.min is None or self.max is None or self.min > self.max):
            raise ValueError('range 需要合法 min/max')
        if self.check_type == 'max_value' and self.max is None:
            raise ValueError('max_value 需要 max')
        if self.check_type == 'equals' and self.value is None:
            raise ValueError('equals 需要 value')
        if self.check_type == 'one_of' and not self.values:
            raise ValueError('one_of 需要非空 values')
        allowed = {'range': {'min','max'}, 'max_value': {'max'}, 'equals': {'value'}, 'one_of': {'values'}, 'predicate': set()}[self.check_type]
        if any(getattr(self, x) is not None for x in {'min','max','value','values'} - allowed):
            raise ValueError('检查操作包含不适用的参数')
        if not self.id or not self.source_section:
            raise ValueError('id 和 source_section 不能为空')
        if self.check_type == 'predicate':
            if self.predicate not in PREDICATES:
                raise ValueError('未知谓词')
        elif self.predicate is not None:
            raise ValueError('非 predicate 规则不能设置谓词')
        return self

class RuleSet(Strict):
    ruleset_id: str
    ruleset_version: str
    model_domain: str
    operator: Literal['Conv', 'Sigmoid', 'Concat', 'Slice', 'Add', 'Mul', 'Gemm', 'MatMul', 'Softmax', 'Resize', 'Reshape', 'Split', 'MaxPool', 'AveragePool']
    scope: Literal['input_rank_4', 'standard_onnx']
    source_document: str
    source_version: str
    source_url: str
    toolchain_version: str
    rules: list[Rule]
    opset_min: int | None = None
    opset_max: int | None = None
    capabilities: list[str] = []

    @model_validator(mode='after')
    def unique_ids(self):
        if not self.rules or len({r.id for r in self.rules}) != len(self.rules):
            raise ValueError('规则不能为空或包含重复 ID')
        if self.model_domain != '':
            raise ValueError('V1 只接受标准 ONNX 域规则')
        if self.operator == 'Conv' and self.scope != 'input_rank_4':
            raise ValueError('Conv scope 必须保留 input_rank_4')
        if self.operator != 'Conv':
            if self.scope != 'standard_onnx' or not self.opset_min or not self.opset_max or not 1 <= self.opset_min <= self.opset_max <= 23:
                raise ValueError('新增算子需要已审查的 scope/opset 区间')
        for rule in self.rules:
            if rule.predicate is not None:
                predicate_fields = {'x5_elementwise_broadcast_mergeable': 'broadcast_mergeable',
                                    'at_most_one_fixed_constant_input': 'constant_count_ok',
                                    'matmul_broadcast_pattern_supported':'matmul_broadcast_pattern_supported'}
                if self.operator not in (('MatMul',) if rule.predicate=='matmul_broadcast_pattern_supported' else ('Add','Mul')) or rule.field != predicate_fields[rule.predicate]:
                    raise ValueError('谓词不属于对应算子/字段')
            if rule.field not in FIELDS_BY_OPERATOR[self.operator] or (rule.when and rule.when.field not in FIELDS_BY_OPERATOR[self.operator]):
                raise ValueError('规则字段不属于对应算子')
            if self.operator != 'Conv' and (rule.operator != self.operator or not all((rule.source_document, rule.source_version, rule.source_column, rule.reason_code))):
                raise ValueError('新增规则缺少 operator/来源版本/章节或原因码')
            if self.operator != 'Conv' and (rule.source_url, rule.source_document, rule.source_version) != (self.source_url, self.source_document, self.source_version):
                raise ValueError('规则与算子文件来源不一致')
        return self

class Source(Strict):
    title: str
    url: str
    version: str
    section: str

    @field_validator('url')
    @classmethod
    def safe_url(cls, value):
        if not value.startswith('https://'):
            raise ValueError('来源必须为 https URL')
        return value

class Manifest(Strict):
    platform: Literal['RDK X5']
    march: Literal['bayes-e']
    ruleset_id: str
    ruleset_version: str
    retrieved_at: str
    toolchain_version: str
    review_status: Literal['reviewed_with_exclusions', 'needs_review']
    sources: list[Source]
    exclusions: list[str]
    operators: dict[str, 'Registration'] | None = None
    toolchain_verified: Literal[False] = False
    source_policy: Literal['X5_ONNX_BPU_COLUMN_ONLY'] = 'X5_ONNX_BPU_COLUMN_ONLY'

class Registration(Strict):
    file: str
    ruleset_id: str
    version: str
    source_version: str
    opset_min: int | None = None
    opset_max: int | None = None

Manifest.model_rebuild()

@dataclass
class RuleRegistry:
    manifest: Manifest
    by_operator: dict[str, RuleSet]

    @property
    def ruleset_id(self): return self.manifest.ruleset_id
    @property
    def ruleset_version(self): return self.manifest.ruleset_version
    @property
    def toolchain_version(self): return self.manifest.toolchain_version
    @property
    def rules(self):
        """Legacy Conv callers continue using the original 19-rule list."""
        return self.by_operator['Conv'].rules
    @property
    def all_rules(self): return [r for op in self.by_operator.values() for r in op.rules]
    @property
    def rules_by_id(self): return {r.id: r for r in self.all_rules}

    def get_rules(self, operator, domain, opset):
        op = self.by_operator.get(operator)
        if not op or domain not in ('', 'ai.onnx'):
            return None
        if op.operator != 'Conv' and (type(opset) is not int or not op.opset_min <= opset <= op.opset_max):
            return None
        return op

def safe_rule_path(directory, name):
    rel = Path(name)
    if rel.is_absolute() or '..' in rel.parts or rel.suffix != '.yaml':
        raise ValueError('危险规则文件路径')
    resolved = (directory / rel).resolve()
    if not resolved.is_relative_to(directory.resolve()):
        raise ValueError('规则文件软链越界')
    return resolved

def read_yaml(path):
    text = path.read_text(encoding='utf-8')
    data = yaml.safe_load(text)
    # Safe YAML still silently overwrites duplicate mapping keys. Reject them,
    # including duplicated operator registrations; inspect nodes without executing.
    root = yaml.compose(text, Loader=yaml.SafeLoader)
    seen = set()
    def visit(node):
        if node is None or id(node) in seen:return
        seen.add(id(node))
        if isinstance(node,yaml.MappingNode):
            keys=set()
            for key,value in node.value:
                if not isinstance(key,yaml.ScalarNode) or key.value in keys:
                    raise ValueError('YAML 重复或非标量键')
                keys.add(key.value);visit(value)
        elif isinstance(node,yaml.SequenceNode):
            for value in node.value:visit(value)
    visit(root)
    return data

def load_ruleset(directory):
    directory = Path(directory)
    try:
        manifest = Manifest.model_validate(read_yaml(safe_rule_path(directory,'manifest.yaml')))
        if manifest.operators is None:
            rules = RuleSet.model_validate(read_yaml(safe_rule_path(directory, 'conv2d.yaml')))
            if (manifest.ruleset_id, manifest.ruleset_version, manifest.toolchain_version) != (rules.ruleset_id, rules.ruleset_version, rules.toolchain_version):
                raise ValueError('manifest 与 conv2d 的 ID/版本/工具链标记不一致')
            by_operator = {'Conv': rules}
        else:
            if not manifest.operators or any(op not in FIELDS_BY_OPERATOR for op in manifest.operators):
                raise ValueError('未知或空 operator 注册')
            by_operator, paths, ids = {}, set(), set()
            for operator, registration in manifest.operators.items():
                path = safe_rule_path(directory, registration.file)
                if path in paths:
                    raise ValueError('重复 operator 文件')
                paths.add(path)
                op = RuleSet.model_validate(read_yaml(path))
                if (operator, registration.ruleset_id, registration.version, registration.source_version, registration.opset_min, registration.opset_max) != (op.operator, op.ruleset_id, op.ruleset_version, op.source_version, op.opset_min, op.opset_max):
                    raise ValueError('manifest 与算子文件版本/域/opset 不一致')
                if manifest.toolchain_version != op.toolchain_version:
                    raise ValueError('工具链标记不一致')
                if (op.source_url, op.source_version) not in {(s.url, s.version) for s in manifest.sources}:
                    raise ValueError('来源未登记在 manifest')
                for rule in op.rules:
                    if rule.id in ids:
                        raise ValueError('跨算子重复规则 ID')
                    ids.add(rule.id)
                by_operator[operator] = op
            if 'Conv' not in by_operator:
                raise ValueError('必须保留 Conv 注册')
            if manifest.ruleset_version not in ('0.2.0','0.3.0','0.4.0'):
                raise ValueError('多算子 Registry 总版本不匹配')
    except Exception as exc:
        raise ValueError(f'规则集校验失败（{directory}）：{exc}') from exc
    return RuleRegistry(manifest, by_operator), manifest

def compare(actual, op, value):
    if op == 'equals':
        return type(actual) is type(value) and actual == value
    return actual > value

def check_rule(rule, fields, scope='conv2d'):
    expected = ({'min': rule.min, 'max': rule.max} if rule.check_type == 'range' else
                {'max': rule.max} if rule.check_type == 'max_value' else
                {'equals': rule.value} if rule.check_type == 'equals' else
                {'predicate': rule.predicate, 'equals': True} if rule.check_type == 'predicate' else {'one_of': rule.values})
    result = {'rule_id': rule.id, 'field': rule.field, 'actual': fields.get(rule.field),
        'expected': expected, 'source_url': rule.source_url, 'source_section': rule.source_section,
        'checkability': rule.checkability, 'status': 'UNKNOWN', 'reason': ''}
    def done(status, reason):
        result.update(status=status, reason=reason)
        return result
    if scope == 'not_applicable':
        return done('NOT_APPLICABLE', '不属于 V1 Conv2D 范围')
    if scope not in ('conv2d', 'standard_onnx'):
        return done('UNKNOWN', '无法确认标准域 rank-4 Conv2D 范围')
    if rule.when:
        condition = fields.get(rule.when.field)
        if condition is None or (rule.when.op == 'greater_than' and not integer(condition)):
            return done('UNKNOWN', f'触发条件字段未知：{rule.when.field}')
        if not compare(condition, rule.when.op, rule.when.value):
            return done('NOT_APPLICABLE', '触发条件不成立')
    if rule.checkability != 'auto_check':
        return done('UNKNOWN', rule.message + '；需工具链或人工确认')
    actual = PREDICATES[rule.predicate](fields) if rule.check_type == 'predicate' else result['actual']
    result['actual'] = actual
    if actual is None or (rule.check_type in ('range', 'max_value') and not integer(actual)):
        return done('UNKNOWN', f'静态字段缺失或符号化：{rule.field}')
    if rule.check_type == 'range':
        passed = rule.min <= actual <= rule.max
    elif rule.check_type == 'max_value':
        passed = actual <= rule.max
    elif rule.check_type == 'predicate':
        passed = actual is True
    elif rule.check_type == 'equals':
        passed = compare(actual, 'equals', rule.value)
    else:
        passed = any(compare(actual, 'equals', v) for v in rule.values)
    return done('PASS' if passed else 'FAIL', '已检查规则未发现违规' if passed else rule.message)

def check_node(node, extracted, ruleset):
    if node['op_type'] != 'Conv':
        if extracted and extracted.get('scope') == 'standard_onnx':
            op = ruleset.by_operator[node['op_type']] if isinstance(ruleset, RuleRegistry) else ruleset
            results = [check_rule(rule, extracted['fields'], 'standard_onnx') for rule in op.rules]
            status = multiop_status(results, op.rules, extracted['issues'])
            suggestions = []
            if status in ('VIOLATION', 'NEEDS_VERIFICATION'):
                relevant = [r for r in results if r['status'] in ('FAIL','UNKNOWN')]
                suggestions = [{'node_id':node['id'], 'rule_id':r['rule_id'],
                    'category':'训练/导出工程复查' if r['status']=='FAIL' else '待验证',
                    'evidence':{'field':r['field'],'actual':r['actual'],'expected':r['expected']},
                    'text':'回到原始模型定义/forward 或导出脚本核对该使用方式；涉及计算结构改变需评估重训或微调，不在本项目修改 ONNX。' if r['status']=='FAIL' else '补齐该节点元信息或由未来工具链核实转换条件；没有静态证据要求立即改变网络。',
                    'validation_needed':'训练/导出工程重新导出 → ONNX checker → analyze → 输入输出接口及数值/任务指标验证；本次不执行工具链'} for r in relevant]
            return {'node_id':node['id'],'status':status,'scope':'standard_onnx','results':results,
                    'issues':extracted['issues'],'suggestions':suggestions}
        return {'node_id': node['id'], 'status': 'NOT_COVERED', 'results': [], 'issues': [], 'suggestions': []}
    results = [check_rule(rule, extracted['fields'], extracted['scope']) for rule in ruleset.rules]
    failed = [r for r in results if r['status'] == 'FAIL']
    if failed:
        status = 'VIOLATION'
    elif extracted['scope'] == 'not_applicable':
        status = 'NOT_COVERED'
    elif extracted['issues'] or any(r['status'] == 'UNKNOWN' for r in results):
        status = 'NEEDS_VERIFICATION'
    else:
        status = 'NO_VIOLATION_FOUND'
    suggestions = []
    for result in failed:
        field = result['field']
        if field in ('kernel_h', 'kernel_w'):
            candidate = '评估将大卷积核改成多层较小卷积核；此方案通常改变计算与感受野组合，需要重新训练及任务指标对比。'
        elif field == 'kernel_elements_per_group':
            candidate = '评估减少每组输入通道、引入瓶颈层或调整分组；分组改变通道交互，不可声称数值等价。'
        elif 'divisible' in field:
            candidate = '先核查导出输入尺寸与预处理；评估可被 dilation 整除的输入尺寸，验证坐标、边界与输出语义。'
        elif field.startswith('dilation_'):
            candidate = '评估减小膨胀率或用多层卷积恢复目标感受野；核验输入整除及量化输出条件。'
        elif field.startswith('strides_'):
            candidate = '若存在 dilation，先评估 stride=1 的可行性；基础步长超限则评估分阶段下采样，验证输出尺寸与任务指标。'
        else:
            candidate = '核查显式 pads 与 auto_pad 的有效值；调整边界填充前先验证输出尺寸、边界语义和实际工具链支持。'
        suggestions.append({'node_id': node['id'], 'rule_id': result['rule_id'],
            'category': '可能需要改变网络并重训',
            'evidence': {'field': result['field'], 'actual': result['actual'], 'expected': result['expected']},
            'text': f"检查该节点导出属性 {result['field']} 与训练结构是否一致；如为实际结构约束，评估替代卷积结构并重新训练或微调。{candidate} 不得直接删除节点。",
            'validation_needed': '候选模型数值等价/任务指标验证，以及实际工具链 checker；无性能或精度承诺'})
    if not failed:
        suggestions.append({'node_id': node['id'], 'rule_id': None, 'category': '无需修改' if status == 'NO_VIOLATION_FOUND' else '待验证',
            'evidence': {'status': status}, 'text': '目前没有静态违规证据支持修改该节点；先补足未知信息并验证工具链。',
            'validation_needed': '实际工具链 checker'})
    return {'node_id': node['id'], 'status': status, 'scope': extracted['scope'],
            'results': results, 'issues': extracted['issues'], 'suggestions': suggestions}

def multiop_status(results, rules, issues=()):
    if any(r['status']=='FAIL' for r in results):return 'VIOLATION'
    blocking={r.id for r in rules if r.blocking}
    if issues or any(r['status']=='UNKNOWN' and r['rule_id'] in blocking for r in results):return 'NEEDS_VERIFICATION'
    if any(r['status']=='PASS' and r['checkability']=='auto_check' for r in results):return 'NO_VIOLATION_FOUND'
    if any(r['status']=='UNKNOWN' for r in results):return 'NEEDS_VERIFICATION'
    return 'NOT_COVERED'

def check_operator(ir, node, registry, node_index=None):
    from .operator_facts import extract_operator, imported_opset
    from .conv_extractor import extract_conv
    op = registry.by_operator.get(node['op_type'])
    if node['op_type']=='Conv' and op:
        extracted=extract_conv(node,ir.tensors)
        node['conv']=extracted
        diagnostic=check_node(node,extracted,op)
    else:
        matched=registry.get_rules(node['op_type'],node['domain'],imported_opset(ir))
        extracted=extract_operator(ir,node,node_index) if matched else None
        if extracted and extracted['scope']=='standard_onnx':
            node['operator_facts']=extracted
            diagnostic=check_node(node,extracted,registry)
        else:
            diagnostic={'node_id':node['id'],'status':'NOT_COVERED','results':[],
                        'issues':extracted['issues'] if extracted else [],'suggestions':[]}
    diagnostic['operator']=node['op_type']
    diagnostic['coverage_type']=('NOT_COVERED' if diagnostic['status']=='NOT_COVERED' else
                                 'AUTO_CHECKED' if diagnostic['status']=='NO_VIOLATION_FOUND' else 'PARTIAL_OR_CONDITIONAL')
    if op:
        rule_index={r.id:r for r in op.rules}
        for result in diagnostic['results']:
            rule=rule_index[result['rule_id']]
            source_version=rule.source_version or op.source_version
            result.update(node_id=node['id'],operator=node['op_type'],
                          observed={'field':result['field'],'value':result['actual']},
                          reason_code=rule.reason_code or ('BPU_CONSTRAINT_FAIL' if result['status']=='FAIL' else 'CONV_STATIC_'+result['status']),
                          blocking=rule.blocking,
                          source={'document':rule.source_document or op.source_document,'version':source_version,
                                  'section':rule.source_section,'url':rule.source_url,'column':'X5 BPU 支持约束',
                                  'retrieved_at':registry.manifest.retrieved_at},
                          evidence=extracted.get('field_evidence',{}).get(result['field'],
                              {'inputs':[{k:ir.tensors[name].get(k) for k in ('name','shape','dtype','producer','consumers')} for name in node['inputs'] if name],
                               'attributes':node['attributes'],'derived_field':result['field']}),
                          unverified=['toolchain_version=unverified; 未测量 CPU/BPU 分配或量化精度'])
        for suggestion in diagnostic['suggestions']:
            if diagnostic['status']=='VIOLATION':
                suggestion['text']='仅提供架构/导出方向；若决定改变模型，由所有者在独立原工程实施并重新导出对比；本项目不查找源码或生成补丁。'+suggestion['text']
    return diagnostic
