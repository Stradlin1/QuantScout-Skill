"""Small declarative checker. YAML never executes code."""
from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, ConfigDict, model_validator, field_validator
from .conv_extractor import FIELDS, integer

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
    check_type: Literal['range', 'max_value', 'equals', 'one_of']
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
        allowed = {'range': {'min','max'}, 'max_value': {'max'}, 'equals': {'value'}, 'one_of': {'values'}}[self.check_type]
        if any(getattr(self, x) is not None for x in {'min','max','value','values'} - allowed):
            raise ValueError('检查操作包含不适用的参数')
        if not self.id or not self.source_section:
            raise ValueError('id 和 source_section 不能为空')
        return self

class RuleSet(Strict):
    ruleset_id: str
    ruleset_version: str
    model_domain: str
    operator: Literal['Conv']
    scope: Literal['input_rank_4']
    source_document: str
    source_version: str
    source_url: str
    toolchain_version: str
    rules: list[Rule]

    @model_validator(mode='after')
    def unique_ids(self):
        if not self.rules or len({r.id for r in self.rules}) != len(self.rules):
            raise ValueError('规则不能为空或包含重复 ID')
        if self.model_domain != '':
            raise ValueError('V1 只接受标准 ONNX 域规则')
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

def load_ruleset(directory):
    directory = Path(directory)
    try:
        manifest = Manifest.model_validate(yaml.safe_load((directory / 'manifest.yaml').read_text(encoding='utf-8')))
        rules = RuleSet.model_validate(yaml.safe_load((directory / 'conv2d.yaml').read_text(encoding='utf-8')))
    except Exception as exc:
        raise ValueError(f'规则集校验失败（{directory}）：{exc}') from exc
    if (manifest.ruleset_id, manifest.ruleset_version, manifest.toolchain_version) != (rules.ruleset_id, rules.ruleset_version, rules.toolchain_version):
        raise ValueError('manifest 与 conv2d 的 ID/版本/工具链标记不一致')
    return rules, manifest

def compare(actual, op, value):
    if op == 'equals':
        return type(actual) is type(value) and actual == value
    return actual > value

def check_rule(rule, fields, scope='conv2d'):
    expected = ({'min': rule.min, 'max': rule.max} if rule.check_type == 'range' else
                {'max': rule.max} if rule.check_type == 'max_value' else
                {'equals': rule.value} if rule.check_type == 'equals' else {'one_of': rule.values})
    result = {'rule_id': rule.id, 'field': rule.field, 'actual': fields.get(rule.field),
        'expected': expected, 'source_url': rule.source_url, 'source_section': rule.source_section,
        'checkability': rule.checkability, 'status': 'UNKNOWN', 'reason': ''}
    def done(status, reason):
        result.update(status=status, reason=reason)
        return result
    if scope == 'not_applicable':
        return done('NOT_APPLICABLE', '不属于 V1 Conv2D 范围')
    if scope != 'conv2d':
        return done('UNKNOWN', '无法确认标准域 rank-4 Conv2D 范围')
    if rule.when:
        condition = fields.get(rule.when.field)
        if condition is None or (rule.when.op == 'greater_than' and not integer(condition)):
            return done('UNKNOWN', f'触发条件字段未知：{rule.when.field}')
        if not compare(condition, rule.when.op, rule.when.value):
            return done('NOT_APPLICABLE', '触发条件不成立')
    if rule.checkability != 'auto_check':
        return done('UNKNOWN', rule.message + '；需工具链或人工确认')
    actual = result['actual']
    if actual is None or (rule.check_type in ('range', 'max_value') and not integer(actual)):
        return done('UNKNOWN', f'静态字段缺失或符号化：{rule.field}')
    if rule.check_type == 'range':
        passed = rule.min <= actual <= rule.max
    elif rule.check_type == 'max_value':
        passed = actual <= rule.max
    elif rule.check_type == 'equals':
        passed = compare(actual, 'equals', rule.value)
    else:
        passed = any(compare(actual, 'equals', v) for v in rule.values)
    return done('PASS' if passed else 'FAIL', '已检查规则未发现违规' if passed else rule.message)

def check_node(node, extracted, ruleset):
    if node['op_type'] != 'Conv':
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
