"""Validate Agent-written sidecar provenance, never fetch or execute sources.

This validates the evidence contract, not the truth of extracted prose. The
Agent must inspect the actual official table and retain the independent report.
"""
from datetime import datetime, date
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

URLS={
 'https://developer.d-robotics.cc/rdk_x_doc/Advanced_development/toolchain_development/intermediate/supported_op_list',
 'https://developer.d-robotics.cc/x5_sdk_doc/toolchain_development/intermediate/supported_op_list.html',
 'https://developer.d-robotics.cc/x5_sdk_doc_v2.0.0/en/toolchain_development/intermediate/supported_op_list.html',
}
class Strict(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
class QueryKey(Strict):
    domain: str
    operator: str=Field(min_length=1)
    imported_opset: int | None
class KnowledgeSource(Strict):
    title: str=Field(min_length=1)
    url: str
    section: str=Field(min_length=1)
    document_version: str=Field(min_length=1)
    checked_at: str
    original_reviewed_on: str | None = None
    source_column: Literal['X5 BPU 支持约束','CPU 支持约束','使用限制说明']
    chip: Literal['RDK X5']
    framework: Literal['ONNX']
    @field_validator('url')
    @classmethod
    def allowlisted(cls,value):
        if value.rstrip('/') not in URLS:raise ValueError('Knowledge source must be an allowlisted official X5 URL')
        return value
    @field_validator('checked_at')
    @classmethod
    def dated(cls,value):
        if datetime.fromisoformat(value).utcoffset() is None:raise ValueError('checked_at needs actual timestamp and timezone')
        return value
    @field_validator('original_reviewed_on')
    @classmethod
    def reviewed(cls,value):
        if value is not None:date.fromisoformat(value)
        return value
class KnowledgeRecord(Strict):
    query_key: QueryKey
    lookup_status: Literal['FETCHED','CACHED','FAILED','NOT_REQUESTED']
    knowledge_status: Literal['X5_BPU_DOCUMENTED_WITH_CONSTRAINTS','X5_BPU_DOCUMENTED_NO_EXTRA_CONSTRAINTS',
        'X5_CPU_DOCUMENTED','FOLDED_OR_LOWERED_CONDITIONALLY','NOT_FOUND_IN_REVIEWED_SOURCE',
        'SOURCE_UNAVAILABLE','VERSION_CONFLICT_OR_AMBIGUITY']
    source: KnowledgeSource
    cross_sources: list[KnowledgeSource]=[]
    extracted_conditions: list[str]
    machine_rule_present: bool
    runtime_placement_verified: Literal[False]
    notes: list[str]
    @model_validator(mode='after')
    def evidence_layer(self):
        if self.lookup_status=='CACHED' and self.source.original_reviewed_on is None:
            raise ValueError('Cached evidence needs original review date, distinct from cache-read checked_at')
        if self.lookup_status in ('FAILED','NOT_REQUESTED'):
            if self.knowledge_status!='SOURCE_UNAVAILABLE' or self.extracted_conditions or not self.notes:
                raise ValueError('Unavailable/unrequested source cannot substantiate document conditions')
        elif self.knowledge_status=='SOURCE_UNAVAILABLE':raise ValueError('Fetched/cached source unavailable is contradictory')
        if self.knowledge_status.startswith('X5_BPU_DOCUMENTED') and self.source.source_column!='X5 BPU 支持约束':
            raise ValueError('CPU/general prose is not a BPU constraint')
        if self.knowledge_status=='X5_CPU_DOCUMENTED' and self.source.source_column!='CPU 支持约束':
            raise ValueError('CPU statement needs CPU column')
        if self.knowledge_status=='VERSION_CONFLICT_OR_AMBIGUITY' and (not self.cross_sources or not self.notes):
            raise ValueError('Conflict needs independent sources and explanation')
        return self
class KnowledgeLookup(Strict):
    schema_version: Literal['1.0']
    records: list[KnowledgeRecord]
    @model_validator(mode='after')
    def unique_evidence(self):
        seen=set()
        for r in self.records:
            key=(r.query_key.domain,r.query_key.operator,r.query_key.imported_opset,r.source.url,r.source.section)
            if key in seen:raise ValueError('Duplicate operator/domain/opset/source evidence; deduplicate queries')
            seen.add(key)
        return self

def uncovered_query_keys(analysis):
    """Group real uncovered nodes; Agent chooses priorities/limit and networking."""
    states={d['node_id']:d['status'] for d in analysis['diagnostics']};groups={}
    imports=analysis.get('model',{}).get('opset_imports',[])
    for n in analysis['nodes']:
        if states.get(n['id'])!='NOT_COVERED':continue
        versions=[x.get('version') for x in imports if x.get('domain')==n['domain'] or x.get('domain') in ('','ai.onnx') and n['domain'] in ('','ai.onnx')]
        version=versions[0] if len(versions)==1 and type(versions[0])is int else None
        key=(n['op_type'],n['domain'],version);groups.setdefault(key,[]).append(n['id'])
    return [dict(operator=k[0],domain=k[1],imported_opset=k[2],node_count=len(v),node_ids=v) for k,v in groups.items()]
