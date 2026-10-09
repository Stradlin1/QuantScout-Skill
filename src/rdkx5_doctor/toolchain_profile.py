"""Current-user configuration gate, independent of ONNX/BPU validity."""
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from .rules import read_yaml

class ToolchainProfile(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    profile_id: str = Field(min_length=1)
    chip: Literal['RDK X5']
    march: Literal['bayes-e']
    onnx_default_domain_required_opset: int = Field(ge=1)
    opset_policy: Literal['exact_match_for_current_user_toolchain']
    profile_origin: Literal['user_provided_project_constraint']
    toolchain_version: Literal['unverified']
    compiler_checked: Literal[False]
    onnx_generic_analysis_on_mismatch: Literal[True]

def load_profile(path):
    path = Path(path)
    if path.stat().st_size > 65536:
        raise ValueError('profile 文件超出大小限制')
    return ToolchainProfile.model_validate(read_yaml(path))

def preflight(model, profile):
    model = model if isinstance(model, dict) else {}
    imports = model.get('opset_imports')
    entries = imports if isinstance(imports, list) else []
    standard = [x for x in entries if isinstance(x, dict) and x.get('domain') in ('', 'ai.onnx')]
    valid = (len(standard) == 1 and type(standard[0].get('version')) is int
             and standard[0]['version'] > 0)
    actual = standard[0]['version'] if valid else None
    status = ('MATCH' if actual == profile.onnx_default_domain_required_opset else 'MISMATCH') if valid else 'UNKNOWN'
    custom = [x for x in entries if isinstance(x, dict) and isinstance(x.get('domain'), str)
              and x['domain'] not in ('', 'ai.onnx')]
    return dict(schema_version='1.0', profile_id=profile.profile_id,
                profile_origin=profile.profile_origin, required_standard_opset=profile.onnx_default_domain_required_opset,
                actual_standard_opset=actual, status=status, checker_status=model.get('validation', 'unknown'),
                custom_domains=custom, custom_domains_present=bool(custom), requires_separate_review=bool(custom),
                toolchain_version=profile.toolchain_version, compiler_checked=False,
                reason='exact current-user profile comparison' if valid else 'standard ONNX opset missing, ambiguous or invalid; rerun analyze if metadata is absent',
                effects=[{'MATCH':'version_condition_only', 'MISMATCH':'current_target_profile_not_matched', 'UNKNOWN':'current_target_profile_unverified'}[status],
                         'generic_onnx_analysis_still_available'])
