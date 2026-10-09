"""Per-operator field allowlists used by strict rule configuration validation."""
from .conv_extractor import FIELDS as CONV_FIELDS

FIELDS_BY_OPERATOR = {
    'Conv': CONV_FIELDS,
    'Sigmoid': {'input_rank', 'output_rank'},
    'Concat': {'axis_not_batch'},
    'Slice': {'slice_parameters_fixed'},
    'Add': {'min_input_rank', 'max_input_rank', 'output_rank', 'constant_count_ok', 'broadcast_mergeable', 'shortcut_review'},
    'Mul': {'min_input_rank', 'max_input_rank', 'output_rank', 'constant_count_ok', 'broadcast_mergeable'},
    'Gemm': {'conversion_layout_known'},
}
ALL_FIELDS = set().union(*FIELDS_BY_OPERATOR.values())
