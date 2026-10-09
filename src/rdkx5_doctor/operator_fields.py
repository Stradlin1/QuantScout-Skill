"""Per-operator field allowlists used by strict rule configuration validation."""
from .conv_extractor import FIELDS as CONV_FIELDS

FIELDS_BY_OPERATOR = {
    'Conv': CONV_FIELDS,
    'Sigmoid': {'input_rank', 'output_rank'},
    'Concat': {'axis_not_batch'},
    'Slice': {'slice_parameters_fixed'},
    'Add': {'min_input_rank', 'max_input_rank', 'output_rank', 'constant_count_ok', 'broadcast_mergeable', 'shortcut_review'},
    'Mul': {'min_input_rank', 'max_input_rank', 'output_rank', 'constant_count_ok', 'broadcast_mergeable'},
    'MatMul': {'matmul_rank_relation_ok','matmul_matrix_dims_within_limits','matmul_higher_dims_within_limits','matmul_broadcast_pattern_supported'},
    'Softmax': {'softmax_input_rank','softmax_axis','run_on_bpu_verified'},
    'Resize': {'resize_input_rank','resize_layout_nchw','resize_spatial_only','resize_mode','resize_coordinate_mode','resize_nearest_enlargement_ok','resize_roi_ok'},
    'Gemm': {'conversion_layout_known'},
    'Reshape': {'input_rank','output_rank','reshape_conversion_verified'},
    'Split': {'split_axis_not_batch','split_lengths_divide_input','split_count_divides_input'},
    'MaxPool': {'pool_kernel_max','pool_stride_max','pool_padding_max','pool_no_dilation'},
    'AveragePool': {'pool_kernel_h','pool_kernel_w','pool_kernel_area','pool_stride_h','pool_stride_w','pool_padding_min','pool_padding_max'},
}
ALL_FIELDS = set().union(*FIELDS_BY_OPERATOR.values())
