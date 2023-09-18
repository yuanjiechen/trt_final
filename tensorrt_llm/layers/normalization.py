import tensorrt as trt

from ..functional import group_norm, layer_norm, rms_norm, rms_normquant_reorder, index_select, div, unsqueeze, mul
from ..module import Module
from ..parameter import Parameter


class LayerNorm(Module):

    def __init__(self,
                 normalized_shape,
                 eps=1e-05,
                 elementwise_affine=True,
                 dtype=None):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape, )
        self.normalized_shape = tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(shape=self.normalized_shape, dtype=dtype)
            self.bias = Parameter(shape=self.normalized_shape, dtype=dtype)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.eps = eps

    def forward(self, x):
        weight = None if self.weight is None else self.weight.value
        bias = None if self.bias is None else self.bias.value
        return layer_norm(x, self.normalized_shape, weight, bias, self.eps)


class RmsNorm_reindex(Module):

    def __init__(self,
                 normalized_shape,
                 eps=1e-06,
                 elementwise_affine=True,
                 dtype=None,
                 quant_wa=False):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape, )
        self.normalized_shape = tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine
        self.quant_wa = quant_wa
        if self.elementwise_affine:
            self.weight = Parameter(shape=self.normalized_shape, dtype=dtype)
        else:
            self.register_parameter('weight', None)

        self.eps = eps

        # if quant_wa:
        self.index_input = Parameter(shape=normalized_shape, dtype=trt.int32)
        self.scale = Parameter(shape=normalized_shape, dtype=trt.float32)

    def forward(self, x):
        weight = None if self.weight is None else self.weight.value
        if self.quant_wa: 
            return rms_normquant_reorder(x, self.normalized_shape, weight, dst_index=self.index_input.value, scale=self.scale.value)
        else: 
            out = rms_norm(x, self.normalized_shape, weight, self.eps)
            out = index_select(out, 2, self.index_input.value)
            out =  div(out, unsqueeze(unsqueeze(self.scale.value, 0), 0))
            out =  mul(out, unsqueeze(unsqueeze(self.scale.value, 0), 0))
            # out = out.cast(trt.int8)
            return out

class RmsNorm(Module):

    def __init__(self,
                 normalized_shape,
                 eps=1e-06,
                 elementwise_affine=True,
                 dtype=None):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape, )
        self.normalized_shape = tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(shape=self.normalized_shape, dtype=dtype)
        else:
            self.register_parameter('weight', None)

        self.eps = eps

    def forward(self, x):
        weight = None if self.weight is None else self.weight.value
        return rms_norm(x, self.normalized_shape, weight, self.eps)


class GroupNorm(Module):

    def __init__(self,
                 num_groups,
                 num_channels,
                 eps=1e-05,
                 affine=True,
                 dtype=None):
        super().__init__()

        if num_channels % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.affine = affine

        if self.affine:
            self.weight = Parameter(shape=(self.num_channels, ), dtype=dtype)
            self.bias = Parameter(shape=(self.num_channels, ), dtype=dtype)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.eps = eps

    def forward(self, x):
        weight = None if self.weight is None else self.weight.value
        bias = None if self.bias is None else self.bias.value
        return group_norm(x, self.num_groups, weight, bias, self.eps)
