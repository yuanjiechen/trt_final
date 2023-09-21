import numpy as np
import tensorrt as trt

from .._common import default_net, default_trtnet
from .._utils import int32_array, str_dtype_to_trt
from ..functional import (Tensor, _create_tensor, allgather, allreduce, concat,
                          constant, matmul, shape, slice, div, mul, unsqueeze, constant, mean, round, clip)
from ..module import Module
from ..parameter import Parameter
from ..plugin import _TRT_LLM_PLUGIN_NAMESPACE as TRT_LLM_PLUGIN_NAMESPACE

def _w8a8gemm(input: Tensor, weights: Tensor, scales_a: Tensor,
                      scales_b: Tensor) -> Tensor:
    # fake_scale_A = constant(np.ones([1]).astype(np.float32))
    plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'W8A8Gemm', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plg_creator is not None

    per_channel_scaling = trt.PluginField(
        "has_per_channel_scaling",
        np.array(0, dtype=np.int32),
        trt.PluginFieldType.INT32)

    per_token_scaling = trt.PluginField(
        "has_per_token_scaling", 
        np.array(0, dtype=np.int32),
        trt.PluginFieldType.INT32)

    pf_type = trt.PluginField(
        "type_id", np.array([int(trt.float32)], np.int32),
        trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection(
        [per_channel_scaling, per_token_scaling, pf_type])
    gemm_plug = plg_creator.create_plugin("W8A8Gemm", pfc)
    plug_inputs = [
        input.trt_tensor, weights.trt_tensor, scales_a.trt_tensor,
        scales_b.trt_tensor
    ]
    layer = default_trtnet().add_plugin_v2(plug_inputs, gemm_plug)
    layer.get_input(0).set_dynamic_range(-127, 127)
    # layer.get_output(0).set_dynamic_range(-1000000, 1000000)
    # layer.get_input(1).set_dynamic_range(-127, 127)
    # out = _create_tensor(layer.get_output(0), layer)

    # out = mul(out, unsqueeze(unsqueeze(scales_a, 0), 0))
    return _create_tensor(layer.get_output(0), layer)

def _gemm_plugin(input: Tensor,
                 mat2: Tensor,
                 transa: bool = False,
                 transb: bool = False) -> Tensor:
    plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'Gemm', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plg_creator is not None

    transa = 1 if transa else 0
    transa = trt.PluginField("transa", np.array(transa, dtype=np.int32),
                             trt.PluginFieldType.INT32)
    transb = 1 if transb else 0
    transb = trt.PluginField("transb", np.array(transb, dtype=np.int32),
                             trt.PluginFieldType.INT32)
    p_dtype = 'float16'#default_net().plugin_config.gemm_plugin
    pf_type = trt.PluginField(
        "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
        trt.PluginFieldType.INT32)
    pfc = trt.PluginFieldCollection([transa, transb, pf_type])
    gemm_plug = plg_creator.create_plugin("gemm", pfc)
    plug_inputs = [input.trt_tensor, mat2.trt_tensor]
    layer = default_trtnet().add_plugin_v2(plug_inputs, gemm_plug)
    return _create_tensor(layer.get_output(0), layer)


class Linear(Module):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 dtype=None,
                 tp_group=None,
                 tp_size=1,
                 gather_output=True,
                 share_weight=None,
                 quant_wa=False,
                 int8_gemm=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features // tp_size
        self.dtype = dtype
        self.int8_gemm = int8_gemm

        if not share_weight:
            self.weight = Parameter(shape=(self.out_features, self.in_features if not self.int8_gemm else self.in_features),
                                    dtype=dtype)
        else:
            self.weight = share_weight

        self.tp_size = tp_size
        self.tp_group = tp_group
        self.gather_output = gather_output

        if bias:
            self.bias = Parameter(shape=(self.out_features, ), dtype=dtype)
        else:
            self.register_parameter('bias', None)

        if int8_gemm:
            self.scale = Parameter(shape=(out_features, 1), dtype=trt.float32)
            self.scale_A = Parameter(shape=(1, 128, 1), dtype=trt.float32)
        else:
            self.register_parameter("scale", None)
            self.register_parameter("scale_A", None)

    def forward(self, x):
        if default_net().plugin_config.gemm_plugin:
            x = _gemm_plugin(x, self.weight.value, transb=True)
        elif self.int8_gemm:
            x = div(x, self.scale_A.value)
            x = round(x)
            x = clip(x, -127, 127)
            x = x.cast(trt.int8)
            y = self.weight.value.cast('int8')
            x = _w8a8gemm(x, y, self.scale_A.value, self.scale.value)
        else:
            x = matmul(x, self.weight.value, transb=True)

        if self.bias is not None:
            x = x + self.bias.value

        if self.gather_output and self.tp_size > 1 and self.tp_group is not None:
            # 1. [dim0, local_dim] -> [dim0 * tp_size, local_dim]
            x = allgather(x, self.tp_group)

            # 2. [dim0 * tp_size, local_dim] -> [dim0, local_dim * tp_size]
            # 2.1 split
            split_size = shape(x, dim=0) / self.tp_size
            ndim = x.ndim()
            starts = [constant(int32_array([0])) for _ in range(ndim)]
            sizes = [shape(x, dim=d) for d in range(ndim)]
            sizes[0] = split_size
            sections = []
            for i in range(self.tp_size):
                starts[0] = split_size * i
                sections.append(slice(x, concat(starts), concat(sizes)))
            # 2.2 concat
            x = concat(sections, dim=1)

        return x


ColumnLinear = Linear


class RowLinear(Module):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 dtype=None,
                 tp_group=None,
                 tp_size=1,
                 quant_wa=False,
                 int8_gemm=False):
        super().__init__()
        self.in_features = in_features // tp_size
        self.out_features = out_features
        self.dtype = dtype
        self.int8_gemm = int8_gemm

        self.weight = Parameter(shape=(self.out_features, self.in_features if not self.int8_gemm else self.in_features),
                                dtype=dtype)

        if bias:
            self.bias = Parameter(shape=(self.out_features, ), dtype=dtype)
        else:
            self.register_parameter('bias', None)

        self.tp_group = tp_group
        self.tp_size = tp_size

        if int8_gemm:
            self.scale = Parameter(shape=(out_features, 1), dtype=trt.float32)
            self.scale_A = Parameter(shape=(1, 128, 1), dtype=trt.float32)
        else:
            self.register_parameter("scale", None)
            self.register_parameter("scale_A", None)

    def forward(self, x):
        if default_net().plugin_config.gemm_plugin:
            x = _gemm_plugin(x, self.weight.value, transb=True)
        elif self.int8_gemm:
            x = div(x, self.scale_A.value)
            x = round(x)
            x = clip(x, -127, 127)
            x = x.cast(trt.int8)
            y = self.weight.value.cast('int8')
            # x = mul(div(x, self.scale_A.value.view([1, 1, self.scale_A.value.shape[0]])), self.scale_A.value.view([1, 1, self.scale_A.value.shape[0]]))
            # x = _gemm_plugin(x, y, transb=True)
            x = _w8a8gemm(x, y, self.scale_A.value, self.scale.value)
            # x = mul(x, self.scale_A.value)
            # x = mul(x, self.scale.value.view([1, 1, self.scale.value.shape[0]]))
        else:
            x = matmul(x, self.weight.value, transb=True)

        if self.tp_size > 1 and self.tp_group is not None:
            x = allreduce(x, self.tp_group)

        if self.bias is not None:
            x = x + self.bias.value

        return x
