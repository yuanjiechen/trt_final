from collections import OrderedDict

import tensorrt as trt
from tensorrt_llm._common import default_net
from tensorrt_llm._utils import pad_vocab_size, str_dtype_to_trt

from tensorrt_llm.functional import (RaggedTensor, Tensor, assertion, expand_mask,
                           gather_last_token_logits, shape, activation)
from tensorrt_llm.layers import (Attention, AttentionMaskType, ColumnLinear, Embedding,
                       GatedMLP, PositionEmbeddingType, RmsNorm, Linear)
from tensorrt_llm.module import Module, ModuleList


class test_module(Module):
    def __init__(self, act_type="relu"):
        super().__init__()
        self.act_type = act_type
        self.fc = Linear(2, 20, False)

    def forward(self, x):
        y = self.fc(x)
        y = activation(y, trt.ActivationType.RELU)

        y.mark_output("outputs", trt.DataType.FLOAT)
        return y
    
    def prepare_inputs(self, min_batch, opt_batch, max_batch):
        x = Tensor(name='inputs',
                        dtype=trt.float32,
                        shape=[-1, 2],
                        dim_range=OrderedDict([('batch_size', [[min_batch, opt_batch, max_batch]]), ('data', [2])]))
        
        return x