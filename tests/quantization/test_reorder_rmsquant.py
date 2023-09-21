import unittest

import _utils
import numpy as np
import tensorrt as trt
import torch
from parameterized import parameterized
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, TrtRunner

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm.functional import rms_normquant_reorder


class TestSmoothQuantGemm(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    def rms_reorder(self, bs_inseq, hidensize, dtype):
        # Init operands for multiplication in int32
        shape1 = (1,bs_inseq, hidensize)
        input = torch.randint(-128, 128, shape1, dtype=torch.float16)
        shape2 = (hidensize,)
        weight = torch.randint(-128, 128, shape2, dtype=torch.float16)
        # Temporary hack to overcome TRT int8 plugin limitation
        normalized_shape = (hidensize,)

        # Init scales in fp32
        shape_scale_a = (hidensize, )
        scale_a_torch = torch.ones(shape_scale_a, dtype=torch.float32)
       
        shape_index = (hidensize,) 
        dst_index = torch.arange(hidensize-1,-1,-1,dtype=torch.int32)

        # Create builder
        builder = tensorrt_llm.Builder()
        # Create empty network
        net = builder.create_network()
        # Allow SQ plugin of dtype type
        net.plugin_config.set_reorder_plugin()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            # Init TensorRT-LLM tensor for mat1
            input_ = Tensor(name='input',
                       shape=input.shape,
                       dtype=tensorrt_llm._utils.str_dtype_to_trt("float16"))
            # Init TensorRT-LLM tensor for mat2
            weight_ = Tensor(name='weight',
                       shape=weight.shape,
                       dtype=tensorrt_llm._utils.str_dtype_to_trt("float16"))
            # Init TensorRT-LLM tensor for per token scaling
            scale_a = Tensor(
                name='scale',
                shape=scale_a_torch.shape,
                dtype=tensorrt_llm._utils.str_dtype_to_trt("float32"))
            # Init TensorRT-LLM tensor for per channel scaling
            dst_index_ = Tensor(
                name='dst_index',
                shape=dst_index.shape,
                dtype=tensorrt_llm._utils.str_dtype_to_trt("int32"))
            # Get output tensor for SQ gemm
            output = rms_normquant_reorder(input_, normalized_shape,weight_,1e-06 ,dst_index_,scale_a,
                                  ).trt_tensor
            output.name = 'output'
            network.mark_output(output)
            output.dtype = tensorrt_llm._utils.str_dtype_to_trt(dtype)

        # Build engine consisting of only SQ Gemm
        build_engine = EngineFromNetwork(
            (builder.trt_builder, net.trt_network),
            config=CreateConfig(
                int8=False,
                fp16=(dtype == "float16"),
                memory_pool_limits={trt.MemoryPoolType.WORKSPACE: 333554432}))

        # Infer engine
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(
                feed_dict={
                    'input': input.numpy(),
                    'weight': weight.numpy(),
                    'scale': scale_a_torch.numpy(),
                    'dst_index': dst_index.numpy()
                })
            print(outputs['output'])
            outputs_ = self.test_rms_reorder_plugin(input,weight,dst_index)
            
            print("------------------------------------------------------------")
            print(outputs_)
            print("-----------------------------------------------------------")
            print(torch.norm(torch.from_numpy(outputs['output']) - outputs_))



    def test_matmul(self, dtype, per_token_scaling, per_channel_scaling):
        bs = 1
        inseq = 1
        hidden_size = 4096

        self.rms_reorder(bs * inseq, hidden_size, dtype)

    def test_rms_reorder_plugin(self,hidden_states, weight, index):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + 1e-6)
        hidden_states = weight * hidden_states.to(input_dtype)
        hidden_states = torch.index_select(hidden_states, 2, index)
        return hidden_states



if __name__ == '__main__':
    debug = TestSmoothQuantGemm()
    debug.test_matmul('float16', True, True)