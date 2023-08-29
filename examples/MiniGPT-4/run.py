import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import LlamaTokenizer

import tensorrt_llm
from tensorrt_llm.runtime import ModelConfig, SamplingConfig

# from build import get_engine_name  # isort:skip

EOS_TOKEN = 2277
PAD_TOKEN = 2277

class Generation:
    def __init__(self, max_output_len, log_level, engine_dir, tokenizer_dir, num_beams):
        tensorrt_llm.logger.set_level(log_level)

        config_path = os.path.join(engine_dir, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        use_gpt_attention_plugin = config['plugin_config']['gpt_attention_plugin']
        remove_input_padding = config['plugin_config']['remove_input_padding']
        dtype = config['builder_config']['precision']
        world_size = config['builder_config']['tensor_parallel']
        num_heads = config['builder_config']['num_heads'] // world_size
        hidden_size = config['builder_config']['hidden_size'] // world_size
        vocab_size = config['builder_config']['vocab_size']
        num_layers = config['builder_config']['num_layers']
        multi_query_mode = config['builder_config']['multi_query_mode']

        runtime_mapping = tensorrt_llm.Mapping(world_size, 0)

        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_dir, legacy=False)

        model_config = ModelConfig(num_heads=num_heads,
                                hidden_size=hidden_size,
                                vocab_size=vocab_size,
                                num_layers=num_layers,
                                gpt_attention_plugin=use_gpt_attention_plugin,
                                multi_query_mode=multi_query_mode,
                                remove_input_padding=remove_input_padding)
        self.sampling_config = SamplingConfig(end_id=EOS_TOKEN,
                                        pad_id=PAD_TOKEN,
                                        num_beams=num_beams)
        self.num_beams = num_beams
        self.max_output_len = max_output_len
        engine_name = "llama_float16_tp1_rank0.engine"#get_engine_name('llama', dtype, world_size, runtime_rank)
        serialize_path = os.path.join(engine_dir, engine_name)
        with open(serialize_path, 'rb') as f:
            engine_buffer = f.read()
        self.decoder = tensorrt_llm.runtime.GenerationSession(model_config,
                                                        engine_buffer,
                                                        runtime_mapping)

    def generate(self, input_values):

        # inputs: Give the following image: <Img>ImageContent</Img>. 
        # You will be able to see the image once I provide it to you. 
        # Please answer my questions.###Human: <Img><ImageHere></Img> what this man doing in image###Assistant:

        input_data = input_values.half()
        input_ids = torch.randint(20, 50, [input_data.size(0), input_data.size(1)], dtype=torch.int32).cuda()

        input_lengths = torch.cuda.IntTensor([input_ids.size(1)])

        max_input_length = torch.max(input_lengths).item()
        self.decoder.setup(input_lengths.size(0), max_input_length, self.max_output_len)

        output_ids = self.decoder.decode(input_ids, input_lengths, self.sampling_config, input_data=input_data)
        torch.cuda.synchronize()


        for b in range(input_lengths.size(0)):
            # inputs = input_tokens[b]
            # input_text = tokenizer.decode(inputs)
            # print(f'Input: \"{input_text}\"')
            if self.num_beams <= 1:
                output_begin = max_input_length
                outputs = output_ids[b][0][output_begin:].tolist()
                output_text = self.tokenizer.decode(outputs)
                print(f'Output: \"{output_text}\"')
            else:
                for beam in range(self.num_beams):
                    output_begin = input_lengths[b]
                    output_end = input_lengths[b] + self.max_output_len
                    outputs = output_ids[b][beam][
                        output_begin:output_end].tolist()
                    output_text = self.tokenizer.decode(outputs)
                    print(f'Output: \"{output_text}\"')

        output_ids = output_ids.reshape((-1, output_ids.size(2)))
        return output_text
