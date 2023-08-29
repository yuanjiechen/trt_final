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

EOS_TOKEN = 2
PAD_TOKEN = 2


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_output_len', type=int, required=True)
    parser.add_argument('--log_level', type=str, default='error')
    parser.add_argument('--engine_dir', type=str, default='llama_outputs')
    parser.add_argument('--tokenizer_dir',
                        type=str,
                        default=".",
                        help="Directory containing the tokenizer.model.")
    parser.add_argument('--input_text',
                        type=str,
                        default='Born in north-east France, Soyer trained as a')
    parser.add_argument(
        '--input_tokens',
        dest='input_file',
        type=str,
        help=
        'CSV or Numpy file containing tokenized input. Alternative to text input.',
        default=None)
    parser.add_argument('--output_csv',
                        type=str,
                        help='CSV file where the tokenized output is stored.',
                        default=None)
    parser.add_argument('--output_npy',
                        type=str,
                        help='Numpy file where the tokenized output is stored.',
                        default=None)
    parser.add_argument('--num_beams',
                        type=int,
                        help="Use beam search if num_beams >1",
                        default=1)
    return parser.parse_args()


def generate(
    max_output_len: int,
    log_level: str = 'error',
    engine_dir: str = '/root/workspace/trt_final/examples/llama/llama_outputs',
    input_text: str = 'Born in north-east France, Soyer trained as a',
    tokenizer_dir: str = "/home/player/docker_data/weight/",
    num_beams: int = 1,
):
    tensorrt_llm.logger.set_level(log_level)

    config_path = os.path.join(engine_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    use_gpt_attention_plugin = config['plugin_config']['gpt_attention_plugin']
    remove_input_padding = config['plugin_config']['remove_input_padding']
    dtype = config['builder_config']['precision']
    world_size = config['builder_config']['tensor_parallel']
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'
    num_heads = config['builder_config']['num_heads'] // world_size
    hidden_size = config['builder_config']['hidden_size'] // world_size
    vocab_size = config['builder_config']['vocab_size']
    num_layers = config['builder_config']['num_layers']
    multi_query_mode = config['builder_config']['multi_query_mode']

    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_dir, legacy=False)

    model_config = ModelConfig(num_heads=num_heads,
                               hidden_size=hidden_size,
                               vocab_size=vocab_size,
                               num_layers=num_layers,
                               gpt_attention_plugin=use_gpt_attention_plugin,
                               multi_query_mode=multi_query_mode,
                               remove_input_padding=remove_input_padding)
    sampling_config = SamplingConfig(end_id=EOS_TOKEN,
                                     pad_id=PAD_TOKEN,
                                     num_beams=num_beams)

    engine_name = "llama_float16_tp1_rank0.engine"#get_engine_name('llama', dtype, world_size, runtime_rank)
    serialize_path = os.path.join(engine_dir, engine_name)
    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    decoder = tensorrt_llm.runtime.GenerationSession(model_config,
                                                     engine_buffer,
                                                     runtime_mapping)
    input_tokens = []

    input_tokens.append(
        tokenizer.encode(input_text, add_special_tokens=False))


    input_ids = None
    input_lengths = None

    input_ids = torch.cuda.IntTensor(input_tokens)
    input_lengths = torch.cuda.IntTensor([input_ids.size(1)])


    max_input_length = torch.max(input_lengths).item()
    decoder.setup(input_lengths.size(0), max_input_length, max_output_len)

    output_ids = decoder.decode(input_ids, input_lengths, sampling_config)
    torch.cuda.synchronize()

    if runtime_rank == 0:

        for b in range(input_lengths.size(0)):
            inputs = input_tokens[b]
            input_text = tokenizer.decode(inputs)
            print(f'Input: \"{input_text}\"')
            if num_beams <= 1:
                output_begin = max_input_length
                outputs = output_ids[b][0][output_begin:].tolist()
                output_text = tokenizer.decode(outputs)
                print(f'Output: \"{output_text}\"')
            else:
                for beam in range(num_beams):
                    output_begin = input_lengths[b]
                    output_end = input_lengths[b] + max_output_len
                    outputs = output_ids[b][beam][
                        output_begin:output_end].tolist()
                    output_text = tokenizer.decode(outputs)
                    print(f'Output: \"{output_text}\"')

        output_ids = output_ids.reshape((-1, output_ids.size(2)))

    return


if __name__ == '__main__':
    args = parse_arguments()
    generate(**vars(args))
