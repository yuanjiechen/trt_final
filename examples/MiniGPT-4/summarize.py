import argparse
import copy
import json
import os

import numpy as np
import torch
from datasets import load_dataset, load_metric
from evaluate import load
from transformers import AutoModelForCausalLM, LlamaTokenizer

import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm.logger import logger

from build import get_engine_name  # isort:skip

from test_gpt import init, get_image
from dataloader import Image_text_set

def main(args):
    runtime_rank = tensorrt_llm.mpi_rank()
    logger.set_level(args.log_level)

    test_hf = args.load_torch
    test_trt_llm = not args.load_torch
    hf_model_location = args.hf_model_location
    profiler.start('load tokenizer')
    tokenizer = LlamaTokenizer.from_pretrained(hf_model_location,
                                               use_fast=False)
    profiler.stop('load tokenizer')
    tensorrt_llm.logger.info(
        f'Load tokenizer takes: {profiler.elapsed_time_in_sec("load tokenizer")} sec'
    )
    tokenizer.pad_token = tokenizer.eos_token

    dataset_cnn = Image_text_set("./test_data")

    max_batch_size = 1

    if test_trt_llm:
        chat, _ = init(torch_from_summarize=False)

    if test_hf:
        chat_torch, _ = init(torch_from_summarize=True)

    def summarize_tensorrt_llm(image, text):
        # Extract a list of tensors of shape beam_width x output_ids.
        output_beams_list = [chat.forward(image, text)]
        return output_beams_list

    def summarize_hf(image, text):

        output_lines_list = [chat_torch.forward(image, text)]
        return output_lines_list



    ite_count = 0
    data_point_idx = 0
    while (data_point_idx < len(dataset_cnn)) and (ite_count <
                                                           args.max_ite):
        if runtime_rank == 0:
            logger.debug(
                f"run data_point {data_point_idx} ~ {data_point_idx + max_batch_size}"
            )
        image, text = dataset_cnn[data_point_idx]

        if test_trt_llm:
            profiler.start('tensorrt_llm')
            summary_tensorrt_llm = summarize_tensorrt_llm(image, text)
            profiler.stop('tensorrt_llm')
            with open("trt_output.txt", "a+") as f:
                f.write(summary_tensorrt_llm[0].strip('\n')  + "\n")

        if test_hf:
            profiler.start('hf')
            summary_hf = summarize_hf(image, text)
            profiler.stop('hf')
            with open("torch_output.txt", "a+") as f:
                f.write(summary_hf[0].split('\n')[0] + "\n")



        data_point_idx += max_batch_size
        ite_count += 1



    print(
        f'TensorRT-LLM (total latency: {profiler.elapsed_time_in_sec("tensorrt_llm")} sec)'
    )
    print(
        f'Pytorch (total latency: {profiler.elapsed_time_in_sec("hf")} sec)'
    )

def eval():
    if not os.path.exists("trt_output.txt"): raise FileNotFoundError("Please run tensorrt summary first!")
    if not os.path.exists("torch_output.txt"): raise FileNotFoundError("Please run torch summary first!")
    
    i = 0
    trt_output_ = []
    torch_output_ = []
    with open("trt_output.txt", "r") as f:
        trt_output = f.readlines()
        while i < len(trt_output):
            if len(trt_output[i]) == 1: i += 1
            else: trt_output_.append(trt_output[i])
            i += 1
    
    i = 0
    with open("torch_output.txt", "r") as f:
        torch_output = f.readlines()
        while i < len(torch_output):
            if len(torch_output[i]) == 1: i += 1
            else: torch_output_.append(torch_output[i])
            i += 1

    metric_tensorrt_llm = [load("rouge")]
    for line_trt, line_torch in zip(trt_output_, torch_output_):
        metric_tensorrt_llm[0].add_batch(
            predictions=[line_trt],
            references=[line_torch])

    computed_metrics_tensorrt_llm = metric_tensorrt_llm[0].compute()
    for key in computed_metrics_tensorrt_llm.keys():
        print(f'  {key} : {computed_metrics_tensorrt_llm[key]}')
            
        

    # if args.check_accuracy == 0:
    #     assert computed_metrics_tensorrt_llm['rouge1'].mid[
    #         2] * 100 > args.tensorrt_llm_rouge1_threshold

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_model_location',
                        type=str,
                        default='./weight')
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--engine_dir', type=str, default='llama_outputs')
    parser.add_argument("--load_torch", action="store_true", help="load torch model or trt engine")
    parser.add_argument("--eval", action="store_true", help="eval the result")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_ite', type=int, default=40)
    parser.add_argument('--check_accuracy', action='store_true')
    parser.add_argument('--tensorrt_llm_rouge1_threshold',
                        type=float,
                        default=15.0)
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")

    args = parser.parse_args()
    if args.eval: eval()
    else: main(args)
