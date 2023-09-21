import argparse
import json
import os
import time
from pathlib import Path
from dataclasses import dataclass
from functools import partial
import ctypes

import tensorrt as trt
import torch
import torch.multiprocessing as mp
from transformers import LlamaConfig, LlamaForCausalLM

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.builder import Builder
from tensorrt_llm.logger import logger
from tensorrt_llm.models import weight_only_quantize
from tensorrt_llm.network import net_guard
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.layers import LayerNorm

from weight import load_from_hf_llama, load_from_hf_vit
from minigpt4.models.eva_vit import create_eva_vit_g  # isort:skip
from vit import VisionTransformer

MODEL_NAME = "llama"

# 2 routines: get_engine_name, serialize_engine
# are direct copy from gpt example, TODO: put in utils?

import onnx
import tensorrt as trt
from onnx import TensorProto, helper


@dataclass
class Vicuna_args:
    model_dir = "./weight/"
    meta_ckpt_dir = None
    dtype='float16' # [float32, bfloat16, float16]
    timing_cache = 'model.cache'
    log_level = 'info'
    vocab_size = 32000
    n_layer = 32
    n_positions = 2048
    n_embd = 4096 # hidden_size
    n_head = 32
    n_kv_head = None
    multiple_of = None
    ffn_dim_multiplier = 1
    inter_size = 11008
    hidden_act = 'silu'
    max_batch_size = 8
    max_input_len = 512
    max_output_len = 1024
    use_apt_attention_plugin = 'float16' # [float32, bfloat16, float16]
    use_gemm_plugin = 'float16' # [float32, bfloat16, float16]
    enable_debug_output = False
    builder_opt = None 
    output_dir = 'llama_outputs' # output dir
    remove_input_padding = False
    use_weight_only = False
    weight_only_precition = 'int8' #[int8, int4]
    quant_mode = None
    quant_wa = True
    int8_gemm = False

@dataclass
class ViT_args:
    img_size = 224
    patch_size = 14
    in_chans = 3
    num_classes = 1000
    embed_dim = 1408
    depth = 39
    num_heads = 1408 //88
    mlp_ratio = 4.3637
    qkv_bias = True
    qk_scale = None
    norm_layer = partial(LayerNorm, eps=1e-6)
    use_abs_pos_emb = True
    build_opt = None
    model_dir = ""
    output_dir = "llama_outputs"
    timing_cache = 'model.cache'
    dtype = 'float16'


def get_engine_name(model, dtype, tp_size, rank):
    return '{}_{}_tp{}_rank{}.engine'.format(model, dtype, tp_size, rank)


def serialize_engine(engine, path):
    logger.info(f'Serializing engine to {path}...')
    tik = time.time()
    with open(path, 'wb') as f:
        f.write(bytearray(engine))
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Engine serialized. Total time: {t}')


def parse_arguments():

    args = Vicuna_args()
    args_vit = ViT_args()

    if args.use_weight_only:
        args.quant_mode = QuantMode.use_weight_only(
            args.weight_only_precision == 'int4')
    else:
        args.quant_mode = QuantMode(0)
    # Since gpt_attenttion_plugin is the only way to apply RoPE now,
    # force use the plugin for now with the correct data type.
    args.use_gpt_attention_plugin = args.dtype
    if args.model_dir is not None:
        hf_config = LlamaConfig.from_pretrained(args.model_dir)
        args.inter_size = hf_config.intermediate_size  # override the inter_size for LLaMA
        args.n_embd = hf_config.hidden_size
        args.n_head = hf_config.num_attention_heads
        if hasattr(hf_config, "num_key_value_heads"):
            args.n_kv_head = hf_config.num_key_value_heads
        args.n_layer = hf_config.num_hidden_layers
        args.n_positions = hf_config.max_position_embeddings
        args.vocab_size = hf_config.vocab_size
        args.hidden_act = hf_config.hidden_act

    assert args.use_gpt_attention_plugin != None, "LLaMa must use gpt attention plugin"

    return args, args_vit


def build_rank_engine(builder: Builder,
                      builder_config: tensorrt_llm.builder.BuilderConfig,
                      engine_name, rank, multi_query_mode, args):
    '''
       @brief: Build the engine on the given rank.
       @param rank: The rank to build the engine.
       @param args: The cmd line arguments.
       @return: The built engine.
    '''
    kv_dtype = str_dtype_to_trt(args.dtype)

    # Initialize Module
    tensorrt_llm_llama = tensorrt_llm.models.LLaMAForCausalLM(
        num_layers=args.n_layer,
        num_heads=args.n_head,
        hidden_size=args.n_embd,
        vocab_size=args.vocab_size,
        hidden_act=args.hidden_act,
        max_position_embeddings=args.n_positions,
        dtype=kv_dtype,
        mlp_hidden_size=args.inter_size,
        neox_rotary_style=True,
        multi_query_mode=multi_query_mode,
        tensor_parallel=1,
        tensor_parallel_group=list(range(1)),
        quant_wa = args.quant_wa,
        int8_gemm=args.int8_gemm)
    if args.use_weight_only and args.weight_only_precision == 'int8':
        tensorrt_llm_llama = weight_only_quantize(tensorrt_llm_llama,
                                                  QuantMode.use_weight_only())
    elif args.use_weight_only and args.weight_only_precision == 'int4':
        tensorrt_llm_llama = weight_only_quantize(
            tensorrt_llm_llama,
            QuantMode.use_weight_only(use_int4_weights=True))
    elif args.quant_wa == True:
        # ctypes.cdll.LoadLibrary("/root/workspace/trt_final/cpp/build/tensorrt_llm/libtensorrt_llm_static.a")
        setattr(tensorrt_llm_llama, "quant_wa", True)
    if args.int8_gemm == True:
        setattr(tensorrt_llm_llama, "int8_gemm", True)
    if args.model_dir is not None:
        logger.info(f'Loading HF LLaMA ... from {args.model_dir}')
        tik = time.time()
        hf_llama = LlamaForCausalLM.from_pretrained(
            args.model_dir,
            device_map={
                "model": "cpu",
                "lm_head": "cpu"
            },  # Load to CPU memory
            torch_dtype="auto")
        tok = time.time()
        t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
        logger.info(f'HF LLaMA loaded. Total time: {t}')
        load_from_hf_llama(tensorrt_llm_llama,
                           hf_llama,
                           rank,
                           1,#args.world_size,
                           dtype=args.dtype,
                           multi_query_mode=multi_query_mode)
        del hf_llama

    # Module -> Network
    network = builder.create_network()
    network.trt_network.name = engine_name
    if args.use_gpt_attention_plugin:
        network.plugin_config.set_gpt_attention_plugin(
            dtype=args.use_gpt_attention_plugin)
    if args.use_gemm_plugin:
        network.plugin_config.set_gemm_plugin(dtype=args.use_gemm_plugin)
    if args.use_weight_only:
        network.plugin_config.set_weight_only_quant_matmul_plugin(
            dtype='float16')
    if args.remove_input_padding:
        network.plugin_config.enable_remove_input_padding()
    if args.quant_wa:
        network.plugin_config.set_reorder_plugin()

    with net_guard(network):
        # Prepare
        network.set_named_parameters(tensorrt_llm_llama.named_parameters())

        # Forward
        inputs = tensorrt_llm_llama.prepare_inputs(args.max_batch_size,
                                                   args.max_input_len,
                                                   args.max_output_len, True,
                                                   1)
        tensorrt_llm_llama(*inputs)
        if args.enable_debug_output:
            # mark intermediate nodes' outputs
            for k, v in tensorrt_llm_llama.named_network_outputs():
                v = v.trt_tensor
                v.name = k
                network.trt_network.mark_output(v)
                v.dtype = kv_dtype
    for i in range(network.trt_network.num_layers):
        layer = network.trt_network.get_layer(i)
        if layer.type == trt.LayerType.ELEMENTWISE \
            and "SUB" not in layer.name and "MIN" not in layer.name \
            and "SUM" not in layer.name and "EQUAL" not in layer.name and "POW" not in layer.name:
            layer.precision = trt.float32
            layer.set_output_type(0, trt.float32)
            print(layer.name)
    engine = None

    # Network -> Engine
    engine = builder.build_engine(network, builder_config)
    if rank == 0:
        config_path = os.path.join(args.output_dir, 'config.json')
        builder.save_config(builder_config, config_path)
    return engine


def build(rank, args:Vicuna_args):
    torch.cuda.set_device(0)
    tensorrt_llm.logger.set_level(args.log_level)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    multi_query_mode = (args.n_kv_head
                        is not None) and (args.n_kv_head != args.n_head)

    # when doing serializing build, all ranks share one engine
    builder = Builder()

    cache = None
    cur_rank = 0

    builder_config = builder.create_builder_config(
        name=MODEL_NAME,
        precision=args.dtype,
        timing_cache=args.timing_cache if cache is None else cache,
        tensor_parallel=1,#args.world_size,  # TP only
        parallel_build=False,
        num_layers=args.n_layer,
        num_heads=args.n_head,
        hidden_size=args.n_embd,
        vocab_size=args.vocab_size,
        hidden_act=args.hidden_act,
        max_position_embeddings=args.n_positions,
        max_batch_size=args.max_batch_size,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len,
        int8=args.int8_gemm, #args.quant_mode.has_act_and_weight_quant(),
        opt_level=args.builder_opt,
        multi_query_mode=multi_query_mode)
    engine_name = get_engine_name(MODEL_NAME, args.dtype, 1,
                                    cur_rank)
    engine = build_rank_engine(builder, builder_config, engine_name,
                                cur_rank, multi_query_mode, args)
    assert engine is not None, f'Failed to build engine for rank {cur_rank}'

    cache = builder_config.trt_builder_config.get_timing_cache()

    serialize_engine(engine, os.path.join(args.output_dir, engine_name))

    ok = builder.save_timing_cache(
        builder_config, os.path.join(args.output_dir, "model.cache"))
    assert ok, "Failed to save timing cache."

def build_vit(rank, args_vit:ViT_args):
    if not os.path.exists(args_vit.output_dir):
        os.makedirs(args_vit.output_dir)

    model_name = 'vit'
    builder = Builder()
    builder_config = builder.create_builder_config(
        name=model_name,
        precision=args_vit.dtype,
        timing_cache=args_vit.timing_cache,
        opt_level=args_vit.build_opt
    )
    builder_config.trt_builder_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    engine_name = get_engine_name(model_name, args_vit.dtype, 0, 0)
    dtype = str_dtype_to_trt(args_vit.dtype)
    vit = VisionTransformer(img_size=args_vit.img_size,
                            patch_size=args_vit.patch_size,
                            in_chans=args_vit.in_chans,
                            num_classes=args_vit.num_classes,
                            embed_dim=args_vit.embed_dim,
                            depth=args_vit.depth,
                            num_heads=args_vit.num_heads,
                            mlp_ratio=args_vit.mlp_ratio,
                            qkv_bias=args_vit.qkv_bias,
                            qk_scale=args_vit.qk_scale,
                            norm_layer=args_vit.norm_layer,
                            use_abs_pos_emb=args_vit.use_abs_pos_emb,
                            dtype=dtype)

    # load form vit here
    vit_torch = create_eva_vit_g(img_size=args_vit.img_size)
    load_from_hf_vit(vit, vit_torch, dtype=args_vit.dtype)
    network = builder.create_network()
    network.trt_network.name = engine_name
    with net_guard(network):
        network.set_named_parameters(vit.named_parameters())
        inputs = vit.prepare_inputs()
        vit(inputs)

    # Network -> Engine
    engine = builder.build_engine(network, builder_config)

    config_path = os.path.join(args_vit.output_dir, 'config.json')
    builder.save_config(builder_config, config_path)

    serialize_engine(engine, os.path.join(args_vit.output_dir, engine_name))

    ok = builder.save_timing_cache(
        builder_config, os.path.join(args_vit.output_dir, "model.cache"))
    assert ok, "Failed to save timing cache."

if __name__ == '__main__':
    args, args_vit = parse_arguments()
    logger.set_level(args.log_level)
    tik = time.time()

    logger.info('Serially build TensorRT engines.')
    build(0, args)
    # build_vit(0, args_vit) never open this line !!!

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Total time of building all {1} engines: {t}')#world_size
