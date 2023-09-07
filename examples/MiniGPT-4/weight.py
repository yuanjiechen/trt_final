import time
from pathlib import Path

import numpy as np
import torch

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy
from tensorrt_llm.quantization import QuantMode


def extract_layer_idx(name):
    ss = name.split('.')
    for s in ss:
        if s.isdigit():
            return s
    return None


def split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    if len(v.shape) == 1:
        return np.ascontiguousarray(np.split(v, tp_size)[idx])
    else:
        return np.ascontiguousarray(np.split(v, tp_size, axis=dim)[idx])


def load_from_hf_llama(tensorrt_llm_llama,
                       hf_llama,
                       rank=0,
                       tensor_parallel=1,
                       dtype="float32",
                       multi_query_mode=False):
    tensorrt_llm.logger.info('Loading weights from HF LLaMA...')
    tik = time.time()

    quant_mode = getattr(tensorrt_llm_llama, 'quant_mode', QuantMode(0))
 
    use_weight_only = False#quant_mode.is_weight_only()
    has_act_and_weight_quant = quant_mode.has_act_and_weight_quant()

    model_params = dict(hf_llama.named_parameters())
    for l in range(hf_llama.config.num_hidden_layers):
        prefix = f'model.layers.{l}.self_attn.'
        q_weight = model_params[prefix + 'q_proj.weight']
        k_weight = model_params[prefix + 'k_proj.weight']
        v_weight = model_params[prefix + 'v_proj.weight']
        if multi_query_mode:
            head_size = tensorrt_llm_llama.hidden_size // tensorrt_llm_llama.num_heads
            assert k_weight.shape[0] == tensor_parallel * head_size
            assert v_weight.shape[0] == tensor_parallel * head_size
            qkv_weight = [q_weight, k_weight, v_weight]
        else:
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)

        model_params[prefix + 'qkv_proj.weight'] = qkv_weight

    torch_dtype = str_dtype_to_torch(dtype)
    for k, v in model_params.items():
        if isinstance(v, list):
            v = [torch_to_numpy(vv.to(torch_dtype).detach().cpu()) for vv in v]
        else:
            v = torch_to_numpy(v.to(torch_dtype).detach().cpu())
        if 'model.embed_tokens.weight' in k:
            pass
            # tensorrt_llm_llama.vocab_embedding.weight.value = v
        elif 'model.norm.weight' in k:
            tensorrt_llm_llama.ln_f.weight.value = v
        elif 'lm_head.weight' in k:
            tensorrt_llm_llama.lm_head.weight.value = np.ascontiguousarray(
                split(v, tensor_parallel, rank))
        else:
            layer_idx = extract_layer_idx(k)
            if layer_idx is None:
                continue
            idx = int(layer_idx)
            if idx >= tensorrt_llm_llama.num_layers:
                continue
            
            if has_act_and_weight_quant:
                continue

            else: 
                if 'input_layernorm.weight' in k:
                    tensorrt_llm_llama.layers[idx].input_layernorm.weight.value = v
                elif 'post_attention_layernorm.weight' in k:
                    dst = tensorrt_llm_llama.layers[idx].post_layernorm.weight
                    dst.value = v
                elif 'self_attn.qkv_proj.weight' in k:
                    dst = tensorrt_llm_llama.layers[idx].attention.qkv.weight
                    if multi_query_mode:
                        assert isinstance(v, list) and len(v) == 3
                        wq = split(v[0], tensor_parallel, rank)
                        wk = split(v[1], tensor_parallel, rank)
                        wv = split(v[2], tensor_parallel, rank)
                        split_v = np.concatenate((wq, wk, wv))
                    else:
                        q_emb = v.shape[0] // 3
                        model_emb = v.shape[1]
                        v = v.reshape(3, q_emb, model_emb)
                        split_v = split(v, tensor_parallel, rank, dim=1)
                        split_v = split_v.reshape(3 * (q_emb // tensor_parallel),
                                                model_emb)
                    if use_weight_only:
                        v = np.ascontiguousarray(split_v.transpose())
                        processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                            torch.tensor(v), plugin_weight_only_quant_type)
                        # workaround for trt not supporting int8 inputs in plugins currently
                        dst.value = processed_torch_weights.view(
                            dtype=torch.float32).numpy()
                        scales = tensorrt_llm_llama.layers[
                            idx].attention.qkv.per_channel_scale
                        scales.value = torch_weight_scales.numpy()
                    else:
                        dst.value = np.ascontiguousarray(split_v)
                elif 'self_attn.o_proj.weight' in k:
                    dst = tensorrt_llm_llama.layers[idx].attention.dense.weight
                    split_v = split(v, tensor_parallel, rank, dim=1)
                    if use_weight_only:
                        v = np.ascontiguousarray(split_v.transpose())
                        processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                            torch.tensor(v), plugin_weight_only_quant_type)
                        # workaround for trt not supporting int8 inputs in plugins currently
                        dst.value = processed_torch_weights.view(
                            dtype=torch.float32).numpy()
                        scales = tensorrt_llm_llama.layers[
                            idx].attention.dense.per_channel_scale
                        scales.value = torch_weight_scales.numpy()
                    else:
                        dst.value = np.ascontiguousarray(split_v)
                elif 'mlp.up_proj.weight' in k:
                    dst = tensorrt_llm_llama.layers[idx].mlp.gate.weight
                    split_v = split(v, tensor_parallel, rank, dim=0)
                    if use_weight_only:
                        v = np.ascontiguousarray(split_v.transpose())
                        processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                            torch.tensor(v), plugin_weight_only_quant_type)
                        # workaround for trt not supporting int8 inputs in plugins currently
                        dst.value = processed_torch_weights.view(
                            dtype=torch.float32).numpy()
                        scales = tensorrt_llm_llama.layers[
                            idx].mlp.gate.per_channel_scale
                        scales.value = torch_weight_scales.numpy()
                    else:
                        dst.value = np.ascontiguousarray(split_v)
                elif 'mlp.down_proj.weight' in k:
                    dst = tensorrt_llm_llama.layers[idx].mlp.proj.weight
                    split_v = split(v, tensor_parallel, rank, dim=1)
                    if use_weight_only:
                        v = np.ascontiguousarray(split_v.transpose())
                        processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                            torch.tensor(v), plugin_weight_only_quant_type)
                        # workaround for trt not supporting int8 inputs in plugins currently
                        dst.value = processed_torch_weights.view(
                            dtype=torch.float32).numpy()
                        scales = tensorrt_llm_llama.layers[
                            idx].mlp.proj.per_channel_scale
                        scales.value = torch_weight_scales.numpy()
                    else:
                        dst.value = np.ascontiguousarray(split_v)
                elif 'mlp.gate_proj.weight' in k:
                    dst = tensorrt_llm_llama.layers[idx].mlp.fc.weight
                    split_v = split(v, tensor_parallel, rank, dim=0)
                    if use_weight_only:
                        v = np.ascontiguousarray(split_v.transpose())
                        processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                            torch.tensor(v), plugin_weight_only_quant_type)
                        # workaround for trt not supporting int8 inputs in plugins currently
                        dst.value = processed_torch_weights.view(
                            dtype=torch.float32).numpy()
                        scales = tensorrt_llm_llama.layers[
                            idx].mlp.fc.per_channel_scale
                        scales.value = torch_weight_scales.numpy()
                    else:
                        dst.value = np.ascontiguousarray(split_v)
    if has_act_and_weight_quant:
        base_path = Path("/root/workspace/RPTQ4LLM-master/output/")
        for qlayer in base_path.glob("*.pth"):
            layer = torch.load(qlayer)
            
    '''
        input_layernorm.ori_layer_norm.weight
        input_layernorm.reorder_index
        input_layernorm.out_quantizer.scale
        input_layernorm.out_quantizer.round_zero_point

        post_attention_layernorm.ori_layer_norm.weight
        post_attention_layernorm.reorder_index
        post_attention_layernorm.out_quantizer.scale
        post_attention_layernorm.out_quantizer.round_zero_point

        mlp.gate_proj.weight
        mlp.gate_proj.bias
        mlp.gate_proj.scale
        mlp.gate_proj.round_zero_point

        mlp.up_proj.weight
        mlp.up_proj.bias
        mlp.up_proj.scale
        mlp.up_proj.round_zero_point       

        mlp.down_proj.weight
        mlp.down_proj.bias
        mlp.down_proj.scale
        mlp.down_proj.round_zero_point     
        mlp.down_proj.act_quantizer.scale
        mlp.down_proj.act_quantizer.round_zero_point

        self_attn.k_proj.weight
        self_attn.k_proj.bias
        self_attn.k_proj.scale
        self_attn.k_proj.round_zero_point

        self_attn.v_proj.weight
        self_attn.v_proj.bias
        self_attn.v_proj.scale
        self_attn.v_proj.round_zero_point

        self_attn.q_proj.weight
        self_attn.q_proj.bias
        self_attn.q_proj.scale
        self_attn.q_proj.round_zero_point

        self_attn.o_proj.weight
        self_attn.o_proj.bias
        self_attn.o_proj.scale
        self_attn.o_proj.round_zero_point
        self_attn.o_proj.act_quantizer.scale
        self_attn.o_proj.act_quantizer.round_zero_point
    '''

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
    return
def load_from_hf_vit(tensorrt_llm_vit,
                       hf_vit,
                       rank=0,
                       tensor_parallel=1,
                       dtype="float32",
                       multi_query_mode=False):
    tensorrt_llm.logger.info('Loading weights from HF VIT...')
    tik = time.time()

    quant_mode = getattr(tensorrt_llm_vit, 'quant_mode', QuantMode(0))
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    use_weight_only = quant_mode.is_weight_only()

    model_params = dict(hf_vit.named_parameters())
    for l in range(len(hf_vit.blocks)):
        prefix = f'blocks.{l}.attn.'
        k_bias_name = prefix + 'k_bias'
        k_bias = torch.zeros_like(model_params[prefix + 'q_bias'])

        model_params[k_bias_name] = k_bias

    torch_dtype = str_dtype_to_torch(dtype)
    for k, v in model_params.items():
        if isinstance(v, list):
            v = [torch_to_numpy(vv.to(torch_dtype).detach().cpu()) for vv in v]
        else:
            v = torch_to_numpy(v.to(torch_dtype).detach().cpu())
            
        if 'patch_embed.proj.weight' in k:
            tensorrt_llm_vit.patch_embed.proj.weight.value = v
        elif 'patch_embed.proj.bias' in k:
            tensorrt_llm_vit.patch_embed.proj.bias.value = v
        elif 'cls_token' in k:
            tensorrt_llm_vit.cls_token.value = v
        elif 'pos_embed' in k:
            tensorrt_llm_vit.pos_embed.value = v 
        else:
            layer_idx = extract_layer_idx(k)
            if layer_idx is None:
                continue
            idx = int(layer_idx)
            if idx >= len(tensorrt_llm_vit.blocks):
                continue
            if 'norm1.weight' in k:
                tensorrt_llm_vit.blocks[idx].norm1.weight.value = v
            elif 'norm2.weight' in k:
                tensorrt_llm_vit.blocks[idx].norm2.weight.value = v
            elif 'norm1.bias' in k:
                tensorrt_llm_vit.blocks[idx].norm1.bias.value = v
            elif 'norm2.bias' in k:
                tensorrt_llm_vit.blocks[idx].norm2.bias.value = v
            elif 'drop_path.weight' in k:
                continue ##tensorrt_llm_vit.blocks[idx].norm2.weight.value = v
            elif 'attn.qkv.weight' in k:
                dst = tensorrt_llm_vit.blocks[idx].attn.qkv.weight
                dst.value = v
                # if multi_query_mode:
                #     assert isinstance(v, list) and len(v) == 3
                #     wq = split(v[0], tensor_parallel, rank)
                #     wk = split(v[1], tensor_parallel, rank)
                #     wv = split(v[2], tensor_parallel, rank)
                #     split_v = np.concatenate((wq, wk, wv))
                # else:
                #     q_emb = v.shape[0] // 3
                #     model_emb = v.shape[1]
                #     v = v.reshape(3, q_emb, model_emb)
                #     split_v = split(v, tensor_parallel, rank, dim=1)
                #     split_v = split_v.reshape(3 * (q_emb // tensor_parallel),
                #                               model_emb)
                # if use_weight_only:
                #     v = np.ascontiguousarray(split_v.transpose())
                #     processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                #         torch.tensor(v), plugin_weight_only_quant_type)
                #     # workaround for trt not supporting int8 inputs in plugins currently
                #     dst.value = processed_torch_weights.view(
                #         dtype=torch.float32).numpy()
                #     scales = tensorrt_llm_vit.layers[
                #         idx].attention.qkv.per_channel_scale
                #     scales.value = torch_weight_scales.numpy()
                # else:
                #    dst.value = np.ascontiguousarray(split_v)
            elif 'attn.q_bias' in k:
                tensorrt_llm_vit.blocks[idx].attn.q_bias.value= v
            elif 'attn.k_bias' in k:
                tensorrt_llm_vit.blocks[idx].attn.k_bias.value= v
            elif 'attn.v_bias' in k:
                tensorrt_llm_vit.blocks[idx].attn.v_bias.value= v
            elif 'attn.attn_drop.weight' in k:
                continue ##tensorrt_llm_vit.blocks[idx].attn.v_bias.weight.value= v
            elif 'attn.proj.weight' in k:
                tensorrt_llm_vit.blocks[idx].attn.proj.weight.value= v
            elif 'attn.proj.bias' in k:
                tensorrt_llm_vit.blocks[idx].attn.proj.bias.value= v
            elif 'attn.proj_drop.weight' in k:
                continue
            elif 'mlp.fc1.weight' in k:
                tensorrt_llm_vit.blocks[idx].mlp.fc.weight.value= v
            elif 'mlp.fc2.weight' in k:
                tensorrt_llm_vit.blocks[idx].mlp.proj.weight.value = v
            elif 'mlp.fc1.bias' in k:
                tensorrt_llm_vit.blocks[idx].mlp.fc.bias.value= v
            elif 'mlp.fc2.bias' in k:
                tensorrt_llm_vit.blocks[idx].mlp.proj.bias.value = v
            elif 'mlp.drop.weight' in k:
                continue
            elif 'mlp.act.weight' in k:
                continue ##tensorrt_llm_vit.blocks[idx].attn.mlp.hidden_act = v
            

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
    return
