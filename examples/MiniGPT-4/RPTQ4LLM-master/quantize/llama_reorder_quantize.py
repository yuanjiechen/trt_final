import os

import torch
import torch.nn as nn
from quantize.int_linear import QuantLinear
from quantize.int_matmul import QuantMatMul
from quantize.reorder_layer_norm import ReorderLayerNorm
from models.int_llama_layer import QuantLlamaDecoderLayer
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from quantize.quant_transformer_layer import quant_layer, quant_layer_llama
from quantize.reorder_utils import (
    tensor_calc_reorder_index,
    ic_maxmin_dict,
    oc_maxmin_dict,
    oc_maxmin_dict_debug,
    layer_i0max_hook,
    layer_omax_hook,
)
# os.environ["CUDA_VISIBLE_DEVICES"]="5"
def R1_reorder(layer_norm, qproj, kproj, vproj, index, counts):
    layer_norm.register_buffer("reorder_index", index)
    layer_norm.out_quantizer.cluster_dim = 2
    layer_norm.out_quantizer.cluster_counts = counts

    qproj.weight.data = torch.index_select(qproj.weight.data, 1, index)
    qproj.set_ic_cluster_counts(counts, a_dim=None)

    kproj.weight.data = torch.index_select(kproj.weight.data, 1, index)
    kproj.set_ic_cluster_counts(counts, a_dim=None)
    vproj.weight.data = torch.index_select(vproj.weight.data, 1, index)
    vproj.set_ic_cluster_counts(counts, a_dim=None)

def R4_reorder(layer_norm, fc1, fc2, index, counts):
    layer_norm.register_buffer("reorder_index", index)

    layer_norm.out_quantizer.cluster_dim = 2
    layer_norm.out_quantizer.cluster_counts = counts

    fc1.weight.data = torch.index_select(fc1.weight.data, 1, index)
    fc1.set_ic_cluster_counts(counts, a_dim=None)

    fc2.weight.data = torch.index_select(fc2.weight.data, 1, index)
    fc2.set_ic_cluster_counts(counts, a_dim=None)

def R5_reorder(fc1, fc2, fc3, index, counts):
    fc1.weight.data = torch.index_select(fc1.weight.data, 0, index)
    if fc1.bias != None:
        fc1.bias.data = torch.index_select(fc1.bias.data, 0, index)

    fc2.weight.data = torch.index_select(fc2.weight.data, 0, index)
    if fc1.bias != None:
        fc2.bias.data = torch.index_select(fc2.bias.data, 0, index)

    fc3.weight.data = torch.index_select(fc3.weight.data, 1, index)
    fc3.set_ic_cluster_counts(counts, a_dim=2)

@torch.no_grad()
def llama_reorder_quantize(
    lm,
    args,
    dataloader,
    n_clusters={"R1": 4, "R2": 4, "R3": 4, "R4": 32, "R5": 4},
    reorder="12345",
):
    print("Starting ...")

    model = lm.model
    dev = lm.device

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None, "position_ids": None, "output_attenstions": False, "use_cache": False}

    # only catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            # print(inp.size())
            # raise
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])

    for batch in dataloader:
        if cache["i"] >= args.nsamples:
            break
        try:
            # print(cache["attention_mask"])
            model(inputs_embeds=batch[0].to(dev))

        except ValueError:
            pass
    # print("inps.sum()",inps.sum())
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    enable_R1 = True if "1" in reorder else False
    enable_R2 = True if "2" in reorder else False
    enable_R3 = True if "3" in reorder else False
    enable_R4 = True if "4" in reorder else False
    enable_R5 = True if "5" in reorder else False
    print(f"Ready for reorder {reorder}.")

    for i in range(len(layers)):
        print(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(dev)
        qlayer = QuantLlamaDecoderLayer(lm.model.config, layer, args).to(dev)

        handlers = []
        for name, module in layer.named_modules():
            # print(name,type(module))
            if(
                enable_R1
                and isinstance(module, LlamaRMSNorm)
                and "input_layernorm" in name
            ):
                module.name = name
                handler = module.register_forward_hook(layer_omax_hook)
                handlers.append(handler)
            if (
                enable_R2
            ):
                raise NotImplementedError
            if (
                enable_R3
            ):
                raise NotImplementedError
            if (
                enable_R4
                and isinstance(module, LlamaRMSNorm)
                and "post_attention_layernorm" in name
            ):
                module.name = name
                handler = module.register_forward_hook(layer_omax_hook)
                handlers.append(handler)
            if enable_R5 and isinstance(module, nn.Linear) and "down_proj" in name:
                module.name = name
                handler = module.register_forward_hook(layer_i0max_hook)
                handlers.append(handler)
            
        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask.to(dev) if attention_mask is not None else None, 
                position_ids=position_ids.to(dev) if position_ids is not None else None)[0]

        for handler in handlers:
            handler.remove()

        if enable_R1:
            #print(oc_maxmin_dict)
            feature_max, feature_min = oc_maxmin_dict[f"input_layernorm"]

            R1_index, counts = tensor_calc_reorder_index(
                feature_max, feature_min, n_clusters["R1"]
            )

            R1_reorder(
                qlayer.input_layernorm,
                qlayer.self_attn.q_proj,
                qlayer.self_attn.k_proj,
                qlayer.self_attn.v_proj,
                R1_index,
                counts,
            )
            

        if enable_R2:
            raise NotImplementedError
        
        if enable_R3:
            raise NotImplementedError

        if enable_R4:
            feature_max, feature_min = oc_maxmin_dict[f"post_attention_layernorm"]

            R4_index, counts = tensor_calc_reorder_index(
                feature_max, feature_min, n_clusters["R4"]
            )

            R4_reorder(
                qlayer.post_attention_layernorm,
                qlayer.mlp.up_proj,
                qlayer.mlp.gate_proj,
                R4_index,
                counts,
            )

        if enable_R5:
            feature_max, feature_min = ic_maxmin_dict[f"mlp.down_proj"]
            R5_index, counts = tensor_calc_reorder_index(
                feature_max, feature_min, n_clusters["R5"]
            )

            R5_reorder(
                qlayer.mlp.up_proj,
                qlayer.mlp.gate_proj,
                qlayer.mlp.down_proj,
                R5_index,
                counts,
            )

        outs = quant_layer_llama(qlayer, args, outs, inps, attention_mask, position_ids, dev)

        ic_maxmin_dict.clear()
        oc_maxmin_dict.clear()
        layers[i] = qlayer.to("cpu")
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        print(
            lm._device,
            "memory_allocated",
            i,
            torch.cuda.memory_allocated(lm._device) / 1024 / 1024,
            "max memory_allocated",
            torch.cuda.max_memory_allocated(lm._device) / 1024**2,
        )

    del inps, outs
    model.config.use_cache = use_cache
    
    for i, qlayer in enumerate(layers):
        torch.save(qlayer.state_dict(), f"./output/qlayer_{i}.pth")
    return model