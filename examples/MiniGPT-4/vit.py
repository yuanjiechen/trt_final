import numpy as np
import tensorrt as trt

from tensorrt_llm.module import Module, ModuleList
from tensorrt_llm.layers import Linear, Conv2d, LayerNorm, MLP
from tensorrt_llm.functional import Tensor, gelu, concat, matmul, split, softmax
from tensorrt_llm.parameter import Parameter

class Attention(Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = Linear(dim, all_head_dim * 3, bias=False, dtype=dtype)
        if qkv_bias:
            self.q_bias = Parameter(shape=[all_head_dim], dtype=dtype)
            self.k_bias = Parameter(shape=[all_head_dim], dtype=dtype) # Notice: always zeros when load 
            self.v_bias = Parameter(shape=[all_head_dim], dtype=dtype)
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        self.proj = Linear(all_head_dim, dim, dtype=dtype)

    def forward(self, x, rel_pos_bias=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = concat([self.q_bias, self.k_bias, self.v_bias])

        qkv = matmul(x, self.qkv.weight.value, transb=True)

        if qkv_bias is not None:
            qkv = qkv + qkv_bias
        
        remain_shape = int(trt.volume(qkv.shape) // B // N // 3 // self.num_heads)
        qkv = qkv.view([B, N, 3, self.num_heads, remain_shape]).permute([2, 0, 3, 1, 4])
        q, k, v = split(qkv, 3, dim=0) # 1, 16, 257, 88 (B, H, N, F)
        q = q * self.scale

        attn = matmul(q, k.transpose(2, 3)) # B, H, N, F @ B, H, F, N -> B, H, N, N
        attn = softmax(attn, dim=3)

        x = matmul(attn, v).transpose(1, 2).view([B, N, self.num_heads * remain_shape]) # B, N, F(C)
        x = self.proj(x)

        return x


class Block(Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer="gelu", norm_layer=LayerNorm, dtype=None):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, dtype=dtype)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim, act_layer, dtype=dtype)

    def forward(self, x, rel_pos_bias=None):
        x = x + self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias)
        x = x + self.mlp(self.norm2(x))

        return x

class PatchEmbed(Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

class VisionTransformer(Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, norm_layer=LayerNorm, 
                 use_abs_pos_emb=True, dtype='float16'):
        super().__init__()
        self.image_size = img_size
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = Parameter(shape=[1, 1, embed_dim])
        if use_abs_pos_emb:
            self.pos_embed = Parameter(shape=(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        
        self.blocks = ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  norm_layer=norm_layer, dtype=dtype)
            for _ in range(depth)])

        self.dtype = dtype

    def forward(self, x:Tensor):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.shape
        if batch_size > 1:
            cls_tokens = concat([self.cls_token for _ in range(batch_size)], dim=0)
        else:
            cls_tokens = self.cls_token

        x = concat([cls_tokens, x], dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        
        rel_pos_bias = None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias)
        
        x.mark_output('vit_outputs', self.dtype)
        return x