# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import copy
import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp
import math
from functools import partial
from einops import rearrange, repeat
from inspect import isfunction
from torch import Tensor
from typing import Optional
import einops
import torch
from einops import rearrange, repeat
from torch import nn, Tensor
from zeta.nn import SSM
from einops.layers.torch import Reduce
import torch
import torch.nn as nn
import numpy as np
from functools import partial

from timm.models.vision_transformer import Mlp
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn


if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
    ATTENTION_MODE = "flash"
else:
    try:
        import xformers
        import xformers.ops

        ATTENTION_MODE = "xformers"
    except:
        ATTENTION_MODE = "math"
print(f"attention mode is {ATTENTION_MODE}")


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


class PatchEmbed_Music(PatchEmbed):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = super().forward(x)
        # (b t) n c
        x = rearrange(x, "(b t) n c -> b (t n) c", t=T)
        return x


def output_head(dim: int, num_classes: int):

    return nn.Sequential(
        Reduce("b s d -> b d", "mean"),
        nn.LayerNorm(dim),
        nn.Linear(dim, num_classes),
    )


class Bi_MambaBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dt_rank: int,
        dim_inner: int,
        d_state: int,
    ):
        super().__init__()
        self.dim = dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state

        self.forward_conv1d = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=1
        )
        self.backward_conv1d = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=1
        )
        self.norm = nn.LayerNorm(dim)
        self.silu = nn.SiLU()
        self.ssm = SSM(dim, dt_rank, dim_inner, d_state)

        # Linear layer for z and x
        self.proj = nn.Linear(dim, dim)

        # Softplus
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor):
        b, s, d = x.shape

        # Skip connection
        skip = x

        # Normalization
        x = self.norm(x)

        # Split x into x1 and x2 with linears
        z1 = self.proj(x)
        x = self.proj(x)

        # forward con1d
        x1 = self.process_direction(
            x,
            self.forward_conv1d,
            self.ssm,
        )

        # backward conv1d
        x2 = self.process_direction(
            x,
            self.backward_conv1d,
            self.ssm,
        )

        # Activation
        z = self.silu(z1)

        # Matmul
        x1 *= z
        x2 *= z

        # Residual connection
        return x1 + x2 + skip

    def process_direction(
        self,
        x: Tensor,
        conv1d: nn.Conv1d,
        ssm: SSM,
    ):
        x = rearrange(x, "b s d -> b d s")
        x = self.softplus(conv1d(x))
        print(f"Conv1d: {x}")
        x = rearrange(x, "b d s -> b s d")
        x = ssm(x)
        return x





def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def forward(self, x, text, mask=None):
        B, L, C = x.shape
        q = self.to_q(x.mean(dim=1).unsqueeze(1))
        # text = default(text, x)
        k = self.to_k(text)
        v = self.to_v(text)

        q, k, v = map(
            lambda t: rearrange(t, "B L (H D) -> B L H D", H=self.heads), (q, k, v)
        )  # B H L D
        if ATTENTION_MODE == "flash":
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, "B H L D -> B L (H D)")
        elif ATTENTION_MODE == "xformers":
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, "B L H D -> B L (H D)", H=self.heads)
        elif ATTENTION_MODE == "math":
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented
        return self.to_out(x)


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):

    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class CaptionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(
        self,
        in_channels,
        hidden_size,
        uncond_prob,
        act_layer=nn.GELU(approximate="tanh"),
        token_num=120,
    ):
        super().__init__()
        self.y_proj = Mlp(
            in_features=in_channels,
            hidden_features=hidden_size,
            out_features=hidden_size,
            act_layer=act_layer,
            drop=0,
        )
        self.register_buffer(
            "y_embedding",
            nn.Parameter(torch.randn(token_num, in_channels) / in_channels**0.5),
        )
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0]).cuda() < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None, None], self.y_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        if train:
            assert (
                caption.shape[1:] == self.y_embedding.shape
            ), f"{caption.shape} is not {self.y_embedding.shape}"
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        caption = self.y_proj(caption)
        return caption


#################################################################################
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, dtype, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.dtype = dtype
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, dtype, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=dtype) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(
            t, self.frequency_embedding_size, dtype=self.dtype
        )
        t_emb = self.mlp(t_freq.to(dtype=self.dtype))
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

class T2IFinalLayer(nn.Module):
    """
    The final layer of PixArt.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size ** 0.5)
        self.out_channels = out_channels

    def forward(self, x, t):
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels, cond=False):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        if cond:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
            )

    def forward(self, x, c=None):
        if c is not None:
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
            x = modulate(self.norm_final(x), shift, scale)
            x = self.linear(x)
        else:
            x = self.norm_final(x)
            x = self.linear(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        has_text=False,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
        drop_path=0.0,
        skip=False,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.has_text = has_text
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None

        adaln_num = 3 * 2 if self.has_text else 3
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, adaln_num * dim, bias=True)
        )
        if self.has_text:
            self.msa = CrossAttention(
                query_dim=dim, context_dim=dim, heads=8, dim_head=64, dropout=0.0
            )
            self.norm_msa = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        x: Tensor,
        residual: Optional[Tensor] = None,
        c=None,
        text=None,
        inference_params=None,
        skip=None,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))

        if not self.fused_add_norm:
            if residual is None:
                residual = x
            else:
                residual = residual + self.drop_path(x)

            x = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            )
            if residual is None:
                x, residual = fused_add_norm_fn(
                    x,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                x, residual = fused_add_norm_fn(
                    self.drop_path(x),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )

        if not self.has_text:
            shift_mba, scale_mba, gate_mba = self.adaLN_modulation(c).chunk(3, dim=1)
            x = x + gate_mba.unsqueeze(1) * self.mixer(
                modulate(x, shift_mba, scale_mba),
                inference_params=inference_params,
            )
        else:
            shift_mba, scale_mba, gate_mba, shift_msa, scale_msa, gate_msa = (
                self.adaLN_modulation(c).chunk(6, dim=1)
            )
            x = x + gate_mba.unsqueeze(1) * self.mixer(
                modulate(x, shift_mba, scale_mba),
                inference_params=inference_params,
            )
            x = x + gate_msa.unsqueeze(1) * self.msa(
                modulate(self.norm_msa(x), shift_msa, scale_msa),
                text=text,
                mask=None,  #
            )

        return x, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )


def create_block(
    d_model,
    ssm_cfg=None,
    has_text=False,
    norm_epsilon=1e-5,
    drop_path=0.0,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    skip=False,
    layer_idx=None,
    device=None,
    dtype=None,
    scan_type="none",
    **block_kwargs,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(
        Bi_MambaBlock,
        layer_idx=layer_idx,
        scan_type=scan_type,
        **ssm_cfg,
        **block_kwargs,
        **factory_kwargs,
    )
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        has_text=has_text,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        skip=skip,
    )
    block.layer_idx = layer_idx
    return block


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class Audio_Mamba_Block(nn.Module):
    """
    A DiT-styled Mamba model with ZigZag scan.
    """

    def __init__(
        self,
        in_channels: 8,
        embed_dim: 512,
        depth: 24,
        patch_size: int = 2 ,
        has_text: bool = True,
        num_classes=-1,
        drop_path_rate=0.1,
        d_context: int = 512,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        fused_add_norm=True,
        residual_in_fp32=True,
        initializer_cfg=None,
        scan_type="zigzagN8",
        Music_frames=0,
        tpe=False,  # apply temporal positional encoding for Music-related task
        device="cuda",
        use_pe=2,
        use_jit=True,
        m_init=True,

        dtype=torch.float32,
    ):
        # assert num_classes == -1, "num_classes should be -1"
        # assert n_context_token == 0, "n_context_token should be 0"

        self.factory_kwargs = factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.tpe = tpe

        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.Music_frames = Music_frames
        self.use_pe = use_pe
        num_patches = 4096 // patch_size ** 2
   

        
        self.x_embedder = (
                PatchEmbed(
                    (256,16), patch_size, self.in_channels, self.embed_dim, bias=True
                )
                .to(dtype)
            )
   

        self.t_embedder = (
            TimestepEmbedder(self.embed_dim, dtype=dtype).to(dtype)
        )

        
        num_patches_4pe = num_patches



        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches_4pe, embed_dim, dtype=dtype)
        )
  

        if self.tpe:
            self.temporal_pos_embedding = nn.Parameter(
                torch.zeros(1, Music_frames, embed_dim, dtype=dtype)
            )

        self.n_layer = depth

        self.has_text = has_text
        self.num_classes = num_classes
        # print("has_text", has_text)
        if has_text:
            self.y_embedder = nn.Linear(d_context, embed_dim).to(dtype)
            # print("has_text=", num_classes)
        elif num_classes > 0:
            self.y_embedder = (
                LabelEmbedder(num_classes, hidden_size=embed_dim, dropout_prob=0.0)
                .to(dtype)
            )
            # print("num_classes=", num_classes)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, self.n_layer)
        ]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

        self.extras = 0
        block_kwargs = {"use_jit": use_jit}

        self.blocks = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    has_text=has_text,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    scan_type=scan_type,
                    drop_path=inter_dpr[i],
                    **block_kwargs,
                    **factory_kwargs,
                )
                .to(dtype)
                for i in range(self.n_layer)
            ]
        )
        self.final_layer = (
            T2IFinalLayer(512, patch_size, self.out_channels)
            .to(dtype)
        )

        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        self.initialize_weights()
        self.m_init = m_init
        if m_init:
            self.apply(
                partial(
                    _init_weights,
                    n_layer=depth,
                    **(initializer_cfg if initializer_cfg is not None else {}),
                )
            )
        # print("m_init", m_init)

    def initialize_weights(self):

        if self.use_pe == 1:
            # Initialize (and freeze) pos_embed by sin-cos embedding:
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # if self.has_text:
        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.y_embedding, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        try:
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        except:
            pass

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = 128
        w = 8
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def unpatchify_Music(self, x, Music_frames):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int((x.shape[1] // Music_frames) ** 0.5)
        assert h * w * Music_frames == x.shape[1]

        x = x.reshape(shape=(x.shape[0], Music_frames, h, w, p, p, c))
        x = torch.einsum("nthwpqc->ntchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], Music_frames, c, h * p, h * p))
        return imgs

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def forward(
        self,
        hidden_states,
        t,
        y=None,
    ):
        """
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images),

        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        hidden_states = self.x_embedder(
            hidden_states
        )  # (N, T, D), where T = H * W / patch_size ** 2, if Music_frames>0, T = H * W * Music_frames / patch_size ** 2
        _B, _T, _D = hidden_states.shape

        t = (t * 1000.0).to(hidden_states)
        t = self.t_embedder(t)  # (N, D)
        if self.has_text:
            # y = self.y_embedder(y, self.training)  # (B, N, D)
            # y = self.y_embedder(y)  # (B, N, D)
            c = t + y.mean(dim=1)  # (N, D)
        else:
            c = t

        if self.use_pe == 1 or self.use_pe == 2:
            hidden_states = hidden_states + self.pos_embed
        if self.Music_frames > 0 and self.tpe:
            # temporal pos
            hidden_states = rearrange(
                hidden_states, "b (t l) c -> (b l) t c", t=self.Music_frames
            )
            hidden_states = hidden_states + self.temporal_pos_embedding
            hidden_states = rearrange(hidden_states, "(b l) t c -> b (t l) c", b=_B)

        residual = None
        for layer_idx, block in enumerate(self.blocks):
            if self.use_pe == 3:
                hidden_states = hidden_states + self.pos_embed_list[layer_idx]

            hidden_states, residual = block(
                hidden_states, residual=residual, c=c, text=y
            )  # (N, T, D)

        ##### finished the Mamba blocks, here we apply the last Normalization layer
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            )
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        hidden_states = self.final_layer(hidden_states,t)
        if self.Music_frames > 0:
            hidden_states = self.unpatchify_Music(hidden_states, self.Music_frames)
        else:
            hidden_states = self.unpatchify(hidden_states)

        return hidden_states



#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb



def flops_selective_scan_fn(
    B=1,
    L=256,
    D=768,
    N=16,
    with_D=True,
    with_Z=False,
    with_Group=True,
    with_complex=False,
):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    assert not with_complex
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


def selective_scan_flop_jit(inputs, outputs):
    # print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(
        B=B, L=L, D=D, N=N, with_D=True, with_Z=False, with_Group=True
    )
    return flops


def flops(model, shape=(3, 32, 32)):
    from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

    # shape = self.__input_shape__[1:]
    supported_ops = {
        "aten::silu": None,  # as relu is in _IGNORED_OPS
        "aten::neg": None,  # as relu is in _IGNORED_OPS
        "aten::exp": None,  # as relu is in _IGNORED_OPS
        "aten::flip": None,  # as permute is in _IGNORED_OPS
        # "prim::PythonOp.CrossScan": None,
        # "prim::PythonOp.CrossMerge": None,
        "prim::PythonOp.SelectiveScanFn": partial(
            selective_scan_flop_jit, flops_fn=flops_selective_scan_fn
        ),
    }

    model = copy.deepcopy(model)
    model.cuda().eval()

    input = torch.randn((1, *shape), device=next(model.parameters()).device)
    timestep = torch.rand((1), device=next(model.parameters()).device)
    params = parameter_count(model)[""]
    Gflops, unsupported = flop_count(
        model=model, inputs=(input, timestep), supported_ops=supported_ops
    )

    del model, input
    print(f"params {params} GFLOPs {sum(Gflops.values())}")
    return sum(Gflops.values()) * 1e9
