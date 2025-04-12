# This file modifies https://github.com/hustvl/Vim/blob/main/vim/models_mamba.py

import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional
from utilities.diffusion_util import (
    timestep_embedding,
    linear,
)
from timm.models.layers import trunc_normal_, lecun_normal_

from timm.models.layers import DropPath

import math

from mamba_ssm.modules.mamba_simple import Mamba

from src.utilities.rope import * 
from src.utilities.tokenization import UnFlexiPatchEmbed, FlexiPatchEmbed, FlexiPosEmbed, resample_patch_embed, vanilla_resample_patch_embed
from torch.cuda.amp import autocast
import random

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None



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
from inspect import isfunction
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

import einops
def exists(val):
    return val is not None

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
        x = x[:, 256, :]
        x = x.unsqueeze(1)
        B, L, C = x.shape
        
        q = self.to_q(x)
        # text = default(text, x)
        k = self.to_k(text)
        v = self.to_v(text)

        q, k, v = map(
            lambda t: rearrange(t, "B L (H D) -> B H L D", H=self.heads), (q, k, v)
        )  # B H L D
        if ATTENTION_MODE == "flash":
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, "B H L D -> B L (H D)")
        elif ATTENTION_MODE == "xformers":
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, "B H L D -> B L (H D)", H=self.heads)
        elif ATTENTION_MODE == "math":
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented
        return self.to_out(x)
def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,
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
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

        self.msa = CrossAttention(
            query_dim=dim, context_dim=dim, heads=8, dim_head=64, dropout=0.0
        )
        self.norm_msa = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
    def forward(
        self, hidden_states: Tensor, c = None, text = None,residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )    
        shift_mba, scale_mba, gate_mba, shift_msa, scale_msa, gate_msa = (
            self.adaLN_modulation(c).chunk(6, dim=2)
        )
        x = hidden_states + gate_mba * self.mixer(hidden_states, inference_params=inference_params)
        x = x + gate_msa * self.msa(
            modulate(self.norm_msa(x), shift_msa, scale_msa),
            text=text,
            mask=None,  #
        )


        # hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return x, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    if_bimamba=False,
    bimamba_type="none",
    if_devide_out=False,
    init_layer_scale=None,
):
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
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


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class AudioMamba(nn.Module):
    def __init__(self, 
                 spectrogram_size=(64, 1024),
                 patch_size=(8, 16),
                 strides=(8, 16),
                 depth=24, 
                 embed_dim=512,
                 channels=1,
                 num_classes=527,
                 ssm_cfg=None, 
                 drop_rate=0.,
                 drop_path_rate=0,
                 norm_epsilon: float = 1e-5, 
                 rms_norm: bool = True,
                 initializer_cfg=None,
                 fused_add_norm=True, 
                 residual_in_fp32=True, 
                 device=None,
                 dtype=None,
                 ft_seq_len=None,
                 abs_pos_patch_grid_size=None,
                 pt_hw_seq_len=None,
                 final_pool_type='mean',
                 if_abs_pos_embed=True,
                 if_rope=False,
                 if_rope_residual=False,
                 if_cls_token=True,
                 bilinear_rope=False,
                 flip_img_sequences_ratio=-1.,
                 if_bidirectional=False,
                 if_bimamba=False,
                 bimamba_type="v1",
                 if_devide_out=True,
                 init_layer_scale=None,
                 use_double_cls_token=False,
                 use_middle_cls_token=True,
                 transpose_token_sequence=False,
                 use_end_cls_token=False,
                 use_PI_for_patch_embed=True,
                 flexible_patch_sizes=None,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs) 
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.flip_img_sequences_ratio = flip_img_sequences_ratio
        self.if_cls_token = if_cls_token
        self.use_double_cls_token = use_double_cls_token
        self.use_middle_cls_token = use_middle_cls_token
        self.use_end_cls_token = use_end_cls_token
        self.num_tokens = 0
        self.spectrogram_size = spectrogram_size
        self.patch_size = to_2tuple(patch_size)
        self.strides = strides
        self.channels = channels
        self.embed_dim = embed_dim
        self.pt_hw_seq_len = pt_hw_seq_len
        self.ft_seq_len = ft_seq_len
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim
        self.transpose_token_sequence = transpose_token_sequence

        self.patch_grid_size = FlexiPosEmbed.get_shape(fstride = 16, tstride = 16, patch_size = patch_size, input_fdim = 128, input_tdim = 1024)
        
        self.num_patches = self.patch_grid_size[0] * self.patch_grid_size[1]
        self.time_embed = nn.Sequential(
            linear(128, 512),
            nn.SiLU(),
            linear(512, 512),
        )
        # TODO: Add a checker that looks at the patch size and spectrogram size to make sure they are compatible
        self.unpatch_embed = UnFlexiPatchEmbed()
        if if_cls_token:
            if use_double_cls_token:
                self.cls_token_head = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.cls_token_tail = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.num_tokens = 2
            else:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.num_tokens = 1

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()


        # TODO: release this comment
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=if_bimamba,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    if_devide_out=if_devide_out,
                    init_layer_scale=init_layer_scale,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        
        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        # self.pre_logits = nn.Identity()

        # original init
        self.head.apply(segm_init_weights)
        if if_cls_token:
            if use_double_cls_token:
                trunc_normal_(self.cls_token_head, std=.02)
                trunc_normal_(self.cls_token_tail, std=.02)
            else:
                trunc_normal_(self.cls_token, std=.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

        if if_rope and bilinear_rope:
            self.init_rope()

        proj_load = None
        pos_embed_load = None
        pos_grid_size_load = (-1, -1)


        self.patch_embed = FlexiPatchEmbed(
            patch_size=patch_size,
            strides=strides,
            in_chans=channels,
            embed_dim=embed_dim,
            proj_load=proj_load,
            resize_func=resample_patch_embed if use_PI_for_patch_embed else vanilla_resample_patch_embed,
            precompute_for=flexible_patch_sizes,
        )

        if if_abs_pos_embed:
            self.pos_embed = FlexiPosEmbed(
                input_size=spectrogram_size,
                patch_size=patch_size,
                strides=strides,
                pos_grid_size=self.patch_grid_size if abs_pos_patch_grid_size is None else abs_pos_patch_grid_size,
                embed_dim=embed_dim,
                n_prefix_tokens=self.num_tokens,
                pos_embed_load=pos_embed_load,
                pos_grid_size_load=pos_grid_size_load,
            )
            self.pos_drop = nn.Dropout(p=drop_rate)

    def interp_rope(self, weights, load_grid_size):
        weights = weights.view(1, load_grid_size[0], load_grid_size[1], -1).permute(0, 3, 1, 2)
        target_grid_size = self.patch_grid_size
        weights = nn.functional.interpolate(weights, size=target_grid_size, mode='bilinear')
        weights = weights.permute(0, 2, 3, 1).view(target_grid_size[0] * target_grid_size[1], -1)
        return weights

    def init_rope(self):
        half_head_dim = self.embed_dim // 2
            
        if self.pt_hw_seq_len is None:
            self.pt_hw_seq_len = self.patch_grid_size
        
        self.rope = VisionRotaryEmbedding(
            dim=half_head_dim,
            pt_seq_len=self.pt_hw_seq_len,
            ft_seq_len=self.patch_grid_size if self.ft_seq_len is None else self.ft_seq_len,
        )


    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}

    def forward_features(self, x,t,y,inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False, patch_size=None, strides=None):
        # x = x.unsqueeze(1) # B x C=1 x T x F
        x = x.transpose(2, 3) # B x C=1 x F x T
        # print(t.size())
        t_emb = timestep_embedding(t, 128, repeat_only=False)
        # print(t_emb.size())
        emb = self.time_embed(t_emb)
        # emb = torch.cat([emb.unsqueeze(1), y], dim=1)
        emb = emb.unsqueeze(1) + y
        c = emb
        text = y
        # print(emb.size())
        B, C, F, T = x.shape

        x = self.patch_embed(x, patch_size=patch_size, strides=strides)
        B, N, _ = x.shape

        if self.if_cls_token:
            if self.use_double_cls_token:
                cls_token_head = self.cls_token_head.expand(B, -1, -1)
                cls_token_tail = self.cls_token_tail.expand(B, -1, -1)
                token_position = [0, N + 1]
                x = torch.cat((cls_token_head, x, cls_token_tail), dim=1)
            else:
                cls_token = emb.expand(B, -1, -1)
                if if_random_cls_token_position:
                    token_position = random.randint(0, N)
                elif self.use_middle_cls_token:
                    token_position = N // 2
                elif self.use_end_cls_token:
                    token_position = N
                else:
                    token_position = 0
                x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
            N = x.shape[1]
        else:
            token_position = None

        if self.if_abs_pos_embed:
            x = self.pos_embed(x, token_position=token_position, patch_size=patch_size, strides=strides)
            x = self.pos_drop(x)

        if self.transpose_token_sequence:
            if self.if_cls_token:
                if self.use_double_cls_token:
                    head_t, tail_t = x[:, 0, :].unsqueeze(1), x[:, -1, :].unsqueeze(1)
                    x = x[:, 1:-1, :]
                else:
                    t = x[:, token_position, :].unsqueeze(1)
                    x = torch.cat((x[:, :token_position, :], x[:, token_position + 1:, :]), dim=1)
            
            # reshape x to be B x F x T x D
            _F, _T = F // self.patch_size[0], T // self.patch_size[1]
            x = x.reshape(B, _F, _T, -1)
            x = x.transpose(1, 2)
            x = x.reshape(B, _T * _F, -1)

            if self.if_cls_token:
                if self.use_double_cls_token:
                    x = torch.cat((head_t, x, tail_t), dim=1)
                else:
                    x = torch.cat((x[:, :token_position, :], t, x[:, token_position:, :]), dim=1)

        if if_random_token_rank:

            # 生成随机 shuffle 索引
            shuffle_indices = torch.randperm(N)

            if isinstance(token_position, list):
                print("original value: ", x[0, token_position[0], 0], x[0, token_position[1], 0])
            else:
                print("original value: ", x[0, token_position, 0])
            print("original token_position: ", token_position)

            # 执行 shuffle
            x = x[:, shuffle_indices, :]

            if isinstance(token_position, list):
                # 找到 cls token 在 shuffle 之后的新位置
                new_token_position = [torch.where(shuffle_indices == token_position[i])[0].item() for i in range(len(token_position))]
                token_position = new_token_position
            else:
                # 找到 cls token 在 shuffle 之后的新位置
                token_position = torch.where(shuffle_indices == token_position)[0].item()

            if isinstance(token_position, list):
                print("new value: ", x[0, token_position[0], 0], x[0, token_position[1], 0])
            else:
                print("new value: ", x[0, token_position, 0])
            print("new token_position: ", token_position)


        if_flip_img_sequences = False
        if self.flip_img_sequences_ratio > 0 and (self.flip_img_sequences_ratio - random.random()) > 1e-5:
            x = x.flip([1])
            if_flip_img_sequences = True

        # mamba impl
        residual = None
        hidden_states = x
        if not self.if_bidirectional:
            for layer in self.layers:

                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                # rope about
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                hidden_states, residual = layer(
                    hidden_states,c,text, residual, inference_params=inference_params
                )
        else:
            # get two layers in a single for-loop
            for i in range(len(self.layers) // 2):
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                hidden_states_f, residual_f = self.layers[i * 2](
                    hidden_states, c,text,residual, inference_params=inference_params
                )
                hidden_states_b, residual_b = self.layers[i * 2 + 1](
                    hidden_states.flip([1]),c,text, None if residual == None else residual.flip([1]), inference_params=inference_params
                )
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        # return only cls token if it exists
        if self.if_cls_token:
            if self.use_double_cls_token:
                return (hidden_states[:, token_position[0], :] + hidden_states[:, token_position[1], :]) / 2
            else:
                X= torch.cat((hidden_states[:, :token_position, :], hidden_states[:, token_position+1:, :]), dim=1)
                return self.unpatch_embed(X)


    # @autocast() # disabled because accelerate training configs already incorporate autocast
    def forward(self, x,t,y,return_features=False, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False, patch_size=None, strides=None): # NOTE: For now, these are all being used as default. Later, these could be set through the args param
        x = self.forward_features(x,t,y,inference_params, if_random_cls_token_position=if_random_cls_token_position, if_random_token_rank=if_random_token_rank, patch_size=patch_size, strides=strides)
        return x


