import os
import random as random
from typing import Optional, List, Tuple
import torch.nn.functional as F
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch

# from audiosliders.aduio.tools import wav_to_fbank
from pydub import AudioSegment

from inspect import isfunction

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


from transformers import CLIPTextModel, CLIPTokenizer, RobertaTokenizerFast, ClapTextModelWithProjection
from diffusers import UNet2DConditionModel, SchedulerMixin
from diffusers.image_processor import VaeImageProcessor


from tqdm import tqdm

UNET_IN_CHANNELS = 4  # Stable Diffusion の in_channels は 4 で固定。XLも同じ。
VAE_SCALE_FACTOR = 8  # 2 ** (len(vae.config.block_out_channels) - 1) = 8

UNET_ATTENTION_TIME_EMBED_DIM = 256  # XL
TEXT_ENCODER_2_PROJECTION_DIM = 1280
UNET_PROJECTION_CLASS_EMBEDDING_INPUT_DIM = 2816


def get_random_noise(
		batch_size: int, height: int, width: int, generator: torch.Generator = None
) -> torch.Tensor:
	return torch.randn(
		(
			batch_size,
			UNET_IN_CHANNELS,
			height // VAE_SCALE_FACTOR,  # 縦と横これであってるのかわからないけど、どっちにしろ大きな問題は発生しないのでこれでいいや
			width // VAE_SCALE_FACTOR,
		),
		generator=generator,
		device="cpu",
	)


# https://www.crosslabs.org/blog/diffusion-with-offset-noise
def apply_noise_offset(latents: torch.FloatTensor, noise_offset: float):
	latents = latents + noise_offset * torch.randn(
		(latents.shape[0], latents.shape[1], 1, 1), device=latents.device
	)
	return latents


def get_initial_latents(
		scheduler: SchedulerMixin,
		n_imgs: int,
		height: int,
		width: int,
		n_prompts: int,
		generator=None,
) -> torch.Tensor:
	noise = get_random_noise(n_imgs, height, width, generator=generator).repeat(
		n_prompts, 1, 1, 1
	)
	
	latents = noise * scheduler.init_noise_sigma
	
	return latents


def text_tokenize(
		tokenizer: CLIPTokenizer,  # 普通ならひとつ、XLならふたつ！
		prompts: List[str],
):
	return tokenizer(
		prompts,
		padding="max_length",
		max_length=tokenizer.model_max_length,
		truncation=True,
		return_tensors="pt",
	).input_ids


def text_encode(text_encoder: CLIPTextModel, tokens):
	return text_encoder(tokens.to(text_encoder.device))[0]


def encode_prompts(
		tokenizer: RobertaTokenizerFast,
		text_encoder: ClapTextModelWithProjection,
		prompts: List[str],
		device="cuda:0",
		num_waveforms_per_prompt=1,
		weight_dtype=torch.float16,
		negative_prompt=None,
):
	if prompts is not None and isinstance(prompts, str):
		batch_size = 1
	elif prompts is not None and isinstance(prompts, tuple):
		batch_size = len(prompts)
	
	text_inputs = tokenizer(
		prompts,
		padding="max_length",
		max_length=8,
		truncation=True,
		return_tensors="pt",
	)
	text_input_ids = text_inputs.input_ids
	attention_mask = text_inputs.attention_mask
	prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device),
								 attention_mask=attention_mask.to(device)).text_embeds
	prompt_embeds = F.normalize(prompt_embeds, dim=-1)
	prompt_embeds = prompt_embeds.repeat(1, num_waveforms_per_prompt)
	
	(
		bs_embed,
		seq_len,
	) = prompt_embeds.shape
	prompt_embeds = prompt_embeds.view(bs_embed * num_waveforms_per_prompt, seq_len)
	
	return prompt_embeds


def concat_embeddings(
		unconditional: torch.FloatTensor,
		conditional: torch.FloatTensor,
		batch_size: int,
):
	return torch.cat([unconditional, conditional]).repeat_interleave(batch_size, dim=0)


def predict_noise(
		mamba: UNet2DConditionModel,
		timestep: int,  # 現在のタイムステップ
		latents: torch.FloatTensor,  # uncond な text embed と cond な text embed を結合したもの
		class_labels: torch.FloatTensor,
		guidance_scale=7.5,
) -> torch.FloatTensor:
	latent_model_input = torch.cat([latents] * 2)
	noise_pred = mamba(
		latent_model_input,
		timestep,
		class_labels.unsqueeze(1)
	)
	
	noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
	guided_target = noise_pred_uncond + guidance_scale * (
			noise_pred_text - noise_pred_uncond
	)
	
	return guided_target


def predict_noise_without_lora(
		mamba: UNet2DConditionModel,
		timestep: int,  # 現在のタイムステップ
		latents: torch.FloatTensor,  # uncond な text embed と cond な text embed を結合したもの
		class_labels: torch.FloatTensor,
		guidance_scale=7.5,
) -> torch.FloatTensor:
	noise_pred = mamba(
		latents,
		timestep,
		class_labels.unsqueeze(1)
	)
	
	noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
	guided_target = noise_pred_uncond + guidance_scale * (
			noise_pred_text - noise_pred_uncond
	)
	
	return guided_target


# ref: https://github.com/huggingface/diffusers/blob/0bab447670f47c28df60fbd2f6a0f833f75a16f5/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L746
@torch.no_grad()
def diffusion(
		unet: UNet2DConditionModel,
		scheduler: SchedulerMixin,
		latents: torch.FloatTensor,  # ただのノイズだけのlatents
		text_embeddings: torch.FloatTensor,
		total_timesteps: int = 1000,
		start_timesteps=0,
		**kwargs,
):
	# latents_steps = []
	
	for timestep in tqdm(scheduler.timesteps[start_timesteps:total_timesteps]):
		noise_pred = predict_noise(
			unet, scheduler, timestep, latents, text_embeddings, **kwargs
		)
		
		# compute the previous noisy sample x_t -> x_t-1
		latents = scheduler.step(noise_pred, timestep, latents).prev_sample
	
	# return latents_steps
	return latents




def rescale_noise_cfg(
		noise_cfg: torch.FloatTensor, noise_pred_text, guidance_rescale=0.0
):
	"""
	Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
	Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
	"""
	std_text = noise_pred_text.std(
		dim=list(range(1, noise_pred_text.ndim)), keepdim=True
	)
	std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
	# rescale the results from guidance (fixes overexposure)
	noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
	# mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
	noise_cfg = (
			guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
	)
	
	return noise_cfg


def predict_noise_xl(
		unet: UNet2DConditionModel,
		scheduler: SchedulerMixin,
		timestep: int,  # 現在のタイムステップ
		latents: torch.FloatTensor,
		text_embeddings: torch.FloatTensor,  # uncond な text embed と cond な text embed を結合したもの
		add_text_embeddings: torch.FloatTensor,  # pooled なやつ
		add_time_ids: torch.FloatTensor,
		guidance_scale=7.5,
		guidance_rescale=0.7,
) -> torch.FloatTensor:
	# expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
	latent_model_input = torch.cat([latents] * 2)
	
	latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)
	
	added_cond_kwargs = {
		"text_embeds": add_text_embeddings,
		"time_ids": add_time_ids,
	}
	
	# predict the noise residual
	noise_pred = unet(
		latent_model_input,
		timestep,
		encoder_hidden_states=text_embeddings,
		added_cond_kwargs=added_cond_kwargs,
	).sample
	
	# perform guidance
	noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
	guided_target = noise_pred_uncond + guidance_scale * (
			noise_pred_text - noise_pred_uncond
	)
	
	# https://github.com/huggingface/diffusers/blob/7a91ea6c2b53f94da930a61ed571364022b21044/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L775
	noise_pred = rescale_noise_cfg(
		noise_pred, noise_pred_text, guidance_rescale=guidance_rescale
	)
	
	return guided_target


@torch.no_grad()
def diffusion_xl(
		unet: UNet2DConditionModel,
		scheduler: SchedulerMixin,
		latents: torch.FloatTensor,  # ただのノイズだけのlatents
		text_embeddings: Tuple[torch.FloatTensor, torch.FloatTensor],
		add_text_embeddings: torch.FloatTensor,  # pooled なやつ
		add_time_ids: torch.FloatTensor,
		guidance_scale: float = 1.0,
		total_timesteps: int = 1000,
		start_timesteps=0,
):
	# latents_steps = []
	
	for timestep in tqdm(scheduler.timesteps[start_timesteps:total_timesteps]):
		noise_pred = predict_noise_xl(
			unet,
			scheduler,
			timestep,
			latents,
			text_embeddings,
			add_text_embeddings,
			add_time_ids,
			guidance_scale=guidance_scale,
			guidance_rescale=0.7,
		)
		
		# compute the previous noisy sample x_t -> x_t-1
		latents = scheduler.step(noise_pred, timestep, latents).prev_sample
	
	# return latents_steps
	return latents


# for XL
def get_add_time_ids(
		height: int,
		width: int,
		dynamic_crops: bool = False,
		dtype: torch.dtype = torch.float32,
):
	if dynamic_crops:
		# random float scale between 1 and 3
		random_scale = torch.rand(1).item() * 2 + 1
		original_size = (int(height * random_scale), int(width * random_scale))
		# random position
		crops_coords_top_left = (
			torch.randint(0, original_size[0] - height, (1,)).item(),
			torch.randint(0, original_size[1] - width, (1,)).item(),
		)
		target_size = (height, width)
	else:
		original_size = (height, width)
		crops_coords_top_left = (0, 0)
		target_size = (height, width)
	
	# this is expected as 6
	add_time_ids = list(original_size + crops_coords_top_left + target_size)
	
	# this is expected as 2816
	passed_add_embed_dim = (
			UNET_ATTENTION_TIME_EMBED_DIM * len(add_time_ids)  # 256 * 6
			+ TEXT_ENCODER_2_PROJECTION_DIM  # + 1280
	)
	if passed_add_embed_dim != UNET_PROJECTION_CLASS_EMBEDDING_INPUT_DIM:
		raise ValueError(
			f"Model expects an added time embedding vector of length {UNET_PROJECTION_CLASS_EMBEDDING_INPUT_DIM}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
		)
	
	add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
	return add_time_ids


def get_optimizer(name: str):
	name = name.lower()
	
	if name.startswith("dadapt"):
		import dadaptation
		
		if name == "dadaptadam":
			return dadaptation.DAdaptAdam
		elif name == "dadaptlion":
			return dadaptation.DAdaptLion
		else:
			raise ValueError("DAdapt optimizer must be dadaptadam or dadaptlion")
	
	elif name.endswith("8bit"):  # 検証してない
		import bitsandbytes as bnb
		
		if name == "adam8bit":
			return bnb.optim.Adam8bit
		elif name == "lion8bit":
			return bnb.optim.Lion8bit
		else:
			raise ValueError("8bit optimizer must be adam8bit or lion8bit")
	
	else:
		if name == "adam":
			return torch.optim.Adam
		elif name == "adamw":
			return torch.optim.AdamW
		elif name == "lion":
			from lion_pytorch import Lion
			
			return Lion
		elif name == "prodigy":
			import prodigyopt
			
			return prodigyopt.Prodigy
		else:
			raise ValueError("Optimizer must be adam, adamw, lion or Prodigy")


def get_lr_scheduler(
		name: Optional[str],
		optimizer: torch.optim.Optimizer,
		max_iterations: Optional[int],
		lr_min: Optional[float],
		**kwargs,
):
	if name == "cosine":
		return torch.optim.lr_scheduler.CosineAnnealingLR(
			optimizer, T_max=max_iterations, eta_min=lr_min, **kwargs
		)
	elif name == "cosine_with_restarts":
		return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
			optimizer, T_0=max_iterations // 10, T_mult=2, eta_min=lr_min, **kwargs
		)
	elif name == "step":
		return torch.optim.lr_scheduler.StepLR(
			optimizer, step_size=max_iterations // 100, gamma=0.999, **kwargs
		)
	elif name == "constant":
		return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, **kwargs)
	elif name == "linear":
		return torch.optim.lr_scheduler.LinearLR(
			optimizer, factor=0.5, total_iters=max_iterations // 100, **kwargs
		)
	else:
		raise ValueError(
			"Scheduler must be cosine, cosine_with_restarts, step, linear or constant"
		)


def get_random_resolution_in_bucket(bucket_resolution: int = 512) -> Tuple[int, int]:
	max_resolution = bucket_resolution
	min_resolution = bucket_resolution // 2
	
	step = 64
	
	min_step = min_resolution // step
	max_step = max_resolution // step
	
	height = torch.randint(min_step, max_step, (1,)).item() * step
	width = torch.randint(min_step, max_step, (1,)).item() * step
	
	return height, width
