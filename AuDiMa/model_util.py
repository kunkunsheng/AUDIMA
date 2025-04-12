from typing import Literal, Union, Optional
import os
import torch
from transformers import ClapTextModel, CLIPTokenizer, CLIPTextModelWithProjection

AVAILABLE_SCHEDULERS = Literal["ddim", "ddpm", "lms", "euler_a"]

from diffusers import (

	SchedulerMixin,
	AudioLDMPipeline
)
from diffusers.schedulers import (
	DDIMScheduler,
	DDPMScheduler,
	LMSDiscreteScheduler,
	EulerAncestralDiscreteScheduler,
)


def load_diffusers_model(
		pretrained_model_name_or_path: str,
		weight_dtype: torch.dtype = torch.float32,
		device: str = "cuda:0",
):
	# VAE はいらない
	pipe = AudioLDMPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtyp=weight_dtype).to(device)
	tokenizer = pipe.tokenizer
	pipe.unet.config.class_embed_type= None
	unet = pipe.unet
	noise_scheduler = pipe.scheduler
	text_encoder = pipe.text_encoder
	
	return tokenizer, text_encoder, unet, noise_scheduler, pipe


def load_models(
		pretrained_model_name_or_path: str,
		weight_dtype: torch.dtype = torch.bfloat16,
		device: str = "cuda:0",
):
	tokenizer, text_encoder, unet, noise_scheduler, pipe = load_diffusers_model(
		pretrained_model_name_or_path, weight_dtype=weight_dtype, device=device)
	
	return tokenizer, text_encoder, unet, noise_scheduler, pipe


def create_noise_scheduler(
		scheduler_name: AVAILABLE_SCHEDULERS = "ddpm",
		prediction_type: Literal["epsilon", "v_prediction"] = "epsilon",
) -> SchedulerMixin:
	# 正直、どれがいいのかわからない。元の実装だとDDIMとDDPMとLMSを選べたのだけど、どれがいいのかわからぬ。
	
	name = scheduler_name.lower().replace(" ", "_")
	if name == "ddim":
		# https://huggingface.co/docs/diffusers/v0.17.1/en/api/schedulers/ddim
		scheduler = DDIMScheduler(
			beta_start=0.0015,
			beta_end=0.0195,
			beta_schedule="scaled_linear",
			num_train_timesteps=1000,
			clip_sample=False,
			prediction_type=prediction_type,  # これでいいの？
		)
	elif name == "ddpm":
		# https://huggingface.co/docs/diffusers/v0.17.1/en/api/schedulers/ddpm
		scheduler = DDPMScheduler(
			beta_start=0.00085,
			beta_end=0.012,
			beta_schedule="scaled_linear",
			num_train_timesteps=1000,
			clip_sample=False,
			prediction_type=prediction_type,
		)
	elif name == "lms":
		# https://huggingface.co/docs/diffusers/v0.17.1/en/api/schedulers/lms_discrete
		scheduler = LMSDiscreteScheduler(
			beta_start=0.00085,
			beta_end=0.012,
			beta_schedule="scaled_linear",
			num_train_timesteps=1000,
			prediction_type=prediction_type,
		)
	elif name == "euler_a":
		# https://huggingface.co/docs/diffusers/v0.17.1/en/api/schedulers/euler_ancestral
		scheduler = EulerAncestralDiscreteScheduler(
			beta_start=0.00085,
			beta_end=0.012,
			beta_schedule="scaled_linear",
			num_train_timesteps=1000,
			prediction_type=prediction_type,
		)
	else:
		raise ValueError(f"Unknown scheduler name: {name}")
	
	return scheduler
