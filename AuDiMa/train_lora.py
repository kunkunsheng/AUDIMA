import os
from modules.diffusionmodules.ema import LitEma
import yaml
from typing import List, Optional
import argparse
import ast
from pathlib import Path
import gc
from utilities.model_util import instantiate_from_config
import torch
from tqdm import tqdm
import os, glob
from model_util import create_noise_scheduler
from lora import LoRANetwork, DEFAULT_TARGET_REPLACE
import train_util
import model_util
import prompt_util
from prompt_util import PromptEmbedsCache, PromptEmbedsPair, PromptSettings
import debug_util
import config_util
from config_util import RootConfig
import random
import numpy as np
import wandb
from PIL import Image
from pytorch_lightning import seed_everything

def flush():
    torch.cuda.empty_cache()
    gc.collect()


def train(
        config: RootConfig,
        prompts: List[PromptSettings],
        device: str,
        scales,
):
    seed_everything(0)
    scales = np.array(scales)
    metadata = {
        "prompts": ",".join([prompt.json() for prompt in prompts]),
        "config": config.json(),
    }
    save_path = Path(config.save.path)
 

    
    if config.logging.use_wandb:
        wandb.init(project=f"LECO_{config.save.name}", config=metadata)
    
    weight_dtype = config_util.parse_precision(config.train.precision)
    save_weight_dtype = config_util.parse_precision(config.train.precision)

    config_yaml = './config/AuDiMa_musicbench.yaml'
    configs = yaml.load(open(config_yaml, "r"), Loader=yaml.FullLoader)
    latent_diffusion = instantiate_from_config(configs["model"])
    checkpoint = torch.load("./log/AuDiMa_musicbench/checkpoints/checkpoint-fad-133.00-global_step=39999.ckpt",map_location="cuda:0")
    latent_diffusion.load_state_dict(checkpoint["state_dict"])
    model = latent_diffusion.model
    mamba = model.M_mamba_model
    mamba.to(device, dtype=weight_dtype)
    mamba.requires_grad_(False)
    mamba.eval()
    latent_diffusion.to(device, dtype=weight_dtype)
    latent_diffusion.requires_grad_(False)
    latent_diffusion.eval()
    
    network = LoRANetwork(
        mamba,
        rank=config.network.rank,
        multiplier=1.0,
        alpha=config.network.alpha,
        train_method=config.network.training_method,
    ).to(device, dtype=weight_dtype)
    
    optimizer_module = train_util.get_optimizer(config.train.optimizer)
    optimizer_kwargs = {}
    if config.train.optimizer_args is not None and len(config.train.optimizer_args) > 0:
        for arg in config.train.optimizer_args.split(" "):
            key, value = arg.split("=")
            value = ast.literal_eval(value)
            optimizer_kwargs[key] = value
    
    optimizer = optimizer_module(network.prepare_optimizer_params(), lr=config.train.lr, **optimizer_kwargs)
    lr_scheduler = train_util.get_lr_scheduler(
        config.train.lr_scheduler,
        optimizer,
        max_iterations=config.train.iterations,
        lr_min=config.train.lr / 100,
    )
    criteria = torch.nn.MSELoss()
    for settings in prompts:
        print(settings)
    
    debug_util.check_requires_grad(network)
    debug_util.check_training_mode(network)
    
    cache = PromptEmbedsCache()
    prompt_pairs: List[PromptEmbedsPair] = []
    
    with torch.no_grad():
        for settings in prompts:
            print(settings)
            for prompt in [
                settings.target,
                settings.positive,
                settings.neutral,
                settings.unconditional,
            ]:
                print(prompt)

                if cache[prompt] == None:
                    cache[prompt] = latent_diffusion.get_learned_conditioning([prompt], key="film_clap_cond1", unconditional_cfg=False).squeeze(0)
            
            prompt_pairs.append(
                PromptEmbedsPair(
                    criteria,
                    cache[settings.target],
                    cache[settings.positive],
                    cache[settings.unconditional],
                    cache[settings.neutral],
                    settings,
                )
            )
    

    flush()
    
    pbar = tqdm(range(config.train.iterations))
    for i in pbar:
        with torch.no_grad():
            shape = (
                    1,
                    8,
                    256,
                    16,
                )
            latents = torch.randn(shape,device=device, dtype=weight_dtype, layout=None).to(
                    device)
            
            prompt_pair: PromptEmbedsPair = prompt_pairs[
                torch.randint(0, len(prompt_pairs), (1,)).item()
            ]                   
            current_timestep =  torch.randint(0, 1000, (1,), device=device).long()
            denoised_latents_low = latents
            noise_0 = train_util.predict_noise_without_lora(
                mamba,
                current_timestep,
                denoised_latents_low,
                prompt_pair.neutral,
                guidance_scale=1,
            ).to(device, dtype=weight_dtype)
            noise_1 = train_util.predict_noise_without_lora(
                mamba,
                current_timestep,
                denoised_latents_low,
                prompt_pair.unconditional,
                guidance_scale=1,
            ).to(device, dtype=weight_dtype)
            noise_2 = train_util.predict_noise_without_lora(
                mamba,
                current_timestep,
                denoised_latents_low,
                prompt_pair.positive,
                guidance_scale=1,
            ).to(device, dtype=weight_dtype)
            low_noise = noise_0+noise_2-noise_1
            denoised_latents_low = denoised_latents_low.to(device, dtype=weight_dtype)
            low_noise = low_noise.to(device, dtype=weight_dtype)
            denoised_latents_high = latents, 
            high_noise = noise_0-noise_2+noise_1
            denoised_latents_high = denoised_latents_high.to(device, dtype=weight_dtype)
            high_noise = high_noise.to(device, dtype=weight_dtype)


        
        network.set_lora_slider(scale=1)
        with network:
            target_latents_high = train_util.predict_noise(
                mamba,
                current_timestep,
                denoised_latents_high,
                train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.positive,
                    prompt_pair.batch_size,
                ),
                guidance_scale=1,
            ).to(device, dtype=weight_dtype)
        
        loss_high = criteria(target_latents_high, high_noise.to(weight_dtype))
        pbar.set_description(f"Loss*1k: {loss_high.item() * 1000:.4f}")
        loss_high.backward()

        network.set_lora_slider(scale=1)
        with network:
            target_latents_low = train_util.predict_noise(
                mamba,
                current_timestep,
                denoised_latents_low,
                train_util.concat_embeddings(
                    prompt_pair.positive,
                    prompt_pair.unconditional,
                    prompt_pair.batch_size,
                ),
                guidance_scale=1,
            ).to(device, dtype=torch.float32)
        loss_low = criteria(target_latents_low, low_noise.to(torch.float32))
        pbar.set_description(f"Loss*1k: {loss_low.item() * 1000:.4f}")
        loss_low.backward()
        

        
        optimizer.step()
        lr_scheduler.step()
        
        del (
            target_latents_low,
            target_latents_high,
        )
        flush()
        
        if (
                i % config.save.per_steps == 0
                and i != 0
                and i != config.train.iterations - 1
        ):
            print("Saving...")
            save_path.mkdir(parents=True, exist_ok=True)
            network.save_weights(
                save_path / f"{config.save.name}_{i}steps.pt",
                dtype=save_weight_dtype,
            )

    print("Saving...")
    save_path.mkdir(parents=True, exist_ok=True)
    network.save_weights(
        save_path / f"{config.save.name}_last.pt",
        dtype=save_weight_dtype,
    )
    
    del (
        mamba,
        optimizer,
        network,
    )
    
    flush()
    
    print("Done.")


def main(args):
    config_file = args.config_file
    
    config = config_util.load_config_from_yaml(config_file)
    if args.name is not None:
        config.save.name = args.name
    attributes = []
    if args.attributes is not None:
        attributes = args.attributes.split(',')
        attributes = [a.strip() for a in attributes]
    
    config.network.alpha = args.alpha
    config.network.rank = args.rank
    config.save.name += f'_alpha{args.alpha}'
    config.save.name += f'_rank{config.network.rank}'
    config.save.name += f'_{config.network.training_method}'
    config.save.path += f'/{config.save.name}'
    
    prompts = prompt_util.load_prompts_from_yaml(config.prompts_file, attributes)
    device = torch.device(f"{args.device}")   
    scales = args.scales.split(',')
    scales = [f.strip() for f in scales]
    scales = [int(s) for s in scales]

    train(config=config, prompts=prompts, device=device,scales=scales)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        required=False,
        default='config.yaml',
        help="Config file for training.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        required=False,
        help="LoRA weight.",
        default=1,
    )
    
    parser.add_argument(
        "--rank",
        type=int,
        required=False,
        help="Rank of LoRA.",
        default=4,
    )
    
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cuda:0",
        help="Device to train on.",
    )
    
    parser.add_argument(
        "--name",
        type=str,
        required=False,
        default="lora_mamba_musicbench",
        help="Device to train on.",
    )
    
    parser.add_argument(
        "--attributes",
        type=str,
        required=False,
        default=None,
        help="attritbutes to disentangle",
    )
    parser.add_argument(
        "--scales",
        type=str,
        required=False,
        default='1,-1',
        help="scales for different attribute-scaled images",
    )
    
    args = parser.parse_args()
    
    main(args)
