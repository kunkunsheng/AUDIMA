
import numpy as np
import torch
from lora import LoRANetwork, DEFAULT_TARGET_REPLACE
import train_util
from tqdm import tqdm
import yaml
from utilities.model_util import instantiate_from_config
import inspect
from pytorch_lightning import seed_everything
def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()
def make_ddim_timesteps(
    ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True
):
    if ddim_discr_method == "uniform":
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == "quad":
        ddim_timesteps = (
            (np.linspace(0, np.sqrt(num_ddpm_timesteps * 0.8), num_ddim_timesteps)) ** 2
        ).astype(int)
    else:
        raise NotImplementedError(
            f'There is no ddim discretization method called "{ddim_discr_method}"'
        )

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f"Selected timesteps for ddim sampler: {steps_out}")
    return steps_out

def prepare_extra_step_kwargs(noise_scheduler, generator, eta):


    accepts_eta = "eta" in set(inspect.signature(noise_scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta
    
    # 检查调度器函数是否有'generator'关键字
    # check if the scheduler accepts generator
    accepts_generator = "generator" in set(inspect.signature(noise_scheduler.step).parameters.keys())
    if accepts_generator:
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt(
        (1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev)
    )
    if verbose:
        print(
            f"Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}"
        )
        print(
            f"For the chosen value of eta, which is {eta}, "
            f"this results in the following sigma_t schedule for ddim sampler {sigmas}"
        )
    return sigmas, alphas, alphas_prev
def main():
    seed_everything(0)
    pretrained_model_name_or_path = ""
    weight_dtype = torch.float32
    device = 'cuda'
    lora_weights = ""
    config_yaml = ""
    configs = yaml.load(open(config_yaml, "r"), Loader=yaml.FullLoader)
    latent_diffusion = instantiate_from_config(configs["model"])
    checkpoint = torch.load(pretrained_model_name_or_path,map_location="cuda")
    latent_diffusion.load_state_dict(checkpoint["state_dict"])
    model = latent_diffusion.model
    mamba = model.M_mamba_model
    mamba.to(device, dtype=weight_dtype)
    mamba.requires_grad_(False)
    mamba.eval()
    vocoder = latent_diffusion.first_stage_model.vocoder
    vocoder.to(device, dtype=weight_dtype)
    vocoder.requires_grad_(False)
    vocoder.eval()
    latent_diffusion.to(device, dtype=weight_dtype)
    latent_diffusion.requires_grad_(False)
    latent_diffusion.eval()
    prompts = ["A piano piece of music"]
    scales = [1,0,-1]
    start_noise = 1000
    negative_prompt = ""
    # number of images per prompt
    num_images_per_prompt = 1
    guidance_scale = 1
    
    for prompt in prompts:
        # for different seeds on same prompt
        for _ in range(num_images_per_prompt):
            network = LoRANetwork(
                mamba,
                rank=4,
                multiplier=1.0,
                alpha=1,
                train_method="noxattn",
            ).to(device, dtype=weight_dtype)
            network.load_state_dict(torch.load(lora_weights,map_location="cuda"))

            for scale in scales:
                negativet_embeds = latent_diffusion.get_learned_conditioning([negative_prompt], key="film_clap_cond1", unconditional_cfg=False).squeeze(0)
                prompt_embeds = latent_diffusion.get_learned_conditioning([prompt], key="film_clap_cond1", unconditional_cfg=False).squeeze(0)
                prompt_embeds = train_util.concat_embeddings(negativet_embeds, prompt_embeds, batch_size=1)
                
                shape = (
                    1,
                    8,
                    256,
                    16,
                )
                latents = torch.randn(shape,device=device, dtype=weight_dtype, layout=None).to(
                    device)
                
                ddim_timesteps = make_ddim_timesteps(
                    ddim_discr_method="uniform",
                    num_ddim_timesteps=200,
                    num_ddpm_timesteps=1000,
                    verbose=False,
                )

                ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
                    alphacums=latent_diffusion.alphas_cumprod.cpu(),
                    ddim_timesteps=ddim_timesteps,
                    eta=1.0,
                    verbose=False,
                )

                ddim_sqrt_one_minus_alphas = np.sqrt(1.0 - ddim_alphas)
                ddim_sigmas_for_original_num_steps = 1.0 * torch.sqrt((1 - latent_diffusion.alphas_cumprod_prev)/ (1 - latent_diffusion.alphas_cumprod)* (1 -latent_diffusion.alphas_cumprod / latent_diffusion.alphas_cumprod_prev))
                timesteps = (ddim_timesteps)
                time_range = (np.flip(timesteps))
                total_steps = timesteps.shape[0]
                iterator = tqdm(time_range, desc="DDIM Sampler", total=total_steps)
                for i,step in enumerate(iterator):
                    index = total_steps - i - 1
                    if step > start_noise:
                        network.set_lora_slider(scale=0)
                    else:
                        network.set_lora_slider(scale=scale)
                    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                    current_timestep = torch.full((1,), step, device=device, dtype=torch.long)
                    latent_model_input = latents
                    # predict the noise residual

                    with torch.no_grad():
                        # with network:
                        noise_pred_uncond = mamba(
                                latent_model_input,
                                current_timestep,
                                prompt_embeds[0].unsqueeze(0).unsqueeze(0),
                            )
                        noise_pred_text = mamba(
                                latent_model_input,
                                current_timestep,
                                prompt_embeds[1].unsqueeze(0).unsqueeze(0),
                            )
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                    alphas = ddim_alphas
                    alphas_prev = ddim_alphas_prev
                    sqrt_one_minus_alphas = ddim_sqrt_one_minus_alphas
                    sigmas = ddim_sigmas
                    a_t = torch.full((1, 1, 1, 1), alphas[index], device=device)
                    a_prev = torch.full((1, 1, 1, 1), alphas_prev[index], device=device)
                    sigma_t = torch.full((1, 1, 1, 1), sigmas[index], device=device)
                    sqrt_one_minus_at = torch.full((1, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)
                    pred_x0 = (latents - sqrt_one_minus_at * noise_pred) / a_t.sqrt()
                    dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * noise_pred
                    noise = sigma_t * noise_like(latents.shape, device, False) * 1.0
                    x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
                    latents = x_prev


                mel = latent_diffusion.decode_first_stage(latents)
                if len(mel.size()) == 4:
                    mel = mel.squeeze(1)
                mel = mel.permute(0, 2, 1)
                waveform = vocoder(mel)
                waveform = waveform.cpu().detach().numpy()
                filename = f"music_{scale}.wav"
                latent_diffusion.save_waveform(waveform, "", filename)


if __name__ == "__main__":
    main()