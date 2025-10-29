

import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler
import math

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8


def generate(
    prompt,
    uncond_prompt=None,
    input_image=None,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
    schedule: str = "linear",   # "linear" hoặc "karras"
    rho: float = 7.0,           # độ cong Karras
):
    """
    DDPM pipeline + (optional) Karras timetable spacing.
    Karras chỉ thay spacing của timesteps, vẫn dùng DDPM sampler.step().
    """
    with torch.no_grad():
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")

        to_idle = (lambda x: x.to(idle_device)) if idle_device else (lambda x: x)

        # ===== RNG =====
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        # ===== CLIP encode =====
        clip = models["clip"]
        clip.to(device)

        if do_cfg:
            if uncond_prompt is None:
                uncond_prompt = ""
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            cond_context = clip(cond_tokens)

            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            uncond_context = clip(uncond_tokens)

            context = torch.cat([cond_context, uncond_context], dim=0)
        else:
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            context = clip(tokens)
        to_idle(clip)

        # ===== Sampler DDPM =====
        sampler = DDPMSampler(generator)
        sampler.set_inference_timesteps(n_inference_steps)

        # ===== (Optional) Karras schedule – đặt TRƯỚC img2img add_noise =====
        if schedule == "karras":
            print("[INFO] Using Karras timestep schedule with DDPM sampler")
            # 1) sigma(t) từ lưới train
            alphas_cumprod = sampler.alphas_cumprod.to(device)           # [T]
            sigmas_train = torch.sqrt((1 - alphas_cumprod) / (alphas_cumprod + 1e-12))

            # 2) tạo N sigma theo Karras (decreasing)
            def _karras_sigmas(n_steps: int, sigma_min: float, sigma_max: float, rho: float, device):
                ramp = torch.linspace(0, 1, n_steps, device=device)
                min_r, max_r = sigma_min ** (1 / rho), sigma_max ** (1 / rho)
                sigmas = (max_r + ramp * (min_r - max_r)) ** rho
                return torch.flip(sigmas, dims=[0])  # lớn -> nhỏ

            sigmas = _karras_sigmas(
                n_steps=n_inference_steps,
                sigma_min=float(sigmas_train[-1].item()),
                sigma_max=float(sigmas_train[0].item()),
                rho=rho,
                device=device,
            )

            # 3) map mỗi σ_k sang t gần nhất trong lưới train
            diffs = (sigmas_train.view(-1, 1) - sigmas.view(1, -1)).abs()
            t_indices = torch.argmin(diffs, dim=0).to(torch.long)

            # 4) đảm bảo thứ tự giảm dần + bỏ trùng lặp liên tiếp
            t_indices = torch.flip(t_indices, dims=[0])
            t_indices = torch.unique_consecutive(t_indices)
            if t_indices[0] < t_indices[-1]:
                t_indices = torch.flip(t_indices, dims=[0])

            sampler.timesteps = t_indices.to(device)

        # ===== Initialize latents =====
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image is not None:
            # IMG2IMG
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            input_image_tensor = input_image_tensor.unsqueeze(0)          # (B, H, W, C)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)   # (B, C, H, W)

            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            latents = encoder(input_image_tensor, encoder_noise)

            # dùng đúng timestep đầu theo lịch đã set (linear hoặc karras)
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            # TXT2IMG
            latents = torch.randn(latents_shape, generator=generator, device=device)

        # ===== UNet diffusion =====
        diffusion = models["diffusion"]
        diffusion.to(device)

        # ===== Denoise loop (DDPM step) =====
        timesteps_iter = tqdm(sampler.timesteps.tolist())
        for t in timesteps_iter:
            t_int = int(t)
            time_embedding = get_time_embedding(t_int).to(device)

            model_input = latents
            if do_cfg:
                model_input = model_input.repeat(2, 1, 1, 1)

            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = output_uncond + cfg_scale * (output_cond - output_uncond)

            latents = sampler.step(t_int, latents, model_output)

        # ===== VAE decode =====
        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)
        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]


def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x


def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    t = torch.tensor([timestep], dtype=torch.float32)
    x = t[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
