import torch
import numpy as np

class DDIMSampler:
    """
    Deterministic DDIM sampler (eta=0 by default).
    Interface mirrors the provided DDPMSampler so you can drop-in replace.
    """
    def __init__(
        self,
        generator: torch.Generator,
        num_training_steps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
        eta: float = 0.0,
    ):
        """
        Args:
            generator: torch.Generator for noise sampling (only used if eta > 0).
            num_training_steps: T (number of diffusion steps used in training schedule)
            beta_start/beta_end: same schedule as your DDPM for compatibility
            eta: DDIM stochasticity. 0 = fully deterministic (classic DDIM).
        """
        # Match the original DDPM schedule for compatibility
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.eta = eta

        self.num_train_timesteps = num_training_steps
        # default: full reverse order
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_timesteps(self, num_inference_steps: int = 50):
        """
        Subsample the training schedule to num_inference_steps using a uniform stride.
        """
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self, timestep: int) -> int:
        prev_t = timestep - self.num_train_timesteps // self.num_inference_steps
        return prev_t

    def set_strength(self, strength: float = 1.0):
        """
        Keep same semantics as DDPM helper:
        - strength ~ 1.0: start from high noise (more editing)
        - strength ~ 0.0: start from low noise (preserve input)
        """
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    @torch.no_grad()
    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor) -> torch.Tensor:
        """
        Single DDIM update: x_{t-1} from x_t and eps prediction.

        Using DDIM update (eta controls stochasticity):
          x0_pred = (x_t - sqrt(1 - a_t) * eps) / sqrt(a_t)
          sigma_t = eta * sqrt((1 - a_{t-1})/(1 - a_t)) * sqrt(1 - a_t/a_{t-1})
          x_{t-1} = sqrt(a_{t-1}) * x0_pred + sqrt(1 - a_{t-1} - sigma_t^2) * eps + sigma_t * z,  z ~ N(0, I)

        When eta=0, the last term vanishes and the transition is fully deterministic (classic DDIM).
        """
        t = timestep
        prev_t = self._get_previous_timestep(t)

        # 1) Gather cumulatives
        alpha_prod_t = self.alphas_cumprod[t]                         # a_t
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one  # a_{t-1}

        # 2) Predicted noise eps and predicted x0
        eps = model_output
        # x0_pred: eq. (12) style inversion (common in diffusion code)
        x0_pred = (latents - (1.0 - alpha_prod_t).sqrt() * eps) / alpha_prod_t.sqrt()

        # 3) Compute sigma_t following DDIM paper
        # sigma_t = eta * sqrt((1 - a_{t-1})/(1 - a_t)) * sqrt(1 - a_t/a_{t-1})
        sigma_t = 0.0
        if t > 0:
            sqrt_term = torch.sqrt(torch.clamp(1 - alpha_prod_t_prev, min=1e-20) / torch.clamp(1 - alpha_prod_t, min=1e-20))
            sqrt_term2 = torch.sqrt(torch.clamp(1 - (alpha_prod_t / alpha_prod_t_prev), min=0.0))
            sigma_t = self.eta * (sqrt_term * sqrt_term2)

        # 4) Deterministic mean part
        # coeffs for x0 and eps (no random z yet)
        mean_x0 = alpha_prod_t_prev.sqrt() * x0_pred
        # ensure numerical stability for the sqrt inside
        c = torch.clamp(1 - alpha_prod_t_prev - (sigma_t ** 2 if isinstance(sigma_t, torch.Tensor) else sigma_t**2), min=0.0)
        mean_eps = c.sqrt() * eps

        pred_prev = mean_x0 + mean_eps

        # 5) Optional stochasticity if eta>0
        if t > 0 and (isinstance(sigma_t, torch.Tensor) and torch.any(sigma_t > 0)) or (not isinstance(sigma_t, torch.Tensor) and sigma_t > 0):
            device = latents.device
            z = torch.randn(latents.shape, generator=self.generator, device=device, dtype=latents.dtype)
            # sigma_t may be scalar/tensor; broadcast if needed
            if not isinstance(sigma_t, torch.Tensor):
                sigma_t_tensor = torch.tensor(sigma_t, dtype=latents.dtype, device=device)
            else:
                sigma_t_tensor = sigma_t.to(device=latents.device, dtype=latents.dtype)
            while len(sigma_t_tensor.shape) < len(latents.shape):
                sigma_t_tensor = sigma_t_tensor.unsqueeze(-1)
            pred_prev = pred_prev + sigma_t_tensor * z

        return pred_prev

    def add_noise(self, original_samples: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
        """
        Same as in DDPM: forward noising q(x_t | x_0).
        """
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
