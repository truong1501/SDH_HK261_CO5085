import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GaussianDiffusion:
    """
    Gaussian Diffusion process (DDPM) implementation:
    - Linear & Cosine beta schedules
    - Forward diffusion: q(x_t | x_0)
    - Reverse sampling: p_\theta(x_{t-1} | x_t)
    """
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, schedule="linear", device="cuda"):
        if timesteps < 1:
            raise ValueError("timesteps must be positive")
        self.timesteps = timesteps
        self.device = device

        if schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32, device=device)
        elif schedule == "cosine":
            # Improved cosine schedule (Nichol & Dhariwal, 2021)
            s = 0.008
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=torch.float32, device=device)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def _extract(self, a, t, x_shape):
        """Extract coefficients at specified timesteps t and reshape to match x_shape."""
        batch_size = t.shape[0]
        out = a.gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion process: sample x_t from q(x_t | x_0) analytically.
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample(self, model, x_t, t):
        """
        Single reverse step: sample x_{t-1} from p_theta(x_{t-1} | x_t).
        """
        betas_t = self._extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        # Predict noise using the network (U-Net or DiT)
        predicted_noise = model(x_t, t)

        # Clip predicted clean pixels, then use the DDPM posterior mean.
        alpha_bar = self._extract(self.alphas_cumprod, t, x_t.shape)
        alpha_bar_prev = self._extract(self.alphas_cumprod_prev, t, x_t.shape)
        alpha = self._extract(self.alphas, t, x_t.shape)
        x_0 = ((x_t - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / alpha_bar.sqrt()).clamp(-1, 1)
        model_mean = (betas_t * alpha_bar_prev.sqrt() * x_0 +
                      (1-alpha_bar_prev) * alpha.sqrt() * x_t) / (1-alpha_bar)

        posterior_variance_t = self._extract(self.posterior_variance, t, x_t.shape)
        nonzero = (t != 0).reshape(-1, *((1,) * (x_t.ndim-1)))
        return model_mean + nonzero * posterior_variance_t.sqrt() * torch.randn_like(x_t)

    @torch.no_grad()
    def sample(self, model, shape, save_intermediate=False, num_intermediate=8):
        """
        Full reverse diffusion loop: generate images starting from pure Gaussian noise.
        Optionally returns intermediate steps for visualization.
        """
        model.eval()
        device = self.device
        b = shape[0]
        img = torch.randn(shape, device=device)

        intermediate = []
        save_steps = set(torch.linspace(0, self.timesteps - 1, num_intermediate).long().tolist())

        for step in reversed(range(self.timesteps)):
            t = torch.full((b,), step, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t)

            if save_intermediate and step in save_steps:
                intermediate.append((step, img.clamp(-1.0, 1.0)))

        img = img.clamp(-1.0, 1.0)
        if save_intermediate:
            return img, intermediate
        return img
