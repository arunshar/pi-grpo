"""Gaussian diffusion for Pi-DPM (DDPM forward + DDIM reverse).

eps-prediction parameterisation. The training loss is the standard noise MSE
plus the physics-informed S-KBM residual evaluated on the model's predicted
clean trajectory x0_hat (so the physics gradient flows back through the
denoiser), matching the total objective in the paper:

    L = E_{t, x0, eps} [ || eps - eps_theta(x_t, t, c) ||^2
                         + lambda_phys * R_phys(x0_hat)
                         + lambda_rec  * || x0 - x0_hat ||^2 ]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PiDPMConfig
from .model import TrajectoryDenoiser
from .physics import PhysicsResidual


def _cosine_betas(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Nichol and Dhariwal cosine schedule."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    acp = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    acp = acp / acp[0]
    betas = 1 - (acp[1:] / acp[:-1])
    return betas.clamp(1e-8, 0.999)


def _linear_betas(timesteps: int) -> torch.Tensor:
    return torch.linspace(1e-4, 0.02, timesteps)


def _gather(a: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Index schedule tensor a at timesteps t, broadcast to (B, 1, 1)."""
    out = a.gather(0, t)
    return out.reshape(t.shape[0], *([1] * (len(shape) - 1)))


class GaussianDiffusion(nn.Module):
    """Holds the noise schedule and the train/sample/reconstruct logic."""

    def __init__(self, model: TrajectoryDenoiser, cfg: PiDPMConfig) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.physics = PhysicsResidual(cfg)
        self.num_t = cfg.timesteps

        betas = _cosine_betas(cfg.timesteps) if cfg.beta_schedule == "cosine" else _linear_betas(cfg.timesteps)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        # registered as buffers so .to(device) moves the whole schedule
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", acp)
        self.register_buffer("sqrt_acp", acp.sqrt())
        self.register_buffer("sqrt_one_minus_acp", (1.0 - acp).sqrt())

    # --------------------------------------------------------------- forward
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Diffuse x0 to level t: x_t = sqrt(acp) x0 + sqrt(1-acp) eps."""
        return _gather(self.sqrt_acp, t, x0.shape) * x0 + _gather(self.sqrt_one_minus_acp, t, x0.shape) * noise

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Recover the clean estimate from x_t and predicted noise."""
        return (x_t - _gather(self.sqrt_one_minus_acp, t, x_t.shape) * eps) / _gather(self.sqrt_acp, t, x_t.shape)

    # --------------------------------------------------------------- loss
    def loss(self, x0: torch.Tensor, cond: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        b = x0.shape[0]
        t = torch.randint(0, self.num_t, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        eps_hat = self.model(x_t, t, cond)
        loss_eps = F.mse_loss(eps_hat, noise)
        x0_hat = self.predict_x0(x_t, t, eps_hat).clamp(-self.cfg.x0_clamp, self.cfg.x0_clamp)
        # SNR-weight the physics term: at high noise x0_hat is unreliable, so
        # weight each sample by alpha_bar(t) (near 1 at low noise, near 0 at high).
        snr_w = self.alphas_cumprod.index_select(0, t)
        loss_phys = (snr_w * self.physics(x0_hat)).mean()
        loss_rec = F.mse_loss(x0_hat, x0) if self.cfg.recon_weight > 0 else x0.new_zeros(())
        total = loss_eps + self.cfg.physics_weight * loss_phys + self.cfg.recon_weight * loss_rec
        return {"loss": total, "eps": loss_eps.detach(), "phys": loss_phys.detach(), "rec": loss_rec.detach()}

    # --------------------------------------------------------------- reverse
    @torch.no_grad()
    def ddim_sample(
        self,
        shape: tuple[int, int, int],
        cond: torch.Tensor | None = None,
        steps: int | None = None,
        x_init: torch.Tensor | None = None,
        t_start: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Deterministic (eta=0) DDIM reverse process.

        With x_init/t_start given, denoises a partially-noised trajectory back to
        a clean estimate (used for reconstruction-based anomaly scoring); otherwise
        samples from pure noise (generation).
        """
        device = device or self.betas.device
        steps = steps or self.cfg.ddim_steps
        t_start = self.num_t - 1 if t_start is None else t_start
        ts = torch.linspace(t_start, 0, steps + 1).round().long().to(device)
        x = x_init if x_init is not None else torch.randn(shape, device=device)
        for i in range(steps):
            t_cur = ts[i].expand(shape[0])
            t_nxt = ts[i + 1].expand(shape[0])
            eps = self.model(x, t_cur, cond)
            x0 = self.predict_x0(x, t_cur, eps)
            acp_nxt = _gather(self.alphas_cumprod, t_nxt, x.shape)
            x = acp_nxt.sqrt() * x0 + (1 - acp_nxt).sqrt() * eps
        return x

    @torch.no_grad()
    def reconstruct(
        self, x0: torch.Tensor, cond: torch.Tensor | None = None, t_eval: int | None = None
    ) -> torch.Tensor:
        """Noise x0 to level t_eval then DDIM-denoise back to x0_hat."""
        t_eval = t_eval or self.cfg.eval_noise_t
        t = torch.full((x0.shape[0],), t_eval, device=x0.device, dtype=torch.long)
        x_t = self.q_sample(x0, t, torch.randn_like(x0))
        return self.ddim_sample(tuple(x0.shape), cond=cond, x_init=x_t, t_start=t_eval, device=x0.device)
