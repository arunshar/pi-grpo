"""Pi-DPM inference: physics-informed anomaly score and reward log-prob.

The anomaly score for a trajectory (Def 2.4) combines the diffusion
reconstruction residual with the S-KBM physics residual:

    eps(tau) = w_rec * || x0 - x0_hat ||^2  +  w_phy * R_phys(x0)

where x0_hat is the DDIM reconstruction of x0 after forward-noising to level
tau. A trajectory is flagged anomalous when eps(tau) >= lambda. The same score,
negated, is exposed as log_prob for the pi-grpo physics-aware reward model
(higher reward = more physically plausible, better reconstructed).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .config import PiDPMConfig
from .diffusion import GaussianDiffusion
from .model import TrajectoryDenoiser
from .physics import PhysicsResidual


@dataclass
class AnomalyScore:
    score: np.ndarray          # (B,) eps(tau)
    recon: np.ndarray          # (B,) reconstruction residual
    physics: np.ndarray        # (B,) S-KBM residual
    flagged: np.ndarray        # (B,) bool, score >= lambda


class PiDPM(nn.Module):
    """Top-level Pi-DPM: denoiser + diffusion + scoring, GPU-ready.

    Construct, optionally load a checkpoint, then call score()/log_prob() for
    inference or diffusion.loss() during training.
    """

    def __init__(self, cfg: PiDPMConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or PiDPMConfig()
        self.denoiser = TrajectoryDenoiser(self.cfg)
        self.diffusion = GaussianDiffusion(self.denoiser, self.cfg)
        self.physics = PhysicsResidual(self.cfg)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # --------------------------------------------------------------- io
    @classmethod
    def from_checkpoint(cls, path: str, map_location: str | torch.device = "cpu") -> "PiDPM":
        # torch>=2.6 defaults torch.load to weights_only=True; our checkpoint carries
        # a PiDPMConfig dataclass under "cfg", so allow exactly that class to unpickle.
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([PiDPMConfig])
        ckpt = torch.load(path, map_location=map_location)
        cfg = ckpt.get("cfg") or PiDPMConfig()
        model = cls(cfg)
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model

    def save(self, path: str) -> None:
        torch.save({"model": self.state_dict(), "cfg": self.cfg}, path)

    # --------------------------------------------------------------- scoring
    @torch.no_grad()
    def score(self, x0: torch.Tensor, cond: torch.Tensor | None = None, lam: float | None = None) -> AnomalyScore:
        """x0: (B, T, in_dim) -> AnomalyScore. Runs reconstruction + physics."""
        self.eval()
        lam = self.cfg.flag_lambda if lam is None else lam
        x0_hat = self.diffusion.reconstruct(x0, cond)
        recon = (x0 - x0_hat).pow(2).flatten(1).mean(dim=1)
        phys = self.physics(x0)
        eps = self.cfg.score_w_rec * recon + self.cfg.score_w_phy * phys
        return AnomalyScore(
            score=eps.cpu().numpy(),
            recon=recon.cpu().numpy(),
            physics=phys.cpu().numpy(),
            flagged=(eps >= lam).cpu().numpy(),
        )

    @torch.no_grad()
    def log_prob(self, trajectory: np.ndarray | torch.Tensor, cond: torch.Tensor | None = None) -> float:
        """Negative anomaly score of a single trajectory, for the reward model.

        Accepts an (T, >=2) array; uses the first two columns (x, y) or, when an
        AIS-style (t, lat, lon, sog, cog) row is given, the (lon, lat) columns.
        """
        x = trajectory if isinstance(trajectory, torch.Tensor) else torch.as_tensor(trajectory)
        x = x.float().to(self.device)
        if x.ndim == 2:
            x = x[None]
        xy = self._fit_len(self._to_xy(x))
        # map physical metres -> model space (per-sample centred, /pos_scale),
        # matching how the dataset and denoiser were trained
        xy = (xy - xy.mean(dim=1, keepdim=True)) / self.cfg.pos_scale
        return float(-self.score(xy, cond).score[0])

    # --------------------------------------------------------------- helpers
    def _to_xy(self, x: torch.Tensor) -> torch.Tensor:
        """Map an input batch (B, T, C) to the (B, T, in_dim) the denoiser uses."""
        c = x.shape[-1]
        if c >= 5:  # AIS (t, lat, lon, sog, cog) -> (lon, lat)
            return x[..., [2, 1]]
        return x[..., : self.cfg.in_dim]

    def _fit_len(self, xy: torch.Tensor) -> torch.Tensor:
        """Pad or crop the time axis to cfg.seq_len (the denoiser is fixed-length)."""
        t = xy.shape[1]
        if t == self.cfg.seq_len:
            return xy
        if t > self.cfg.seq_len:
            return xy[:, : self.cfg.seq_len]
        pad = xy[:, -1:].expand(-1, self.cfg.seq_len - t, -1)
        return torch.cat([xy, pad], dim=1)
