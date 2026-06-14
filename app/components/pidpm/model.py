"""Conditional Transformer denoiser for Pi-DPM (the encoder-decoder).

A sequence model over the (T, in_dim) trajectory that predicts the diffusion
noise epsilon. Diffusion timestep and an optional external condition (pooled
scene tokens, origin-destination, neighbour context) are injected through
adaptive layer norm (adaLN), the same conditioning DiT uses, which keeps the
backbone permutation/scale stable across noise levels.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import PiDPMConfig


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding of integer diffusion timesteps t -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class AdaLNBlock(nn.Module):
    """Pre-norm Transformer block with adaLN-zero conditioning."""

    def __init__(self, cfg: PiDPMConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d, cfg.ffn_mult * d),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_mult * d, d),
        )
        # adaLN: produce per-block shift/scale/gate from the conditioning vector
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.ada(c).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_a[:, None] * attn
        h = self.norm2(x) * (1 + scale_m[:, None]) + shift_m[:, None]
        x = x + gate_m[:, None] * self.mlp(h)
        return x


class TrajectoryDenoiser(nn.Module):
    """Predict epsilon given (noisy trajectory x_t, timestep t, condition cond)."""

    def __init__(self, cfg: PiDPMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.in_proj = nn.Linear(cfg.in_dim, d)
        self.pos = nn.Parameter(torch.zeros(1, cfg.seq_len, d))
        nn.init.normal_(self.pos, std=0.02)
        self.t_embed = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.cond_proj = nn.Linear(cfg.cond_dim, d) if cfg.cond_dim > 0 else None
        self.blocks = nn.ModuleList(AdaLNBlock(cfg) for _ in range(cfg.n_layers))
        self.norm_out = nn.LayerNorm(d, elementwise_affine=False)
        self.out = nn.Linear(d, cfg.in_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x_t: (B, T, in_dim); t: (B,) long; cond: (B, cond_dim) or None -> (B, T, in_dim)."""
        _, seq, _ = x_t.shape
        h = self.in_proj(x_t) + self.pos[:, :seq]
        c = self.t_embed(timestep_embedding(t, self.cfg.d_model))
        if self.cond_proj is not None and cond is not None:
            c = c + self.cond_proj(cond)
        for blk in self.blocks:
            h = blk(h, c)
        return self.out(self.norm_out(h))
