"""Differentiable S-KBM kinematic-envelope residual (the physics-informed term).

Pi-DPM constrains the generated / reconstructed trajectory to the Single-axle
Kinematic Bicycle Model envelope used elsewhere in this repo
(app/components/kinematic_bicycle.py). Where that module audits a finished
trajectory with numpy, this one is a fully differentiable torch operator so it
can be backpropagated through the diffusion model's predicted clean sample.

Given clean positions x0 (B, T, 2) sampled at uniform dt, the residual penalises
the four ways a track can leave the physical envelope (Eq. 20 of the paper,
soft-constraint form), as dimensionless excesses relative to each threshold so
the loss is well-conditioned regardless of units:

    speed      relu(v / v_max - 1)^2
    accel      relu(|a| / a_max - 1)^2
    curvature  relu(|kappa| / kappa_max - 1)^2
    jerk       (jerk / (a_max/dt))^2          (smoothness prior, soft p95 in eval)

The displacement-consistency terms dx = v cos(psi), dy = v sin(psi) are exact for
finite differences and so contribute no gradient; the discriminative physics
signal lives entirely in the envelope/smoothness terms above, which is what
separates teleport / excess-speed / freeze anomalies from plausible motion.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import PiDPMConfig


def kinematics(xy: torch.Tensor, dt: float = 1.0) -> dict[str, torch.Tensor]:
    """Finite-difference kinematic state from positions.

    Args:
        xy: (B, T, 2) positions.
        dt: sampling interval.

    Returns dict of speed v (B, T-1), accel a (B, T-2), heading psi (B, T-1),
    heading-rate psi_dot (B, T-2), curvature kappa (B, T-2), jerk (B, T-3).
    """
    d = torch.diff(xy, dim=1) / dt                      # (B, T-1, 2)
    dx, dy = d[..., 0], d[..., 1]
    v = torch.linalg.vector_norm(d, dim=-1)             # (B, T-1)
    psi = torch.atan2(dy, dx)                           # (B, T-1)
    a = torch.diff(v, dim=1) / dt                       # (B, T-2)
    dpsi = torch.diff(psi, dim=1)
    # wrap to (-pi, pi] so a heading flip is not read as a huge turn rate
    dpsi = (dpsi + torch.pi) % (2 * torch.pi) - torch.pi
    psi_dot = dpsi / dt                                 # (B, T-2)
    v_mid = 0.5 * (v[:, :-1] + v[:, 1:])
    kappa = psi_dot / v_mid.clamp_min(1e-6)             # kappa = psi_dot / v
    jerk = torch.diff(a, dim=1) / dt if a.shape[1] > 1 else torch.zeros_like(a[:, :0])
    return {"v": v, "a": a, "psi": psi, "psi_dot": psi_dot, "kappa": kappa, "jerk": jerk}


class PhysicsResidual(nn.Module):
    """Per-trajectory S-KBM envelope residual, mean over time -> (B,)."""

    def __init__(self, cfg: PiDPMConfig) -> None:
        super().__init__()
        self.dt = cfg.dt
        self.pos_scale = cfg.pos_scale
        self.v_max = cfg.v_max_mps
        self.a_max = cfg.a_max_mps2
        self.kappa_max = cfg.curvature_max
        self.w_speed, self.w_accel, self.w_curv, self.w_jerk = cfg.phys_weights

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """xy: (B, T, 2) clean positions in MODEL space -> (B,) non-negative residual.

        The envelope thresholds are physical (m/s, m/s^2, rad/m), so the
        model-space positions are de-scaled to metres first; the mean offset
        does not affect the finite-difference kinematics.
        """
        if xy.shape[1] < 4:
            return xy.new_zeros(xy.shape[0])
        k = kinematics(xy * self.pos_scale, self.dt)
        # dimensionless excesses relative to each threshold, so the residual is
        # well-conditioned regardless of units (a 2x overspeed contributes ~1,
        # not (v - v_max)^2 in m^2/s^2)
        speed = torch.relu(k["v"] / self.v_max - 1.0).pow(2).mean(dim=1)
        accel = torch.relu(k["a"].abs() / self.a_max - 1.0).pow(2).mean(dim=1)
        curv = torch.relu(k["kappa"].abs() / self.kappa_max - 1.0).pow(2).mean(dim=1)
        jerk_scale = self.a_max / self.dt  # m/s^3
        jerk = (k["jerk"] / jerk_scale).pow(2).mean(dim=1) if k["jerk"].shape[1] else speed.new_zeros(speed.shape)
        return (
            self.w_speed * speed
            + self.w_accel * accel
            + self.w_curv * curv
            + self.w_jerk * jerk
        )
