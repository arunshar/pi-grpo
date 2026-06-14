"""Pi-DPM: Physics-informed Diffusion Probabilistic Model for trajectories.

Reference implementation of the model in

    Sharma, Yang, Farhadloo, Ghosh, Jayaprakash, Shekhar (2025).
    "Towards Physics-informed Diffusion for Anomaly Detection in
    Trajectories: A Summary of Results." 2nd ACM SIGSPATIAL Workshop on
    Geospatial Anomaly Detection (GeoAnomalies '25), pp. 11-24.

This is a real, GPU-ready PyTorch implementation (not the numpy toy):

  * model.py      -- conditional Transformer denoiser with adaLN time/condition
                     conditioning (the encoder-decoder of Pi-DPM).
  * diffusion.py  -- Gaussian DDPM (cosine schedule, eps-prediction, DDIM
                     reverse sampler, x0 prediction).
  * physics.py    -- differentiable S-KBM kinematic-envelope residual applied to
                     the predicted clean trajectory (the physics-informed term).
  * scoring.py    -- physics-informed anomaly score
                     eps(tau) = w_rec * reconstruction_residual
                              + w_phy * physics_residual,
                     flagged when eps >= lambda (Def 2.4), plus log_prob for the
                     pi-grpo reward model.
  * data.py       -- synthetic trajectory dataset (normal arcs + teleport /
                     excess-speed / freeze anomalies) so train/eval run anywhere
                     with no downloads; a documented hook points at real AIS.
  * train.py      -- training loop (AMP, DDP-ready via accelerate, checkpoints).
  * eval.py       -- anomaly-detection evaluation reporting REAL AUROC / PR.

The numpy miniature this grew out of lives in the arun-papers skill
(code/physics-informed-diffusion-anomaly.py).
"""

from __future__ import annotations

from .config import PiDPMConfig
from .diffusion import GaussianDiffusion
from .model import TrajectoryDenoiser
from .physics import PhysicsResidual, kinematics
from .scoring import PiDPM, AnomalyScore

__all__ = [
    "PiDPMConfig",
    "GaussianDiffusion",
    "TrajectoryDenoiser",
    "PhysicsResidual",
    "kinematics",
    "PiDPM",
    "AnomalyScore",
]
