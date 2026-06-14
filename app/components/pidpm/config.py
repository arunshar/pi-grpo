"""Configuration for Pi-DPM."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PiDPMConfig:
    # ----- data / representation -----------------------------------------
    seq_len: int = 24              # trajectory length T (points per sample)
    in_dim: int = 2                # state dims used by the denoiser (x, y)
    dt: float = 1.0                # uniform sampling interval (seconds)
    pos_scale: float = 50.0        # metres per model-space unit; the diffusion runs
                                   # in model space (x_phys / pos_scale, ~unit std)
                                   # while the S-KBM envelope stays in physical units
    cond_dim: int = 0              # external condition width (0 = unconditional);
                                   # e.g. pooled scene tokens or origin-destination

    # ----- denoiser ------------------------------------------------------
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ffn_mult: int = 4
    dropout: float = 0.0

    # ----- diffusion -----------------------------------------------------
    timesteps: int = 200           # T diffusion steps
    beta_schedule: str = "cosine"  # "cosine" | "linear"
    predict: str = "eps"           # the denoiser predicts the noise epsilon
    x0_clamp: float = 6.0          # clamp the predicted clean sample (model-space
                                   # units) before the physics term, for stability

    # ----- physics envelope (S-KBM), reused from kinematic_bicycle -------
    v_max_mps: float = 12.86       # 25 kts vessel proxy
    a_max_mps2: float = 0.5
    curvature_max: float = 0.20    # rad/m soft cap
    phys_weights: tuple[float, float, float, float] = (1.0, 1.0, 0.5, 0.25)
    # (speed, accel, curvature, jerk)

    # ----- training ------------------------------------------------------
    physics_weight: float = 0.05   # lambda_phys: soft S-KBM regulariser weight
                                   # (denoiser eps-MSE leads; physics shapes x0_hat)
    recon_weight: float = 0.0      # optional extra x0 MSE term in loss
    lr: float = 2e-4
    weight_decay: float = 1e-5
    batch_size: int = 128
    epochs: int = 20
    grad_clip: float = 1.0
    amp: bool = True               # mixed precision on CUDA
    ema_decay: float = 0.999

    # ----- anomaly scoring ----------------------------------------------
    eval_noise_t: int = 50         # tau: forward-noise level for reconstruction
    ddim_steps: int = 20           # reverse steps for reconstruction
    score_w_rec: float = 0.5       # w_rec in eps = w_rec*rec + w_phy*phys
    score_w_phy: float = 0.5       # w_phy
    flag_lambda: float = 1.0       # lambda threshold (set from val quantile)

    seed: int = 0

    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0, "d_model must divide n_heads"
        return self.d_model // self.n_heads
