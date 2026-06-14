"""Tests for the real Pi-DPM (physics-informed diffusion).

Fast: a tiny config so the whole suite runs on CPU in a few seconds.
"""

from __future__ import annotations

import numpy as np
import torch

from app.components.pidpm.config import PiDPMConfig
from app.components.pidpm.data import TrajectoryDataset, _inject, _normal_arc
from app.components.pidpm.physics import PhysicsResidual, kinematics
from app.components.pidpm.scoring import PiDPM
from app.components.pidpm_scorer import PiDpmScorer


def _tiny() -> PiDPMConfig:
    return PiDPMConfig(seq_len=16, d_model=32, n_heads=2, n_layers=2, timesteps=40,
                       ddim_steps=5, eval_noise_t=15, batch_size=32, epochs=1)


def test_denoiser_shapes_and_grad():
    cfg = _tiny()
    model = PiDPM(cfg)
    x = torch.randn(4, cfg.seq_len, cfg.in_dim)
    out = model.diffusion.loss(x)
    assert out["loss"].requires_grad and out["loss"].ndim == 0
    out["loss"].backward()
    assert any(p.grad is not None for p in model.parameters())


def test_q_sample_predict_x0_roundtrip():
    cfg = _tiny()
    model = PiDPM(cfg)
    x0 = torch.randn(3, cfg.seq_len, cfg.in_dim)
    t = torch.zeros(3, dtype=torch.long)  # t=0 => almost no noise
    noise = torch.randn_like(x0)
    x_t = model.diffusion.q_sample(x0, t, noise)
    x0_hat = model.diffusion.predict_x0(x_t, t, noise)
    assert torch.allclose(x0, x0_hat, atol=1e-4)


def _to_model(a: np.ndarray, cfg: PiDPMConfig) -> torch.Tensor:
    """Physical metres -> model space (per-sample centred, /pos_scale)."""
    return torch.tensor(((a - a.mean(0)) / cfg.pos_scale)[None], dtype=torch.float32)


def test_physics_residual_flags_excess_speed():
    cfg = _tiny()
    phys = PhysicsResidual(cfg)
    rng = np.random.default_rng(0)
    normal = _normal_arc(rng, cfg.seq_len, cfg.v_max_mps * 0.5)
    fast = _inject(rng, normal, "excess_speed")
    rn = phys(_to_model(normal, cfg))
    rf = phys(_to_model(fast, cfg))
    assert rf.item() > rn.item()
    assert rn.item() < 1.0  # a plausible track sits inside the S-KBM envelope


def test_kinematics_keys():
    k = kinematics(torch.randn(2, 10, 2))
    assert {"v", "a", "psi", "psi_dot", "kappa", "jerk"} <= set(k)


def test_score_separates_anomaly():
    cfg = _tiny()
    model = PiDPM(cfg).eval()
    rng = np.random.default_rng(1)
    normal = _normal_arc(rng, cfg.seq_len, cfg.v_max_mps * 0.5)
    tele = _inject(rng, normal, "teleport")
    xs = torch.cat([_to_model(normal, cfg), _to_model(tele, cfg)], dim=0)
    s = model.score(xs)
    # the teleport must score strictly higher (physics + reconstruction)
    assert s.score[1] > s.score[0]
    assert s.score.shape == (2,) and s.recon.shape == (2,)


def test_training_reduces_loss():
    cfg = _tiny()
    model = PiDPM(cfg)
    ds = TrajectoryDataset(cfg, n=128, anomaly_ratio=0.0, seed=0)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    xb = torch.from_numpy(ds.x[:64])
    first = float(model.diffusion.loss(xb)["loss"])
    last = first
    for _ in range(40):
        opt.zero_grad()
        loss = model.diffusion.loss(xb)["loss"]
        loss.backward()
        opt.step()
        last = float(loss)
    assert last < first


def test_scorer_bridge_proxy_prefers_smooth():
    scorer = PiDpmScorer(checkpoint_path=None, device="cpu")
    t = np.linspace(0, 1, 20)
    smooth = np.stack([t, 0.5 * t], axis=1)
    jerky = smooth.copy()
    jerky[10] += np.array([5.0, -5.0])
    assert scorer.log_prob(smooth) > scorer.log_prob(jerky)


def test_checkpoint_roundtrip(tmp_path):
    cfg = _tiny()
    model = PiDPM(cfg)
    p = tmp_path / "pidpm.pt"
    model.save(str(p))
    loaded = PiDPM.from_checkpoint(str(p))
    x = torch.randn(2, cfg.seq_len, cfg.in_dim)
    with torch.no_grad():
        a = loaded.score(x).score
    assert a.shape == (2,)
