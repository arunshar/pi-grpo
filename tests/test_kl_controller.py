"""AdaptiveKLController stays inside its bounds."""

from __future__ import annotations

from app.trainers.base import AdaptiveKLController


def test_kl_controller_clips_to_bounds() -> None:
    c = AdaptiveKLController(kl_coef=0.2, target=6.0, horizon=10)
    for _ in range(100):
        c.update(current_kl=200.0, n_steps=1)
    assert c.kl_coef <= c.clip_max
    for _ in range(100):
        c.update(current_kl=0.0, n_steps=1)
    assert c.kl_coef >= c.clip_min
