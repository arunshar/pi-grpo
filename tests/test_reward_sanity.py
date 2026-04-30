"""Sanity test for reward configs. Required to pass before any PPO run."""

from __future__ import annotations

from pathlib import Path

from app.reward_models.physics_reward_model import load


def test_default_reward_config_loads() -> None:
    cfg_path = Path("configs/physics_reward.yaml")
    if not cfg_path.exists():
        # nothing to test in a stripped install
        return
    reward = load(str(cfg_path))
    assert reward.weights.hard >= reward.weights.soft
