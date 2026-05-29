"""Hard physics violations dominate the reward signal."""

from __future__ import annotations

import numpy as np

from app.components.kinematic_bicycle import SkbmConfig
from app.components.physics_reward import (
    EmpiricalEnvelope,
    PhysicsReward,
    RewardWeights,
)


def test_violations_dominate_pref_signal() -> None:
    cfg = SkbmConfig(v_max_mps=10.0)
    reward = PhysicsReward(
        cfg,
        weights=RewardWeights(hard=5.0, soft=1.0, data=1.0, pref=1.0),
        envelope=EmpiricalEnvelope(),
    )
    bad = np.zeros((4, 4))
    bad[:, 3] = [0, 30, 60, 90]   # speed wildly above v_max
    rb = reward.score(bad, pref_logit=10.0)
    assert rb.hard < 0
    # hard penalty must dominate even a maxed-out preference signal
    assert rb.total < 0.0


def test_clean_trajectory_gets_neutral_reward() -> None:
    cfg = SkbmConfig()
    reward = PhysicsReward(cfg)
    clean = np.zeros((10, 4))
    clean[:, 0] = np.arange(10) * 5.0
    clean[:, 3] = 5.0
    rb = reward.score(clean)
    assert rb.hard == 0.0
