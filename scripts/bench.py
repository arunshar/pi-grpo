"""Microbenchmarks for the physics reward and S-KBM evaluator.

Measures throughput of the hot-path operations:

- evaluate (per-step S-KBM violation accounting)
- PhysicsReward.score (full hybrid reward)
- preference synthesis (rank K outputs and form pairs)
"""

from __future__ import annotations

import statistics
import time

import numpy as np

from app.components.kinematic_bicycle import SkbmConfig, evaluate
from app.components.physics_reward import (
    EmpiricalEnvelope,
    PhysicsReward,
    RewardWeights,
)
from app.components.preference_builder import synthesize_from_reward

_CFG = SkbmConfig()
_REWARD = PhysicsReward(_CFG, weights=RewardWeights(), envelope=EmpiricalEnvelope())


def _clean(n: int = 50) -> np.ndarray:
    s = np.zeros((n, 4))
    s[:, 0] = np.arange(n) * 5.0
    s[:, 3] = 5.0
    return s


def _speeding(n: int = 50) -> np.ndarray:
    s = np.zeros((n, 4))
    s[:, 0] = np.arange(n) * 30.0
    s[:, 3] = 30.0
    return s


def _bench(name: str, fn, n_warm: int = 50, n_meas: int = 1_000) -> None:
    for _ in range(n_warm):
        fn()
    samples_us: list[float] = []
    for _ in range(n_meas):
        t0 = time.perf_counter_ns()
        fn()
        samples_us.append((time.perf_counter_ns() - t0) / 1_000)
    samples_us.sort()
    p50 = samples_us[len(samples_us) // 2]
    p95 = samples_us[int(len(samples_us) * 0.95)]
    mean = statistics.mean(samples_us)
    print(f"{name:32s}  p50={p50:7.1f} us  p95={p95:7.1f} us  mean={mean:7.1f} us  "
          f"throughput={1e6 / mean:8.0f} ops/s")


def main() -> None:
    print("benchmarks (n=50 timesteps, n_meas=1000)")
    clean = _clean()
    speeding = _speeding()
    _bench("evaluate (clean)",         lambda: evaluate(clean,    cfg=_CFG))
    _bench("evaluate (speeding)",      lambda: evaluate(speeding, cfg=_CFG))
    _bench("PhysicsReward.score (clean)",    lambda: _REWARD.score(clean))
    _bench("PhysicsReward.score (speeding)", lambda: _REWARD.score(speeding))
    # preference synthesis: K = 4 outputs ranked by reward
    rewards = [
        _REWARD.score(clean).total,
        _REWARD.score(speeding).total,
        _REWARD.score(clean).total - 0.1,
        _REWARD.score(speeding).total - 1.0,
    ]
    inputs = [("Is this trajectory plausible?", ["a", "b", "c", "d"], rewards)]
    _bench("synthesize_from_reward (K=4)", lambda: synthesize_from_reward(inputs, margin_min=1.0))


if __name__ == "__main__":
    main()
