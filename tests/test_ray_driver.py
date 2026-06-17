"""Unit + mock tests for the Ray-parallel GRPO driver.

These run with NO Ray cluster and NO GPU. The injectable reward pool is the seam
that lets us do this: every test passes a tiny pool that delegates straight to
the serial `driver._reward_matrix`, so `train_grpo_ray` exercises its full
control flow (sample -> score -> learn, timing, history, result) on CPU in a
fraction of a second.

Coverage:
  (1) convergence parity   -> reward_end >= reward_start under a stub pool
  (2) bit-identical parity -> stub-pool run == serial train_grpo run (the gate)
  (3) timing decomposition -> t_sample/t_score/t_learn present, >=0, ~ sum
  (4) pool injection       -> the pool is actually called B-shaped each step
  (5) default left serial  -> default_reward_pool degrades to serial; the
                              existing serial train_grpo stays importable/unchanged
"""

from __future__ import annotations

import numpy as np
import torch

from app.policy import driver as serial_driver
from app.policy.driver import TrainConfig, train_grpo
from app.policy.ray_driver import (
    RayTrainResult,
    _SerialRewardPool,
    default_reward_pool,
    train_grpo_ray,
)


# --------------------------------------------------------------------------- pools


class _StubPool:
    """Mock reward pool: delegates to the serial _reward_matrix (no Ray)."""

    def __init__(self) -> None:
        self.calls = 0
        self.last_shape: tuple[int, int, int] | None = None

    def score_rollouts(self, rollouts, codebook, reward, scorer):
        self.calls += 1
        self.last_shape = tuple(rollouts.shape)
        return serial_driver._reward_matrix(rollouts, codebook, reward, scorer)


def _tiny_cfg() -> TrainConfig:
    # Small but enough steps that a real GRPO loop moves mean reward upward,
    # mirroring tests/test_policy.py::test_grpo_increases_mean_reward.
    return TrainConfig(
        steps=40, batch_prompts=4, group_size=6, prompt_len=2, horizon=10,
        lr=1e-2, seed=0, d_model=32, n_layers=1, n_heads=4,
    )


# ----------------------------------------------------------------- (1) convergence


def test_ray_driver_increases_mean_reward_with_stub_pool() -> None:
    """A real GRPO loop scored through a (serial) injected pool still learns."""
    res = train_grpo_ray(_tiny_cfg(), reward_pool=_StubPool())
    assert isinstance(res, RayTrainResult)
    assert np.isfinite(res.reward_start) and np.isfinite(res.reward_end)
    # Same convergence behavior as serial train_grpo on the feasible-region task.
    assert res.reward_end > res.reward_start


# ---------------------------------------------------- (2) bit-identical parity gate


def test_ray_driver_bit_identical_to_serial_train_grpo() -> None:
    """CORRECTNESS GATE: with a pool that delegates to _reward_matrix, the Ray
    driver must reproduce the serial train_grpo run exactly (shared seeding +
    deterministic reward => identical per-step metrics)."""
    cfg = TrainConfig(
        steps=5, batch_prompts=3, group_size=4, prompt_len=2, horizon=6,
        lr=1e-2, seed=0, d_model=32, n_layers=1, n_heads=4,
    )
    serial = train_grpo(cfg)
    parallel = train_grpo_ray(cfg, reward_pool=_StubPool())

    assert parallel.final_step == serial.final_step
    # Reward endpoints come from the same eval generator + same policy trajectory.
    assert parallel.reward_start == serial.reward_start
    assert parallel.reward_end == serial.reward_end
    assert len(parallel.history) == len(serial.history)
    # Every optimization-relevant metric matches exactly, step for step.
    for ph, sh in zip(parallel.history, serial.history):
        for key in ("loss", "pg_loss", "kl", "reward_mean", "reward_std", "grad_norm", "lr"):
            assert ph[key] == sh[key], f"{key} diverged: {ph[key]} != {sh[key]}"


# ------------------------------------------------------------- (3) timing breakdown


def test_per_step_timing_present_nonneg_and_sums() -> None:
    cfg = TrainConfig(
        steps=3, batch_prompts=2, group_size=3, prompt_len=2, horizon=5,
        lr=1e-2, seed=1, d_model=32, n_layers=1, n_heads=4,
    )
    res = train_grpo_ray(cfg, reward_pool=_StubPool())
    assert len(res.history) == cfg.steps
    for m in res.history:
        for key in ("t_sample", "t_score", "t_learn", "t_step"):
            assert key in m, f"missing timing key {key}"
            assert m[key] >= 0.0, f"{key} negative: {m[key]}"
        # t_step is defined as the sum of the three phases; allow fp slack.
        assert abs(m["t_step"] - (m["t_sample"] + m["t_score"] + m["t_learn"])) < 1e-6

    # Aggregate totals equal the summed per-step phase times.
    assert abs(res.t_sample_total - sum(m["t_sample"] for m in res.history)) < 1e-9
    assert abs(res.t_score_total - sum(m["t_score"] for m in res.history)) < 1e-9
    assert abs(res.t_learn_total - sum(m["t_learn"] for m in res.history)) < 1e-9


# --------------------------------------------------------------- (4) pool injection


def test_reward_pool_is_called_once_per_step_with_group_shape() -> None:
    cfg = TrainConfig(
        steps=4, batch_prompts=2, group_size=3, prompt_len=2, horizon=5,
        lr=1e-2, seed=2, d_model=32, n_layers=1, n_heads=4,
    )
    pool = _StubPool()
    train_grpo_ray(cfg, reward_pool=pool)
    assert pool.calls == cfg.steps  # one scoring call per training step
    assert pool.last_shape == (cfg.batch_prompts, cfg.group_size, cfg.horizon)


def test_on_step_callback_receives_timing() -> None:
    cfg = TrainConfig(
        steps=2, batch_prompts=2, group_size=2, prompt_len=2, horizon=4,
        lr=1e-2, seed=3, d_model=32, n_layers=1, n_heads=4,
    )
    seen: list[tuple[int, dict[str, float]]] = []
    train_grpo_ray(cfg, reward_pool=_StubPool(), on_step=lambda s, m: seen.append((s, m)))
    assert [s for s, _ in seen] == [0, 1]
    assert all("t_score" in m for _, m in seen)


# ---------------------------------------------------- (5) serial path left untouched


def test_default_reward_pool_degrades_to_serial_without_ray() -> None:
    """On a box without the sibling Ray module / Ray installed, the default pool
    must fall back to the serial stub so train_grpo_ray still runs."""
    pool = default_reward_pool()
    # Either the real RayRewardPool (if a sibling shipped it) or the serial stub;
    # in this CPU-only env it is the serial stub. It must satisfy the contract.
    assert hasattr(pool, "score_rollouts")
    if not isinstance(pool, _SerialRewardPool):
        # If a Ray pool was importable, we still must not require a cluster here:
        # only assert the contract method exists, do not call it.
        return
    # Serial stub returns (B, K) matrices matching _reward_matrix.
    cfg = TrainConfig(
        steps=1, batch_prompts=2, group_size=3, prompt_len=2, horizon=4,
        lr=1e-2, seed=4, d_model=32, n_layers=1, n_heads=4,
    )
    codebook, reward, scorer = serial_driver._reward_path(cfg)
    g = torch.Generator().manual_seed(99)
    rollouts = torch.randint(0, codebook.vocab_size, (2, 3, 4), generator=g)
    rew, viol = pool.score_rollouts(rollouts, codebook, reward, scorer)
    ref_rew, ref_viol = serial_driver._reward_matrix(rollouts, codebook, reward, scorer)
    assert torch.equal(rew, ref_rew) and torch.equal(viol, ref_viol)
    assert rew.shape == (2, 3)


def test_serial_train_grpo_still_importable_and_unchanged() -> None:
    """The additive Ray path must not shadow or break the serial entrypoint."""
    from app.policy.driver import train_grpo as serial_train_grpo

    assert serial_train_grpo is train_grpo
    # Default cfg construction of the Ray driver does not perturb the serial one;
    # a tiny serial run still produces the expected result fields.
    cfg = TrainConfig(steps=2, batch_prompts=3, group_size=4, horizon=6, lr=1e-2, d_model=32, n_layers=1)
    res = serial_train_grpo(cfg)
    assert res.final_step == 2
    assert np.isfinite(res.final_metrics["loss"])
    assert "kl" in res.final_metrics
