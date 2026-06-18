"""Unit + mock tests for the Ray-parallel physics reward pool.

These run CPU-only and tiny, with NO live Ray cluster:

* The CORRECTNESS GATE asserts the pool's serial path is bit-identical to
  `driver._reward_matrix` on a fixed seeded ``(B, K, T)`` input.
* A MOCK test fakes the actor ``.score_shard.remote`` with a local function and a
  monkeypatched ``ray.get``, then asserts the dispatch splits ``B * K`` work into
  the expected shards and the merged ``(B, K)`` output still matches the serial
  reference. No cluster is started.
* `n_actors=1` is verified to fall back to serial and match.

The single optional live-cluster test is guarded by a Ray-availability skip and
uses ``ray.init(num_cpus=2, include_dashboard=False)`` (no ``local_mode``; it was
removed in Ray>=2.40) with ``ray.shutdown()`` in the fixture teardown.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from app.policy import ray_reward_pool
from app.policy.driver import _reward_matrix, _reward_path
from app.policy.ray_reward_pool import (
    RayRewardPool,
    RewardPathSpec,
    _assemble,
    _score_flat_shard,
    _shard_bounds,
)

# --------------------------------------------------------------------------- fixtures


def _make_rollouts(b: int = 3, k: int = 4, t: int = 6, seed: int = 1234) -> torch.Tensor:
    """A fixed seeded ``(B, K, T)`` int64 token tensor in the default vocab range."""
    spec = RewardPathSpec()
    vocab = spec.codebook.n_accel * spec.codebook.n_steer  # default 25
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (b, k, t), generator=g)


def _serial_reference(rollouts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the canonical serial `_reward_matrix` on the default reward path.

    Builds the reward path the same way `driver._reward_path` does, via a default
    `RewardPathSpec` (default configs, analytic Pi-DPM proxy on CPU).
    """
    codebook, reward, scorer = RewardPathSpec().build()
    return _reward_matrix(rollouts, codebook, reward, scorer)


# --------------------------------------------------------------------------- pure helpers


def test_shard_bounds_partition_and_balance() -> None:
    # 10 items over 3 shards -> [4, 3, 3], contiguous, covering 0..10 exactly.
    bounds = _shard_bounds(10, 3)
    assert bounds == [(0, 4), (4, 7), (7, 10)]
    # concatenation reproduces range(10)
    rebuilt = [i for (s, e) in bounds for i in range(s, e)]
    assert rebuilt == list(range(10))
    # sizes are as even as possible (max-min <= 1)
    sizes = [e - s for (s, e) in bounds]
    assert max(sizes) - min(sizes) <= 1


def test_shard_bounds_more_shards_than_items_drops_empties() -> None:
    # 2 items over 5 shards -> only 2 non-empty shards, no no-op dispatch.
    bounds = _shard_bounds(2, 5)
    assert bounds == [(0, 1), (1, 2)]


def test_shard_bounds_zero_items_and_bad_shards() -> None:
    assert _shard_bounds(0, 4) == []
    with pytest.raises(ValueError):
        _shard_bounds(5, 0)


def test_assemble_row_major_order() -> None:
    # totals/viols laid out row-major over (B=2, K=3) map to [i, j] as expected.
    totals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    viols = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    rew, viol = _assemble(2, 3, totals, viols)
    assert rew.shape == (2, 3) and viol.shape == (2, 3)
    assert rew[0, 0].item() == 0.0 and rew[1, 2].item() == 5.0
    assert viol[0, 1].item() == 11.0 and viol[1, 0].item() == 13.0


def test_assemble_rejects_wrong_count() -> None:
    with pytest.raises(ValueError):
        _assemble(2, 3, [0.0, 1.0], [0.0, 1.0])


def test_score_flat_shard_matches_trajectory_reward() -> None:
    # A single shard scored by the helper equals the serial path on the same rows.
    rollouts = _make_rollouts(b=2, k=2, t=5, seed=7)
    b, k, t = rollouts.shape
    flat = rollouts.cpu().numpy().reshape(b * k, t)
    codebook, reward, scorer = RewardPathSpec().build()
    totals, viols = _score_flat_shard(flat, codebook, reward, scorer)
    assert len(totals) == b * k and len(viols) == b * k
    rew_ref, viol_ref = _serial_reference(rollouts)
    rew_got, viol_got = _assemble(b, k, totals, viols)
    assert torch.equal(rew_got, rew_ref)
    assert torch.equal(viol_got, viol_ref)


# --------------------------------------------------------------------------- correctness gate


def test_pool_matches_serial_reward_matrix() -> None:
    """CORRECTNESS GATE: pool output is bit-identical to the serial matrix.

    n_actors=1 (or any non-parallel config) routes through the serial path, which
    must equal `driver._reward_matrix` exactly (zero tolerance).
    """
    rollouts = _make_rollouts()
    rew_ref, viol_ref = _serial_reference(rollouts)
    pool = RayRewardPool(n_actors=1)
    rew, viol = pool.score_rollouts(rollouts)
    assert torch.equal(rew, rew_ref)
    assert torch.equal(viol, viol_ref)


def test_pool_output_shapes() -> None:
    rollouts = _make_rollouts(b=3, k=4, t=6)
    pool = RayRewardPool(n_actors=1)
    rew, viol = pool.score_rollouts(rollouts)
    assert rew.shape == (3, 4)
    assert viol.shape == (3, 4)


def test_pool_uses_default_reward_path() -> None:
    """The pool's default spec rebuilds the same path as driver._reward_path."""
    # driver._reward_path takes a TrainConfig; the default codebook config matches.
    from app.policy.driver import TrainConfig

    cb_driver, rew_driver, _ = _reward_path(TrainConfig())
    cb_pool, rew_pool, _ = RewardPathSpec().build()
    assert cb_driver.vocab_size == cb_pool.vocab_size
    assert rew_driver.weights == rew_pool.weights
    assert rew_driver.envelope == rew_pool.envelope


def test_n_actors_one_is_not_parallel() -> None:
    pool = RayRewardPool(n_actors=1)
    assert pool.parallel is False


def test_score_rollouts_rejects_bad_rank() -> None:
    pool = RayRewardPool(n_actors=1)
    with pytest.raises(ValueError):
        pool.score_rollouts(torch.zeros(3, 4))  # 2D, not (B, K, T)


# --------------------------------------------------------------------------- mock (no cluster)


class _FakeRemoteMethod:
    """Stand-in for ``actor.score_shard.remote``: runs locally, records the shard."""

    def __init__(self, codebook, reward, scorer, recorder: list) -> None:
        self._codebook = codebook
        self._reward = reward
        self._scorer = scorer
        self._recorder = recorder

    def remote(self, flat_shard: np.ndarray):
        # Record the exact shard this "actor" was dispatched, then score locally.
        self._recorder.append(np.array(flat_shard, copy=True))
        return _score_flat_shard(flat_shard, self._codebook, self._reward, self._scorer)


class _FakeActor:
    def __init__(self, codebook, reward, scorer, recorder: list) -> None:
        self.score_shard = _FakeRemoteMethod(codebook, reward, scorer, recorder)


def test_mock_pool_splits_into_expected_shards(monkeypatch: pytest.MonkeyPatch) -> None:
    """MOCK: assert dispatch splits B*K work into n_actors shards and merges right.

    No live cluster. We force the pool into its parallel branch, swap the actor
    handles for local fakes whose ``.remote`` records the shard, and monkeypatch
    ``ray.get`` to identity (the fakes already return concrete results). We then
    assert (a) the number of shards, (b) the shard sizes partition B*K, and
    (c) the merged output is bit-identical to the serial reference.
    """
    n_actors = 3
    rollouts = _make_rollouts(b=5, k=2, t=6, seed=99)  # B*K = 10 items
    b, k, _t = rollouts.shape

    pool = RayRewardPool(n_actors=1)  # construct serially so no real cluster spins up
    # Force the parallel branch and install fake actors sharing the pool's path.
    recorder: list = []
    pool._use_ray = True
    pool._actors = [
        _FakeActor(pool._codebook, pool._reward, pool._scorer, recorder)
        for _ in range(n_actors)
    ]

    # ray.get must exist for the parallel path; the fakes return real values, so
    # identity is the correct stand-in. Guard for when ray is not installed.
    if ray_reward_pool.ray is None:
        monkeypatch.setattr(ray_reward_pool, "ray", type("R", (), {"get": staticmethod(lambda x: x)}))
    else:
        monkeypatch.setattr(ray_reward_pool.ray, "get", lambda x: x)

    assert pool.parallel is True
    rew, viol = pool.score_rollouts(rollouts)

    # (a) exactly n_actors shards were dispatched (B*K=10 >= n_actors=3)
    assert len(recorder) == n_actors
    # (b) shard sizes partition B*K exactly, balanced to within one item
    sizes = [s.shape[0] for s in recorder]
    assert sum(sizes) == b * k
    assert max(sizes) - min(sizes) <= 1
    # the concatenation of shards reproduces the flat row-major order
    flat_expected = rollouts.cpu().numpy().reshape(b * k, rollouts.shape[2])
    flat_seen = np.concatenate(recorder, axis=0)
    assert np.array_equal(flat_seen, flat_expected)

    # (c) merged result is bit-identical to the serial reference
    rew_ref, viol_ref = _serial_reference(rollouts)
    assert torch.equal(rew, rew_ref)
    assert torch.equal(viol, viol_ref)


def test_mock_single_shard_when_one_actor(monkeypatch: pytest.MonkeyPatch) -> None:
    """One forced fake actor -> one shard holding all B*K items, still matches."""
    rollouts = _make_rollouts(b=2, k=3, t=5, seed=3)
    b, k, _t = rollouts.shape
    pool = RayRewardPool(n_actors=1)
    recorder: list = []
    pool._use_ray = True
    pool._actors = [_FakeActor(pool._codebook, pool._reward, pool._scorer, recorder)]
    if ray_reward_pool.ray is None:
        monkeypatch.setattr(ray_reward_pool, "ray", type("R", (), {"get": staticmethod(lambda x: x)}))
    else:
        monkeypatch.setattr(ray_reward_pool.ray, "get", lambda x: x)

    rew, viol = pool.score_rollouts(rollouts)
    assert len(recorder) == 1
    assert recorder[0].shape[0] == b * k
    rew_ref, viol_ref = _serial_reference(rollouts)
    assert torch.equal(rew, rew_ref)
    assert torch.equal(viol, viol_ref)


# --------------------------------------------------------------------------- optional live cluster


ray_installed = ray_reward_pool._RAY_AVAILABLE


@pytest.fixture()
def ray_cluster():
    """Tiny 2-CPU local Ray cluster. No local_mode (removed in Ray>=2.40)."""
    import ray

    ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True)
    try:
        yield ray
    finally:
        ray.shutdown()


@pytest.mark.skipif(not ray_installed, reason="Ray not installed in this env")
def test_live_pool_matches_serial(ray_cluster) -> None:  # pragma: no cover - needs Ray
    """End-to-end on a real 2-CPU cluster: parallel output == serial, bit-identical."""
    rollouts = _make_rollouts(b=4, k=3, t=6, seed=21)
    rew_ref, viol_ref = _serial_reference(rollouts)
    pool = RayRewardPool(n_actors=2)
    try:
        assert pool.parallel is True
        rew, viol = pool.score_rollouts(rollouts)
        assert rew.shape == (4, 3) and viol.shape == (4, 3)
        assert torch.equal(rew, rew_ref)
        assert torch.equal(viol, viol_ref)
    finally:
        pool.close()


def test_cost_repeats_preserves_reward_value() -> None:
    """EXP-3 per-item cost knob must change cost, not the reward value.

    The analytic Pi-DPM proxy is deterministic, so scoring a trajectory r times and
    averaging equals the single-pass value. A reward-actor scaling sweep run at
    reward_repeats>1 (to make the score step heavy and provoke the pool crossover)
    must therefore yield the same (B, K) rewards as at repeats=1, only slower. If
    this drifts, the heavy-reward sweep would be comparing different rewards.
    """
    rollouts = _make_rollouts(b=3, k=4, t=6, seed=99)
    cb1, r1, s1 = RewardPathSpec(pidpm_cost_repeats=1).build()
    cb3, r3, s3 = RewardPathSpec(pidpm_cost_repeats=3).build()
    assert s1.cost_repeats == 1 and s3.cost_repeats == 3
    rew1, viol1 = _reward_matrix(rollouts, cb1, r1, s1)
    rew3, viol3 = _reward_matrix(rollouts, cb3, r3, s3)
    assert torch.equal(rew1, rew3)
    assert torch.equal(viol1, viol3)
