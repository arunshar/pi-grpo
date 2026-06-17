"""Unit + mock tests for the async/off-policy staleness harness.

All CPU-only, tiny, and fast. NO live Ray cluster and NO real async runtime: the
producer/consumer is driven in-process by a deterministic schedule, and the queue
admission logic is pure Python. What is pinned down here:

* `compute_staleness_lag` is a pure integer function: correct lag, and it rejects
  impossible (negative / future) version pairs.
* `staleness_stats` reduces a lag list correctly (mean / max / on-policy share).
* `BoundedStalenessQueue` enforces BOTH the capacity bound and the staleness
  bound, under each `QueueFullPolicy`, exercised with a mock producer/consumer.
* The on-policy reduction: `max_staleness == 0` makes every consumed batch lag 0
  and reproduces the synchronous `train_grpo` reward trajectory bit-for-bit.
"""

from __future__ import annotations

import pytest

from app.policy.driver import TrainConfig, train_grpo
from app.policy.ray_staleness import (
    BoundedStalenessQueue,
    QueueFullPolicy,
    StaleBatch,
    StalenessConfig,
    compute_staleness_lag,
    run_staleness_comparison,
    staleness_stats,
    train_grpo_async_staleness,
)

# A tiny config: a few steps, small model, so the whole suite runs in seconds.
_TINY = TrainConfig(steps=4, batch_prompts=2, group_size=4, horizon=6, d_model=32, n_layers=1)


# ============================================================ staleness metric (pure)


def test_compute_staleness_lag_basic() -> None:
    assert compute_staleness_lag(rollout_version=0, learner_version=0) == 0
    assert compute_staleness_lag(rollout_version=3, learner_version=7) == 4
    assert compute_staleness_lag(rollout_version=10, learner_version=10) == 0


def test_compute_staleness_lag_rejects_negative_versions() -> None:
    with pytest.raises(ValueError):
        compute_staleness_lag(-1, 5)
    with pytest.raises(ValueError):
        compute_staleness_lag(2, -3)


def test_compute_staleness_lag_rejects_future_rollout() -> None:
    # A rollout produced under weights the learner has not reached is impossible.
    with pytest.raises(ValueError):
        compute_staleness_lag(rollout_version=8, learner_version=5)


def test_staleness_stats_summary() -> None:
    stats = staleness_stats([0, 0, 2, 4])
    assert stats.count == 4
    assert stats.mean_lag == pytest.approx(1.5)
    assert stats.max_lag == 4
    assert stats.on_policy_fraction == pytest.approx(0.5)


def test_staleness_stats_empty_is_all_zero() -> None:
    stats = staleness_stats([])
    assert stats.count == 0
    assert stats.mean_lag == 0.0
    assert stats.max_lag == 0
    assert stats.on_policy_fraction == 0.0


def test_staleness_stats_all_on_policy() -> None:
    stats = staleness_stats([0, 0, 0])
    assert stats.on_policy_fraction == 1.0
    assert stats.max_lag == 0


def test_staleness_stats_rejects_negative() -> None:
    with pytest.raises(ValueError):
        staleness_stats([0, -1, 2])


# ============================================================ bounded queue (mock prod/cons)


def test_queue_rejects_constructor_garbage() -> None:
    with pytest.raises(ValueError):
        BoundedStalenessQueue(maxsize=0)
    with pytest.raises(ValueError):
        BoundedStalenessQueue(max_staleness=-1)
    with pytest.raises(ValueError):
        BoundedStalenessQueue(policy="nonsense")


def test_queue_staleness_gate_rejects_too_old() -> None:
    """A batch staler than max_staleness at enqueue is rejected before capacity."""
    q = BoundedStalenessQueue(maxsize=4, max_staleness=1)
    # producer at version 2, learner at version 5 -> lag 3 > max_staleness 1
    admitted = q.try_put(StaleBatch(policy_version=2), learner_version=5)
    assert admitted is False
    assert q.n_rejected_stale == 1
    assert len(q) == 0


def test_queue_admits_within_staleness_bound() -> None:
    q = BoundedStalenessQueue(maxsize=4, max_staleness=2)
    assert q.try_put(StaleBatch(policy_version=3), learner_version=5) is True  # lag 2 == bound
    assert q.try_put(StaleBatch(policy_version=5), learner_version=5) is True  # lag 0
    assert len(q) == 2
    assert q.n_admitted == 2


def test_queue_drop_new_when_full() -> None:
    """A mock producer over-produces; DROP_NEW rejects the overflow batch."""
    q = BoundedStalenessQueue(maxsize=2, max_staleness=10, policy=QueueFullPolicy.DROP_NEW)
    for v in (0, 1):
        assert q.try_put(StaleBatch(policy_version=v), learner_version=1) is True
    # queue full -> the next put is rejected, queue unchanged
    assert q.try_put(StaleBatch(policy_version=1), learner_version=1) is False
    assert q.n_rejected_full == 1
    assert len(q) == 2
    # the two retained batches are the FIRST two (FIFO, oldest kept)
    assert q.get().policy_version == 0
    assert q.get().policy_version == 1


def test_queue_drop_oldest_when_full() -> None:
    """DROP_OLDEST evicts the front (stalest) batch to admit the newcomer."""
    q = BoundedStalenessQueue(maxsize=2, max_staleness=10, policy=QueueFullPolicy.DROP_OLDEST)
    q.try_put(StaleBatch(policy_version=0, payload="a"), learner_version=2)
    q.try_put(StaleBatch(policy_version=1, payload="b"), learner_version=2)
    # full now; admitting "c" evicts "a"
    assert q.try_put(StaleBatch(policy_version=2, payload="c"), learner_version=2) is True
    assert q.n_evicted == 1
    assert len(q) == 2
    assert q.get().payload == "b"  # oldest survivor
    assert q.get().payload == "c"


def test_queue_wait_when_full() -> None:
    q = BoundedStalenessQueue(maxsize=1, max_staleness=10, policy=QueueFullPolicy.WAIT)
    assert q.try_put(StaleBatch(policy_version=0), learner_version=0) is True
    assert q.try_put(StaleBatch(policy_version=0), learner_version=0) is False
    assert q.n_wait == 1
    assert len(q) == 1


def test_queue_get_empty_raises() -> None:
    q = BoundedStalenessQueue()
    with pytest.raises(IndexError):
        q.get()


def test_queue_drain_stale_evicts_aged_batches() -> None:
    """Batches fresh at enqueue can age past the bound; drain_stale removes them."""
    q = BoundedStalenessQueue(maxsize=4, max_staleness=1)
    q.try_put(StaleBatch(policy_version=3), learner_version=3)  # lag 0 at enqueue
    q.try_put(StaleBatch(policy_version=4), learner_version=4)  # lag 0 at enqueue
    # learner advances to 5: the v=3 batch is now lag 2 > 1, v=4 is lag 1 == bound
    evicted = q.drain_stale(learner_version=5)
    assert evicted == 1
    assert len(q) == 1
    assert q.get().policy_version == 4


def test_queue_mock_producer_consumer_roundtrip() -> None:
    """End-to-end mock: a producer fills, a consumer drains in FIFO order."""
    q = BoundedStalenessQueue(maxsize=3, max_staleness=5, policy=QueueFullPolicy.DROP_OLDEST)
    # mock producer pushes 3 batches under increasing versions, learner at 2
    produced = [StaleBatch(policy_version=v, payload=v) for v in (0, 1, 2)]
    for b in produced:
        assert q.try_put(b, learner_version=2) is True
    # mock consumer pulls all, FIFO, recording lags against learner version 2
    lags = []
    while not q.empty:
        b = q.get()
        lags.append(compute_staleness_lag(b.policy_version, 2))
    assert lags == [2, 1, 0]
    assert staleness_stats(lags).max_lag == 2


# ============================================================ on-policy reduction


def test_on_policy_reduction_all_lags_zero() -> None:
    """max_staleness == 0 => every consumed batch is exactly on-policy (lag 0)."""
    result = train_grpo_async_staleness(
        _TINY, StalenessConfig(max_staleness=0, queue_maxsize=1, producer_ahead=1)
    )
    assert result.final_step == _TINY.steps           # learner stepped every iter
    assert result.staleness.max_lag == 0
    assert result.staleness.on_policy_fraction == 1.0
    assert result.queue_rejected_stale == 0
    assert all(m["lag"] == 0 for m in result.history)


def test_on_policy_reduction_matches_serial_train_grpo() -> None:
    """max_staleness == 0 reproduces the synchronous train_grpo reward trajectory.

    Same seed, same prompts, same model, same SAMPLE->SCORE->LEARN order, so the
    per-step reward_mean sequence must match bit-for-bit and the final reward must
    be identical.
    """
    serial = train_grpo(_TINY)
    asyncv = train_grpo_async_staleness(
        _TINY, StalenessConfig(max_staleness=0, queue_maxsize=1, producer_ahead=1)
    )
    assert len(asyncv.history) == len(serial.history)
    for a, s in zip(asyncv.history, serial.history, strict=True):
        assert a["reward_mean"] == pytest.approx(s["reward_mean"], abs=0.0)
        assert a["loss"] == pytest.approx(s["loss"], abs=0.0)
        assert a["kl"] == pytest.approx(s["kl"], abs=0.0)
    assert asyncv.reward_start == pytest.approx(serial.reward_start, abs=0.0)
    assert asyncv.reward_end == pytest.approx(serial.reward_end, abs=0.0)


# ============================================================ async run (off-policy)


def test_async_run_admits_offpolicy_batches() -> None:
    """With max_staleness > 0 the producer may run ahead; the run still completes."""
    result = train_grpo_async_staleness(
        _TINY, StalenessConfig(max_staleness=2, queue_maxsize=4, producer_ahead=2)
    )
    assert result.final_step >= 1
    assert result.staleness.count >= 1
    assert result.wall_time >= 0.0
    # producer_ahead=2 over the run produces more batches than the learner consumes
    assert result.queue_admitted >= result.final_step


def test_comparison_reports_throughput_and_lag() -> None:
    """run_staleness_comparison wires baseline vs async and computes the deltas."""
    cmp = run_staleness_comparison(_TINY, max_staleness=2)
    # baseline is the on-policy reduction: all lag 0
    assert cmp.baseline.staleness.max_lag == 0
    # throughput fields are finite numbers (>= 0)
    assert cmp.throughput_baseline >= 0.0
    assert cmp.throughput_async >= 0.0
    # the comparison exposes the honest signals
    assert cmp.mean_lag_async == cmp.async_run.staleness.mean_lag
    assert cmp.reward_gap == pytest.approx(
        cmp.async_run.reward_end - cmp.baseline.reward_end
    )
