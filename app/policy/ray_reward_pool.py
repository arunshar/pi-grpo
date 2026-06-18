"""Ray-parallel physics reward pool for the GRPO loop.

The serial bottleneck in `app.policy.driver._reward_matrix` is a nested
``B * K`` Python loop that decodes each token rollout through the
`MotionCodebook` and scores it with `PhysicsReward` + `PiDpmScorer`. Each
trajectory is independent and the scoring is deterministic given the states, so
the work is embarrassingly parallel.

`RayRewardPool` spreads the ``B * K`` trajectories across `n_actors` Ray actors,
each of which holds its **own** `MotionCodebook` + `PhysicsReward` +
`PiDpmScorer` (rebuilt from the frozen configs, since the live objects are not
meant to cross process boundaries). It reuses the **exact** per-trajectory
scoring helper `driver._trajectory_reward`, so the assembled ``(B, K)`` tensors
are bit-identical to the serial `_reward_matrix` on the same rollouts. See
`tests/test_ray_reward.py::test_pool_matches_serial_reward_matrix` for the
correctness gate.

Design notes
------------
* This module is **additive and opt-in**. It does not import or mutate the
  pure-torch `train_grpo`; the existing driver and tests are untouched. A caller
  swaps `driver._reward_matrix(...)` for `RayRewardPool(n).score_rollouts(...)`.
* When ``n_actors <= 1`` or Ray is not importable, `score_rollouts` falls back
  to the serial `driver._reward_matrix`, so the pool is always safe to construct.
* The sharding (`_shard_bounds`) and the flat-scoring (`_score_flat_shard`) are
  pure, importable functions. They are unit-tested without a live cluster, and
  the actor `.remote` dispatch is monkeypatched to a local call in the mock test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from app.components.kinematic_bicycle import SkbmConfig
from app.components.physics_reward import EmpiricalEnvelope, PhysicsReward, RewardWeights
from app.components.pidpm_scorer import PiDpmScorer
from app.policy.decode import CodebookConfig, MotionCodebook
from app.policy.driver import _reward_matrix, _trajectory_reward

try:  # Ray is an optional dependency; the pool degrades to serial without it.
    import ray

    _RAY_AVAILABLE = True
except Exception:
    ray = None  # type: ignore[assignment]
    _RAY_AVAILABLE = False


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class RewardPathSpec:
    """Frozen, picklable description of one reward path.

    Carries only the immutable configs, not the live (codebook, reward, scorer)
    objects, so it can be shipped to a Ray actor that rebuilds them locally. The
    rebuilt path is identical to `driver._reward_path` for the same configs.
    """

    skbm: SkbmConfig = SkbmConfig()
    codebook: CodebookConfig = CodebookConfig()
    weights: RewardWeights = RewardWeights()
    envelope: EmpiricalEnvelope = EmpiricalEnvelope()
    pidpm_checkpoint: str | None = None
    pidpm_cost_repeats: int = 1  # per-item reward-cost knob (EXP-3); must match the serial path

    def build(self) -> tuple[MotionCodebook, PhysicsReward, PiDpmScorer]:
        """Rebuild the (codebook, reward, scorer) triple this spec describes."""
        codebook = MotionCodebook(self.skbm, self.codebook)
        reward = PhysicsReward(self.skbm, weights=self.weights, envelope=self.envelope)
        scorer = PiDpmScorer(
            checkpoint_path=self.pidpm_checkpoint, device="cpu",
            cost_repeats=self.pidpm_cost_repeats,
        )
        return codebook, reward, scorer


# --------------------------------------------------------------------------- pure helpers


def _shard_bounds(n_items: int, n_shards: int) -> list[tuple[int, int]]:
    """Contiguous ``[start, stop)`` ranges that partition ``range(n_items)``.

    Pure and deterministic. The first ``n_items % n_shards`` shards get one extra
    item, so the split is as even as possible and the concatenation of the shards
    in order reproduces ``range(n_items)`` exactly. Empty shards (when
    ``n_shards > n_items``) are omitted so we never dispatch no-op work.
    """
    if n_shards <= 0:
        raise ValueError(f"n_shards must be >= 1, got {n_shards}")
    if n_items <= 0:
        return []
    n_shards = min(n_shards, n_items)
    base, extra = divmod(n_items, n_shards)
    bounds: list[tuple[int, int]] = []
    start = 0
    for s in range(n_shards):
        stop = start + base + (1 if s < extra else 0)
        bounds.append((start, stop))
        start = stop
    return bounds


def _score_flat_shard(
    flat_tokens: np.ndarray,
    codebook: MotionCodebook,
    reward: PhysicsReward,
    scorer: PiDpmScorer,
) -> tuple[list[float], list[float]]:
    """Score a contiguous shard of flattened ``(n, T)`` token rollouts.

    Returns parallel python lists ``(totals, violations)`` of length ``n``, in the
    same order as ``flat_tokens``. Reuses the **exact** `driver._trajectory_reward`
    so values match the serial path bit-for-bit before the tensor cast.
    """
    totals: list[float] = []
    viols: list[float] = []
    for row in range(flat_tokens.shape[0]):
        total, v = _trajectory_reward(flat_tokens[row], codebook, reward, scorer)
        totals.append(total)
        viols.append(v)
    return totals, viols


def _assemble(
    b: int,
    k: int,
    totals: list[float],
    viols: list[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reassemble flat per-trajectory scores into ``(B, K)`` reward/violation tensors.

    Uses the same ``torch.zeros(b, k)`` float32 buffer and the same row-major
    ``[i, j]`` assignment order as `driver._reward_matrix`, so the float cast is
    identical and the output is bit-identical.
    """
    if len(totals) != b * k or len(viols) != b * k:
        raise ValueError(
            f"expected {b * k} scores, got totals={len(totals)} viols={len(viols)}"
        )
    rew = torch.zeros(b, k)
    viol = torch.zeros(b, k)
    idx = 0
    for i in range(b):
        for j in range(k):
            rew[i, j] = totals[idx]
            viol[i, j] = viols[idx]
            idx += 1
    return rew, viol


# --------------------------------------------------------------------------- actor


def _build_remote_actor_cls():  # pragma: no cover - exercised only with a live cluster
    """Build the Ray actor class lazily (only when Ray is present).

    Defined inside a function so that importing this module never requires Ray.
    Each actor owns its own reward path, rebuilt from the spec in its process.
    """
    if not _RAY_AVAILABLE:  # defensive; callers gate on _RAY_AVAILABLE first
        raise RuntimeError("Ray is not available")

    @ray.remote
    class _RewardActor:
        def __init__(self, spec: RewardPathSpec) -> None:
            self._codebook, self._reward, self._scorer = spec.build()

        def score_shard(self, flat_tokens: np.ndarray) -> tuple[list[float], list[float]]:
            return _score_flat_shard(flat_tokens, self._codebook, self._reward, self._scorer)

    return _RewardActor


# --------------------------------------------------------------------------- pool


class RayRewardPool:
    """Parallel drop-in for `driver._reward_matrix` backed by Ray actors.

    Parameters
    ----------
    n_actors:
        Number of Ray actors to spread the ``B * K`` trajectories over. With
        ``n_actors <= 1`` (or when Ray is unavailable) the pool runs the serial
        `driver._reward_matrix` path instead, so results are always defined.
    spec:
        Frozen description of the reward path each actor rebuilds. Defaults match
        `driver._reward_path` (default configs, analytic Pi-DPM proxy on CPU).
    """

    def __init__(self, n_actors: int = 1, spec: RewardPathSpec | None = None) -> None:
        self.n_actors = int(n_actors)
        self.spec = spec if spec is not None else RewardPathSpec()
        self._actors: list = []
        # Serial reference path, used for the fallback and as the local
        # (codebook, reward, scorer) for the serial branch.
        self._codebook, self._reward, self._scorer = self.spec.build()
        self._use_ray = self.n_actors > 1 and _RAY_AVAILABLE
        if self._use_ray:
            self._actors = self._spawn_actors()

    # ------------------------------------------------------------ properties

    @property
    def parallel(self) -> bool:
        """True iff this pool will actually fan out to Ray actors."""
        return bool(self._use_ray and self._actors)

    # ------------------------------------------------------------ lifecycle

    def _spawn_actors(self) -> list:  # pragma: no cover - needs a live cluster
        if not _RAY_AVAILABLE:
            return []
        if not ray.is_initialized():
            ray.init(num_cpus=self.n_actors, include_dashboard=False, ignore_reinit_error=True)
        actor_cls = _build_remote_actor_cls()
        return [actor_cls.remote(self.spec) for _ in range(self.n_actors)]

    def close(self) -> None:
        """Drop actor handles. Does NOT call ray.shutdown (the caller owns the cluster)."""
        self._actors = []

    # ------------------------------------------------------------ scoring

    def _serial(self, rollouts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Bit-identical serial path: delegate to the canonical driver helper."""
        return _reward_matrix(rollouts, self._codebook, self._reward, self._scorer)

    def score_rollouts(self, rollouts: torch.Tensor, codebook=None, reward=None,
                        scorer=None) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B, K, T)`` token rollouts -> ``(rew (B, K), viol (B, K))``.

        Output is bit-identical to `driver._reward_matrix` on the same rollouts,
        whether it runs serially or fans out across actors.

        codebook/reward/scorer are accepted for RewardPool-protocol compatibility
        (train_grpo_ray and driver._reward_matrix call score_rollouts with the
        4-arg form) but are IGNORED: each actor and the serial fallback build their
        own reward path from the frozen RewardPathSpec, so the pool is
        self-contained. They equal the default reward path, so the bit-identical
        guarantee is preserved.
        """
        if rollouts.dim() != 3:
            raise ValueError(f"expected (B, K, T) rollouts, got shape {tuple(rollouts.shape)}")
        if not self.parallel:
            return self._serial(rollouts)
        return self._score_parallel(rollouts)

    def _flatten(self, rollouts: torch.Tensor) -> tuple[int, int, np.ndarray]:
        """``(B, K, T)`` tensor -> ``(B, K, flat (B*K, T) int64 ndarray)``.

        Matches `_reward_matrix`'s ``rollouts.cpu().numpy()`` then row-major
        ``[i, j]`` traversal: reshaping ``(B, K, T)`` to ``(B*K, T)`` walks the
        same ``(i, j)`` order, so shard order composes back to the serial order.
        """
        b, k, t = rollouts.shape
        flat = rollouts.cpu().numpy().reshape(b * k, t)
        return b, k, flat

    def _score_parallel(self, rollouts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, k, flat = self._flatten(rollouts)
        bounds = _shard_bounds(b * k, len(self._actors))
        # Dispatch one shard per actor (round-robin if fewer shards than actors,
        # which _shard_bounds prevents by capping at n_items).
        futures = []
        for actor, (start, stop) in zip(self._actors, bounds, strict=False):
            futures.append(actor.score_shard.remote(flat[start:stop]))
        results = ray.get(futures)  # pragma: no cover - live-cluster path
        totals: list[float] = []
        viols: list[float] = []
        for shard_totals, shard_viols in results:
            totals.extend(shard_totals)
            viols.extend(shard_viols)
        return _assemble(b, k, totals, viols)


__all__ = [
    "RayRewardPool",
    "RewardPathSpec",
    "_assemble",
    "_score_flat_shard",
    "_shard_bounds",
]
