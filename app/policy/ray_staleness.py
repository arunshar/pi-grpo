"""OPT-IN async / off-policy staleness experiment harness for GRPO.

This module is an *additive* experiment harness. It does NOT modify or replace
the pure-torch `train_grpo` (`app.policy.driver`) or the Ray-parallel
`train_grpo_ray` (`app.policy.ray_driver`); it imports their building blocks and
the reward-pool contract verbatim and wires them into an *asynchronous,
off-policy* loop where rollout production runs AHEAD of the learner.

What this harness studies
-------------------------
In the synchronous GRPO loop (the serial driver and `train_grpo_ray`), each step
is strictly SAMPLE -> SCORE -> LEARN: the learner blocks until the current
rollouts are scored, and rollouts are always exactly on-policy (sampled under the
same parameters the update is about to change). That serialization is simple and
exactly on-policy, but the learner sits idle while rollouts are produced and
scored.

An asynchronous design lets rollout *actors* run ahead of the learner: they keep
sampling and scoring batches under whatever policy weights they last received and
push the results into a bounded queue, while the learner pulls a batch, updates,
and (periodically) broadcasts fresh weights back. The learner never blocks on
production, so throughput (updates per wall-second) can rise. The COST is that a
batch pulled at learner-version V may have been produced at an earlier
policy-version V - lag, so the update is OFF-POLICY by `lag` versions. That
staleness inflates the importance-sampling ratio variance and the off-policy KL,
which is exactly the quantity GRPO already tracks. This harness measures both
sides: the throughput gain AND the staleness / off-policy KL cost.

HONEST SCOPE / TODO
-------------------
On the *tiny* motion-primitive policy used in this project (d_model <= 64, a
couple of layers, a CPU forward over a ~25-token vocab), there is no real wall
clock advantage to extract: sampling, scoring, and the learner step are all
sub-millisecond, and Python-level async bookkeeping plus any Ray actor scheduling
will dominate. So on this model the harness is HONESTLY a *systems demonstration*
of the async/off-policy machinery and of how to *quantify* the staleness/KL cost,
not a throughput-win claim. The staleness-versus-reward trade is real and
measurable even here (a larger `max_staleness` does admit more off-policy
batches), but the throughput numbers only become meaningful when:

  * the policy is large (a real LLM / vLLM-backed generator) so per-prompt
    `generate` dominates, and/or
  * the reward path is expensive enough that the learner would otherwise idle.

  TODO(perf): re-run `StalenessComparison` on a large policy + the real Ray
  reward pool / vLLM generator before making any throughput-gain claim. On the
  tiny model report only the staleness/KL accounting and the reduction-to-serial
  parity, not speedups.
  TODO(ray): this reference harness runs the producer/consumer *in-process*
  (a single thread stepping a deterministic schedule) so it is testable on the
  login node with no cluster. A genuine Ray version would make each producer a
  `@ray.remote` actor holding its own reward path (reuse `RayRewardPool`) and the
  weight broadcast a `ray.put` of the learner state-dict. The queue logic
  (`BoundedStalenessQueue`) and the staleness metric (`compute_staleness_lag`)
  are written to port unchanged to that setting.

The on-policy reduction
-----------------------
With `max_staleness == 0` the harness is REQUIRED to reduce to the synchronous
path: a producer may only enqueue a batch sampled at the current learner version,
so every batch the learner consumes has lag 0, the queue holds at most one
in-flight batch, and the sequence of updates matches a synchronous SAMPLE ->
SCORE -> LEARN loop on the same seed. `tests/test_ray_staleness.py` pins this:
the `max_staleness=0` schedule produces the exact same per-step reward trajectory
as the serial `train_grpo`-style loop the harness falls back to.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import torch

from app.policy.driver import (
    TrainConfig,
    _mean_rollout_reward,
    _policy_config,
    _reward_path,
)
from app.policy.model import build_policy_pair
from app.policy.ray_driver import RewardPool, _SerialRewardPool
from app.trainers.grpo_trainer import GrpoBatch, GrpoConfig, GrpoTrainer

__all__ = [
    "AsyncTrainResult",
    "BoundedStalenessQueue",
    "QueueFullPolicy",
    "StaleBatch",
    "StalenessComparison",
    "StalenessConfig",
    "StalenessStats",
    "compute_staleness_lag",
    "run_staleness_comparison",
    "staleness_stats",
    "train_grpo_async_staleness",
]


# ===================================================================== staleness metric
# Pure functions: no torch, no Ray, no I/O. These are the unit-tested core of the
# "what does staleness cost" measurement.


def compute_staleness_lag(rollout_version: int, learner_version: int) -> int:
    """Policy-version lag of a rollout relative to the learner.

    A rollout was produced under policy parameters at version ``rollout_version``;
    the learner is about to apply it at version ``learner_version``. The
    *staleness lag* is how many learner updates have happened since the rollout's
    policy snapshot:

        lag = learner_version - rollout_version

    ``lag == 0`` means the rollout is exactly on-policy (it was sampled under the
    weights the learner currently holds). ``lag > 0`` means off-policy by ``lag``
    updates. The function is pure and integer-only so it ports unchanged to a Ray
    actor that only knows the two version stamps.

    Raises
    ------
    ValueError
        If either version is negative, or if ``rollout_version`` is in the future
        relative to the learner (negative lag), which signals a bookkeeping bug
        (a rollout cannot be produced under weights the learner has not reached).
    """
    if rollout_version < 0 or learner_version < 0:
        raise ValueError(
            f"versions must be >= 0, got rollout={rollout_version} learner={learner_version}"
        )
    lag = learner_version - rollout_version
    if lag < 0:
        raise ValueError(
            f"negative lag {lag}: rollout_version {rollout_version} is ahead of "
            f"learner_version {learner_version} (impossible)"
        )
    return lag


@dataclass(frozen=True)
class StalenessStats:
    """Summary of the per-consumed-batch staleness lags over a run."""

    count: int
    mean_lag: float
    max_lag: int
    on_policy_fraction: float  # fraction of consumed batches with lag == 0


def staleness_stats(lags: list[int]) -> StalenessStats:
    """Reduce a list of per-batch lags into a `StalenessStats` summary.

    Pure. ``on_policy_fraction`` is the share of batches with ``lag == 0`` (the
    synchronous, exactly-on-policy batches). An empty list yields an all-zero
    summary so callers need not special-case a run that consumed nothing.
    """
    for lag in lags:
        if lag < 0:
            raise ValueError(f"lags must be >= 0, got {lag}")
    n = len(lags)
    if n == 0:
        return StalenessStats(count=0, mean_lag=0.0, max_lag=0, on_policy_fraction=0.0)
    on_policy = sum(1 for lag in lags if lag == 0)
    return StalenessStats(
        count=n,
        mean_lag=sum(lags) / n,
        max_lag=max(lags),
        on_policy_fraction=on_policy / n,
    )


# ===================================================================== bounded queue
# The drop/wait policy of the staleness queue. Tested with a mock producer/
# consumer, no Ray, no torch.


class QueueFullPolicy:
    """Behavior when a producer enqueues into a full bounded queue."""

    DROP_NEW = "drop_new"      # reject the incoming batch (producer must back off)
    DROP_OLDEST = "drop_oldest"  # evict the stalest queued batch to make room
    WAIT = "wait"              # signal the producer to wait (no room right now)


@dataclass
class StaleBatch:
    """One produced-but-not-yet-consumed unit of work in the queue.

    ``policy_version`` is the learner version the producer's weights were at when
    it SAMPLED this batch; the learner uses it with `compute_staleness_lag` to get
    the lag at consume time. ``payload`` is opaque to the queue (in the harness it
    carries the rollouts and their scored rewards / old log-probs).
    """

    policy_version: int
    payload: object = None


class BoundedStalenessQueue:
    """A bounded FIFO queue with an explicit staleness admission rule.

    Two independent bounds:

    * ``maxsize``: capacity (how many produced batches may wait at once). This is
      the systems back-pressure knob: a small queue keeps the learner close to
      on-policy; a large queue decouples producer and learner for throughput.
    * ``max_staleness``: the maximum lag (``learner_version - batch.policy_version``)
      a batch may have AT ENQUEUE time and still be admitted. ``max_staleness == 0``
      means only exactly-on-policy batches are accepted, which (with ``maxsize == 1``)
      reduces the whole harness to the synchronous path.

    The full-queue behavior is set by ``policy`` (`QueueFullPolicy`). ``try_put``
    returns whether the batch was admitted; the learner side calls ``get`` to pull
    the oldest admitted batch. None of this needs Ray: a real Ray version would
    back this with an actor-owned queue, but the admission logic is identical.
    """

    def __init__(
        self,
        maxsize: int = 4,
        max_staleness: int = 2,
        policy: str = QueueFullPolicy.DROP_OLDEST,
    ) -> None:
        if maxsize < 1:
            raise ValueError(f"maxsize must be >= 1, got {maxsize}")
        if max_staleness < 0:
            raise ValueError(f"max_staleness must be >= 0, got {max_staleness}")
        if policy not in (QueueFullPolicy.DROP_NEW, QueueFullPolicy.DROP_OLDEST, QueueFullPolicy.WAIT):
            raise ValueError(f"unknown full-queue policy {policy!r}")
        self.maxsize = int(maxsize)
        self.max_staleness = int(max_staleness)
        self.policy = policy
        self._q: deque[StaleBatch] = deque()
        # Counters for the experiment accounting.
        self.n_admitted = 0
        self.n_rejected_full = 0
        self.n_rejected_stale = 0
        self.n_evicted = 0
        self.n_wait = 0

    def __len__(self) -> int:
        return len(self._q)

    @property
    def full(self) -> bool:
        return len(self._q) >= self.maxsize

    @property
    def empty(self) -> bool:
        return len(self._q) == 0

    def try_put(self, batch: StaleBatch, learner_version: int) -> bool:
        """Attempt to enqueue ``batch``; return True iff it was admitted.

        Admission has two gates, in order:

        1. Staleness gate: if the batch's lag at this moment exceeds
           ``max_staleness`` it is rejected outright (too off-policy to be worth
           queuing). ``lag`` is computed with the shared `compute_staleness_lag`,
           so the same rule the learner uses to MEASURE lag is what GATES it here.
        2. Capacity gate: if the queue is full, the configured `QueueFullPolicy`
           decides. DROP_NEW rejects this batch; DROP_OLDEST evicts the front
           (stalest) batch then admits; WAIT rejects and records a wait (the
           producer is expected to retry after the learner drains one).
        """
        lag = compute_staleness_lag(batch.policy_version, learner_version)
        if lag > self.max_staleness:
            self.n_rejected_stale += 1
            return False

        if self.full:
            if self.policy == QueueFullPolicy.DROP_NEW:
                self.n_rejected_full += 1
                return False
            if self.policy == QueueFullPolicy.WAIT:
                self.n_wait += 1
                return False
            # DROP_OLDEST: evict the front to make room.
            self._q.popleft()
            self.n_evicted += 1

        self._q.append(batch)
        self.n_admitted += 1
        return True

    def get(self) -> StaleBatch:
        """Pop and return the oldest admitted batch. Raises if empty."""
        if self.empty:
            raise IndexError("get from an empty BoundedStalenessQueue")
        return self._q.popleft()

    def drain_stale(self, learner_version: int) -> int:
        """Evict any queued batch that is now staler than ``max_staleness``.

        Called by the learner AFTER an update bumps its version: batches that
        were fresh enough at enqueue time may have aged past the bound. Returns
        the number evicted. Keeps the queue's staleness invariant honest so the
        consumed lags never silently exceed ``max_staleness``.
        """
        evicted = 0
        kept: deque[StaleBatch] = deque()
        for b in self._q:
            if compute_staleness_lag(b.policy_version, learner_version) > self.max_staleness:
                evicted += 1
            else:
                kept.append(b)
        self._q = kept
        self.n_evicted += evicted
        return evicted


# ===================================================================== async harness


@dataclass(frozen=True)
class StalenessConfig:
    """Knobs for the async/off-policy staleness experiment (on top of `TrainConfig`).

    ``max_staleness == 0`` forces the on-policy reduction. ``producer_ahead`` is
    how many batches a producer is allowed to run ahead of the learner per step
    in this in-process schedule (the analogue of actor concurrency); it only has
    an effect when ``max_staleness > 0``.
    """

    max_staleness: int = 2
    queue_maxsize: int = 4
    queue_policy: str = QueueFullPolicy.DROP_OLDEST
    producer_ahead: int = 1
    refresh_every: int = 1   # learner broadcasts weights to producers every N updates


@dataclass
class AsyncTrainResult:
    """Result of one async-staleness run.

    ``history`` holds per-learner-step metrics (the GRPO metrics dict plus the
    consumed batch's ``lag`` and the phase timings). ``staleness`` is the summary
    over all consumed lags. ``wall_time`` is the measured end-to-end seconds for
    the run (so a comparison can report throughput = steps / wall_time).
    """

    final_step: int
    final_metrics: dict
    history: list
    reward_start: float
    reward_end: float
    staleness: StalenessStats
    wall_time: float
    queue_admitted: int
    queue_rejected_stale: int
    queue_evicted: int


@dataclass
class StalenessComparison:
    """Side-by-side of the on-policy baseline and the async/off-policy run."""

    baseline: AsyncTrainResult       # max_staleness == 0 (synchronous reduction)
    async_run: AsyncTrainResult      # max_staleness > 0
    throughput_baseline: float = field(init=False)  # learner steps / wall second
    throughput_async: float = field(init=False)
    throughput_ratio: float = field(init=False)      # async / baseline
    mean_lag_async: float = field(init=False)
    reward_gap: float = field(init=False)             # async reward_end - baseline reward_end

    def __post_init__(self) -> None:
        self.throughput_baseline = _safe_throughput(self.baseline)
        self.throughput_async = _safe_throughput(self.async_run)
        self.throughput_ratio = (
            self.throughput_async / self.throughput_baseline
            if self.throughput_baseline > 0
            else float("nan")
        )
        self.mean_lag_async = self.async_run.staleness.mean_lag
        self.reward_gap = self.async_run.reward_end - self.baseline.reward_end


def _safe_throughput(r: AsyncTrainResult) -> float:
    return r.final_step / r.wall_time if r.wall_time > 0 else 0.0


def _setup(cfg: TrainConfig):
    """Build the shared (policy, ref, trainer, prompts, reward path) for a run.

    Identical construction to `driver.train_grpo` / `ray_driver.train_grpo_ray`
    (same seeds => same prompts => directly comparable), so a `max_staleness == 0`
    async run reproduces the synchronous trajectory.
    """
    torch.manual_seed(cfg.seed)
    codebook, reward, scorer = _reward_path(cfg)
    policy, ref = build_policy_pair(_policy_config(cfg, codebook.vocab_size))
    trainer = GrpoTrainer(
        policy=policy,
        ref_policy=ref,
        cfg=GrpoConfig(group_size=cfg.group_size, lr=cfg.lr, total_steps=cfg.steps),
    )
    gen = torch.Generator().manual_seed(cfg.seed + 1)
    prompts = torch.randint(
        0, codebook.vocab_size, (cfg.batch_prompts, cfg.prompt_len), generator=gen
    )
    return policy, ref, trainer, prompts, gen, (codebook, reward, scorer)


def _sample_and_score(
    policy,
    prompts: torch.Tensor,
    cfg: TrainConfig,
    reward_pool: RewardPool,
    path,
    gen: torch.Generator,
):
    """SAMPLE + SCORE for one batch. Returns (rollouts, logp_old, rew).

    The reward-pool contract is the one from `ray_driver`: the default
    `_SerialRewardPool.score_rollouts(rollouts, codebook, reward, scorer)` takes
    the path objects explicitly, so this harness passes them through and works
    with any pool that honors that 4-argument signature.
    """
    codebook, reward, scorer = path
    rollouts, logp_old = policy.generate(
        prompts,
        k=cfg.group_size,
        max_new_tokens=cfg.horizon,
        temperature=cfg.temperature,
        generator=gen,
    )
    rew, _viol = reward_pool.score_rollouts(rollouts, codebook, reward, scorer)
    return rollouts, logp_old, rew


def train_grpo_async_staleness(
    cfg: TrainConfig = TrainConfig(),
    staleness: StalenessConfig = StalenessConfig(),
    reward_pool: RewardPool | None = None,
) -> AsyncTrainResult:
    """Async / off-policy GRPO with a bounded staleness queue (in-process schedule).

    Single-threaded but ASYNCHRONOUS IN STRUCTURE: each learner step, a producer
    runs ``producer_ahead`` SAMPLE+SCORE batches under its current (possibly
    stale) weight snapshot and offers them to a `BoundedStalenessQueue`; the
    learner then pulls one admitted batch, computes its lag, and applies a GRPO
    update. The producer's weight snapshot is refreshed every
    ``refresh_every`` learner updates. This mirrors a Ray producer/learner split
    without needing a cluster, so it runs on the login node (see module TODO for
    the real Ray version).

    With ``staleness.max_staleness == 0`` (and the queue then naturally holding a
    single on-policy batch) this is exactly the synchronous loop: every consumed
    batch has lag 0 and the update sequence matches `driver.train_grpo` on the
    same seed.
    """
    if reward_pool is None:
        reward_pool = _SerialRewardPool()

    policy, ref, trainer, prompts, gen, path = _setup(cfg)
    codebook, reward, scorer = path

    eval_gen = torch.Generator().manual_seed(cfg.seed + 7)
    reward_start = _mean_rollout_reward(policy, prompts, cfg, codebook, reward, scorer, eval_gen)

    queue = BoundedStalenessQueue(
        maxsize=staleness.queue_maxsize,
        max_staleness=staleness.max_staleness,
        policy=staleness.queue_policy,
    )

    history: list = []
    metrics: dict = {}
    consumed_lags: list[int] = []

    # ``producer_version`` is the learner version the producer's weight snapshot
    # is at. In this in-process harness the producer shares the live ``policy``
    # object, so a batch's true policy version is whatever the learner version was
    # the last time the producer refreshed; we stamp it explicitly.
    producer_version = 0
    wall_t0 = time.perf_counter()

    for _step in range(cfg.steps):
        learner_version = trainer.step  # 0-based; bumps after each step_update

        # ---- PRODUCE: run ahead and offer batches to the queue ----
        # On-policy (max_staleness == 0) admits exactly one batch per step; a
        # larger bound lets the producer get ahead.
        ahead = staleness.producer_ahead if staleness.max_staleness > 0 else 1
        for _ in range(ahead):
            rollouts, logp_old, rew = _sample_and_score(policy, prompts, cfg, reward_pool, path, gen)
            ref_logp = ref.log_prob_token(prompts, rollouts).detach()
            batch = StaleBatch(
                policy_version=producer_version,
                payload=(rollouts, logp_old.detach(), rew, ref_logp),
            )
            queue.try_put(batch, learner_version)

        # ---- CONSUME: learner pulls one admitted batch and updates ----
        if queue.empty:
            # No admissible batch (all too stale): skip this learner step. This is
            # an honest outcome of an aggressive staleness bound, recorded so the
            # study can see the learner starved.
            continue
        stale_batch = queue.get()
        lag = compute_staleness_lag(stale_batch.policy_version, learner_version)
        consumed_lags.append(lag)

        rollouts, logp_old, rew, ref_logp = stale_batch.payload
        grpo_batch = GrpoBatch(
            prompt_ids=prompts,
            rollout_ids=rollouts,
            action_logp_old=logp_old,
            rewards=rew,
            ref_logp=ref_logp,
        )
        metrics = dict(trainer.step_update(grpo_batch))
        metrics["lag"] = lag
        history.append(metrics)

        # ---- BROADCAST: refresh the producer's weight snapshot ----
        if (trainer.step % staleness.refresh_every) == 0:
            producer_version = trainer.step
        # Evict anything that aged past the bound after this update.
        queue.drain_stale(trainer.step)

    wall_time = time.perf_counter() - wall_t0

    eval_gen = torch.Generator().manual_seed(cfg.seed + 7)
    reward_end = _mean_rollout_reward(policy, prompts, cfg, codebook, reward, scorer, eval_gen)

    return AsyncTrainResult(
        final_step=len(history),
        final_metrics=metrics,
        history=history,
        reward_start=reward_start,
        reward_end=reward_end,
        staleness=staleness_stats(consumed_lags),
        wall_time=wall_time,
        queue_admitted=queue.n_admitted,
        queue_rejected_stale=queue.n_rejected_stale,
        queue_evicted=queue.n_evicted,
    )


def run_staleness_comparison(
    cfg: TrainConfig = TrainConfig(),
    max_staleness: int = 2,
    reward_pool: RewardPool | None = None,
) -> StalenessComparison:
    """Run the on-policy baseline and the async/off-policy variant, side by side.

    The baseline uses ``max_staleness == 0`` (synchronous reduction); the async
    run uses the requested ``max_staleness``. Both share the SAME `TrainConfig`
    (so same seed, same prompts, same model), making the throughput and the
    staleness/KL/reward differences attributable to the async machinery alone.

    On the tiny model treat ``throughput_ratio`` as a systems-demonstration
    number, not a speedup claim (see the module docstring). The honest signal here
    is ``mean_lag_async`` (how off-policy the async run ran) against ``reward_gap``
    (what that cost in final reward).
    """
    baseline = train_grpo_async_staleness(
        cfg,
        StalenessConfig(max_staleness=0, queue_maxsize=1, producer_ahead=1),
        reward_pool=reward_pool,
    )
    async_run = train_grpo_async_staleness(
        cfg,
        StalenessConfig(max_staleness=max_staleness),
        reward_pool=reward_pool,
    )
    return StalenessComparison(baseline=baseline, async_run=async_run)
