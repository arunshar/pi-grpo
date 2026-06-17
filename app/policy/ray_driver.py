"""Ray-parallel GRPO driver (opt-in, additive).

`train_grpo_ray` mirrors `app.policy.driver.train_grpo` step for step, but moves
the per-step scoring bottleneck off the serial `_reward_matrix` nested B*K Python
loop and onto an injectable reward pool. Everything else, the policy, the
reference, the `GrpoTrainer`, the seeding, the prompts, and the eval generator,
is REUSED verbatim from the serial driver so a Ray run is directly comparable to
the pure-torch baseline: same seed -> same prompts -> same rollouts -> (when the
reward path is bit-identical) the same advantages and the same parameter updates.

The reward pool is the ONLY swapped component. It must satisfy this contract:

    reward_pool.score_rollouts(rollouts, codebook, reward, scorer)
        -> (rew, viol)                # both torch.Tensor of shape (B, K)

The contract is intentionally identical in inputs and outputs to
`app.policy.driver._reward_matrix`. The production implementation is
`app.policy.ray_reward_pool.RayRewardPool` (built by a sibling task): it shards
the (B, K) rollouts across Ray actors that each call the SAME serial
`_trajectory_reward`, so the merged (B, K) matrix is bit-identical to the serial
path (PhysicsReward is deterministic given states). Because the pool is
injectable, every test here runs with NO Ray cluster by passing a tiny stub pool
that delegates straight to `_reward_matrix`.

Timing: each step is decomposed into `t_sample` (policy.generate), `t_score`
(reward_pool.score_rollouts), and `t_learn` (trainer.step_update). These three
are written into the per-step metrics dict so a scaling study can read the
score-time fraction directly, and they sum (modulo a tiny bookkeeping residual)
to the wall time of the step.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import torch

from app.policy.driver import (
    StepCallback,
    TrainConfig,
    TrainResult,
    _mean_rollout_reward,
    _policy_config,
    _reward_matrix,
    _reward_path,
)
from app.policy.model import build_policy_pair
from app.trainers.grpo_trainer import GrpoBatch, GrpoConfig, GrpoTrainer

# Re-exported so callers can construct a comparable config without importing the
# serial driver as well. This is the SAME class, not a copy.
__all__ = ["RayTrainResult", "RewardPool", "default_reward_pool", "train_grpo_ray"]


class RewardPool(Protocol):
    """Structural contract for any rollout-scoring backend (serial stub or Ray).

    Implementations return rewards and physics-violation magnitudes for a batch
    of token rollouts, shape (B, K), matching `driver._reward_matrix` exactly.
    """

    def score_rollouts(
        self, rollouts: torch.Tensor, codebook, reward, scorer
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...


@dataclass
class RayTrainResult(TrainResult):
    """`TrainResult` plus an aggregate per-phase timing breakdown (seconds).

    `history` still holds the per-step metrics (each including `t_sample`,
    `t_score`, `t_learn`); these aggregates are the summed totals over the run
    for a quick scaling read without re-reducing the history.
    """

    t_sample_total: float = 0.0
    t_score_total: float = 0.0
    t_learn_total: float = 0.0


# ----------------------------------------------------------------- serial stub pool


class _SerialRewardPool:
    """Default-safe, cluster-free reward pool.

    Delegates straight to the serial `driver._reward_matrix`. Used as the default
    when Ray is unavailable and as the reference oracle in tests, so the Ray path
    can be checked for bit-identical parity against it.
    """

    def score_rollouts(
        self, rollouts: torch.Tensor, codebook, reward, scorer
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _reward_matrix(rollouts, codebook, reward, scorer)


def default_reward_pool(num_workers: int | None = None) -> RewardPool:
    """Build the production Ray reward pool, falling back to the serial stub.

    The Ray pool lives in a sibling module (`app.policy.ray_reward_pool`). It is
    imported lazily so this module imports cleanly even when that module or Ray
    itself is absent (e.g. on the CPU-only login node), in which case we degrade
    to the serial pool. This keeps `train_grpo_ray` runnable everywhere while
    still defaulting to the parallel path when the cluster is present.
    """
    try:
        # Imported lazily and by name to avoid a hard import-time dependency on
        # the sibling task's module or on Ray being installed.
        from app.policy.ray_reward_pool import RayRewardPool  # type: ignore[attr-defined]
    except Exception:
        # No Ray / sibling module yet: the serial pool is correct, just serial.
        return _SerialRewardPool()
    # num_workers is None when no worker count was resolved (e.g. Ray absent) and
    # n_actors<=1 is serial anyway, so degrade to the serial stub instead of
    # building a 1-actor pool (this also avoids int(None) in RayRewardPool).
    if not num_workers or num_workers <= 1:
        return _SerialRewardPool()
    return RayRewardPool(n_actors=num_workers)


# ----------------------------------------------------------------------- GRPO (Ray)


def train_grpo_ray(
    cfg: TrainConfig = TrainConfig(),
    reward_pool: RewardPool | None = None,
    on_step: StepCallback | None = None,
) -> RayTrainResult:
    """Ray-parallel GRPO. Drop-in comparable to `driver.train_grpo`.

    The ONLY behavioral difference from the serial driver is that per-step
    scoring goes through `reward_pool.score_rollouts` instead of the inline
    `_reward_matrix`, and each step records `t_sample` / `t_score` / `t_learn`.
    Seeding, prompts, policy/reference construction, and the trainer are
    identical, so with a bit-identical reward pool the trajectory of updates
    matches the serial `train_grpo` exactly.
    """
    if reward_pool is None:
        reward_pool = default_reward_pool()

    # ---- identical setup to driver.train_grpo (same seeds => same prompts) ----
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
    eval_gen = torch.Generator().manual_seed(cfg.seed + 7)
    reward_start = _mean_rollout_reward(
        policy, prompts, cfg, codebook, reward, scorer, eval_gen
    )

    history: list[dict[str, float]] = []
    metrics: dict[str, float] = {}
    t_sample_total = 0.0
    t_score_total = 0.0
    t_learn_total = 0.0

    for step in range(cfg.steps):
        # (1) SAMPLE
        t0 = time.perf_counter()
        rollouts, logp_old = policy.generate(
            prompts,
            k=cfg.group_size,
            max_new_tokens=cfg.horizon,
            temperature=cfg.temperature,
            generator=gen,
        )
        t_sample = time.perf_counter() - t0

        # (2) SCORE  -- the parallelized phase
        t1 = time.perf_counter()
        rew, _viol = reward_pool.score_rollouts(rollouts, codebook, reward, scorer)
        t_score = time.perf_counter() - t1

        # (3) LEARN
        t2 = time.perf_counter()
        ref_logp = ref.log_prob_token(prompts, rollouts).detach()
        batch = GrpoBatch(
            prompt_ids=prompts,
            rollout_ids=rollouts,
            action_logp_old=logp_old.detach(),
            rewards=rew,
            ref_logp=ref_logp,
        )
        metrics = trainer.step_update(batch)
        t_learn = time.perf_counter() - t2

        metrics = dict(metrics)  # do not mutate the trainer's returned dict in place
        metrics["t_sample"] = t_sample
        metrics["t_score"] = t_score
        metrics["t_learn"] = t_learn
        metrics["t_step"] = t_sample + t_score + t_learn
        history.append(metrics)
        t_sample_total += t_sample
        t_score_total += t_score
        t_learn_total += t_learn

        if on_step is not None:
            on_step(step, metrics)

    eval_gen = torch.Generator().manual_seed(cfg.seed + 7)
    reward_end = _mean_rollout_reward(
        policy, prompts, cfg, codebook, reward, scorer, eval_gen
    )
    return RayTrainResult(
        final_step=cfg.steps,
        final_metrics=metrics,
        history=history,
        reward_start=reward_start,
        reward_end=reward_end,
        t_sample_total=t_sample_total,
        t_score_total=t_score_total,
        t_learn_total=t_learn_total,
    )
