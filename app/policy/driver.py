"""Real training drivers for the PPO / DPO / GRPO trainers.

Each driver wires the `CausalPolicy` to the existing physics-aware reward path
(`PhysicsReward` + `PiDpmScorer`) through the motion-primitive `MotionCodebook`,
then runs genuine optimization steps. This is what replaces the orchestrator's
former metric simulation: a tiny model trains on CPU in seconds, and the metrics
reported are measured, not synthesized.

Tokens -> controls -> S-KBM rollout -> trajectory -> reward. GRPO pushes the
policy toward the feasible region of the control grid, so mean on-policy reward
rises measurably over a run (see `tests/test_policy.py`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from app.components.kinematic_bicycle import SkbmConfig
from app.components.physics_reward import EmpiricalEnvelope, PhysicsReward, RewardWeights
from app.components.pidpm_scorer import PiDpmScorer
from app.policy.decode import CodebookConfig, MotionCodebook
from app.policy.model import CausalPolicy, PolicyConfig, ValueHead, build_policy_pair
from app.trainers.dpo_trainer import DpoConfig, DpoTrainer
from app.trainers.grpo_trainer import GrpoBatch, GrpoConfig, GrpoTrainer
from app.trainers.ppo_trainer import PpoConfig, PpoTrainer

StepCallback = Callable[[int, dict[str, float]], None]


@dataclass(frozen=True)
class TrainConfig:
    """End-to-end run config (model + data + optimization)."""

    steps: int = 60
    batch_prompts: int = 4          # B distinct prompts per step
    group_size: int = 6             # K rollouts per prompt (GRPO/DPO selection)
    prompt_len: int = 2             # T_p conditioning tokens
    horizon: int = 12               # T_r control tokens per rollout
    temperature: float = 1.0
    lr: float = 1e-2                # visible learning on the tiny synthetic task
    seed: int = 42
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    codebook: CodebookConfig = field(default_factory=CodebookConfig)


# A fast preset for the API path / smoke runs (must finish well under ~2 s on CPU).
SMOKE = TrainConfig(steps=5, batch_prompts=3, group_size=4, horizon=8, d_model=32, n_layers=1)


@dataclass
class TrainResult:
    final_step: int
    final_metrics: dict[str, float]
    history: list[dict[str, float]]
    reward_start: float
    reward_end: float


# --------------------------------------------------------------------------- setup


def _reward_path(cfg: TrainConfig) -> tuple[MotionCodebook, PhysicsReward, PiDpmScorer]:
    skbm = SkbmConfig()
    codebook = MotionCodebook(skbm, cfg.codebook)
    reward = PhysicsReward(skbm, weights=RewardWeights(), envelope=EmpiricalEnvelope())
    scorer = PiDpmScorer(device="cpu")     # analytic proxy when no checkpoint
    return codebook, reward, scorer


def _policy_config(cfg: TrainConfig, vocab: int) -> PolicyConfig:
    return PolicyConfig(
        vocab_size=vocab,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_model * 2,
        max_len=cfg.prompt_len + cfg.horizon + 2,
    )


def _trajectory_reward(
    tokens: np.ndarray, codebook: MotionCodebook, reward: PhysicsReward, scorer: PiDpmScorer
) -> tuple[float, float]:
    """Decode a token rollout and score it. Returns (total_reward, phys_violation)."""
    states = codebook.tokens_to_states(tokens)
    breakdown = reward.score(states, pi_dpm_log_prob=scorer.log_prob(states))
    return breakdown.total, -breakdown.hard      # hard <= 0; violation magnitude >= 0


def _reward_matrix(
    rollouts: torch.Tensor, codebook: MotionCodebook, reward: PhysicsReward, scorer: PiDpmScorer
) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, K, T_r) token rollouts -> (rewards (B, K), phys_violation (B, K))."""
    b, k, _t = rollouts.shape
    rew = torch.zeros(b, k)
    viol = torch.zeros(b, k)
    arr = rollouts.cpu().numpy()
    for i in range(b):
        for j in range(k):
            total, v = _trajectory_reward(arr[i, j], codebook, reward, scorer)
            rew[i, j] = total
            viol[i, j] = v
    return rew, viol


@torch.no_grad()
def _mean_rollout_reward(
    policy: CausalPolicy,
    prompts: torch.Tensor,
    cfg: TrainConfig,
    codebook: MotionCodebook,
    reward: PhysicsReward,
    scorer: PiDpmScorer,
    generator: torch.Generator,
) -> float:
    rollouts, _ = policy.generate(
        prompts, k=cfg.group_size, max_new_tokens=cfg.horizon,
        temperature=cfg.temperature, generator=generator,
    )
    rew, _ = _reward_matrix(rollouts, codebook, reward, scorer)
    return float(rew.mean().item())


# --------------------------------------------------------------------------- GRPO


def train_grpo(cfg: TrainConfig = TrainConfig(), on_step: StepCallback | None = None) -> TrainResult:
    torch.manual_seed(cfg.seed)
    codebook, reward, scorer = _reward_path(cfg)
    policy, ref = build_policy_pair(_policy_config(cfg, codebook.vocab_size))
    trainer = GrpoTrainer(
        policy=policy, ref_policy=ref,
        cfg=GrpoConfig(group_size=cfg.group_size, lr=cfg.lr, total_steps=cfg.steps),
    )

    gen = torch.Generator().manual_seed(cfg.seed + 1)
    prompts = torch.randint(0, codebook.vocab_size, (cfg.batch_prompts, cfg.prompt_len), generator=gen)
    eval_gen = torch.Generator().manual_seed(cfg.seed + 7)
    reward_start = _mean_rollout_reward(policy, prompts, cfg, codebook, reward, scorer, eval_gen)

    history: list[dict[str, float]] = []
    metrics: dict[str, float] = {}
    for step in range(cfg.steps):
        rollouts, logp_old = policy.generate(
            prompts, k=cfg.group_size, max_new_tokens=cfg.horizon,
            temperature=cfg.temperature, generator=gen,
        )
        rew, _ = _reward_matrix(rollouts, codebook, reward, scorer)
        ref_logp = ref.log_prob_token(prompts, rollouts).detach()
        batch = GrpoBatch(
            prompt_ids=prompts, rollout_ids=rollouts,
            action_logp_old=logp_old.detach(), rewards=rew, ref_logp=ref_logp,
        )
        metrics = trainer.step_update(batch)
        history.append(metrics)
        if on_step is not None:
            on_step(step, metrics)

    eval_gen = torch.Generator().manual_seed(cfg.seed + 7)
    reward_end = _mean_rollout_reward(policy, prompts, cfg, codebook, reward, scorer, eval_gen)
    return TrainResult(cfg.steps, metrics, history, reward_start, reward_end)


# --------------------------------------------------------------------------- PPO


def train_ppo(cfg: TrainConfig = TrainConfig(), on_step: StepCallback | None = None) -> TrainResult:
    torch.manual_seed(cfg.seed)
    codebook, reward, scorer = _reward_path(cfg)
    pcfg = _policy_config(cfg, codebook.vocab_size)
    policy, ref = build_policy_pair(pcfg)
    value_head = ValueHead(pcfg)
    trainer = PpoTrainer(
        policy=policy, ref_policy=ref, value_head=value_head,
        cfg=PpoConfig(lr=cfg.lr, total_steps=cfg.steps),
    )

    gen = torch.Generator().manual_seed(cfg.seed + 1)
    prompts = torch.randint(0, codebook.vocab_size, (cfg.batch_prompts, cfg.prompt_len), generator=gen)
    eval_gen = torch.Generator().manual_seed(cfg.seed + 7)
    reward_start = _mean_rollout_reward(policy, prompts, cfg, codebook, reward, scorer, eval_gen)

    history: list[dict[str, float]] = []
    metrics: dict[str, float] = {}
    t_p = cfg.prompt_len
    for step in range(cfg.steps):
        rollouts, _ = policy.generate(
            prompts, k=1, max_new_tokens=cfg.horizon,
            temperature=cfg.temperature, generator=gen,
        )
        rew, _ = _reward_matrix(rollouts, codebook, reward, scorer)     # (B, 1)
        obs = torch.cat([prompts, rollouts.squeeze(1)], dim=1)          # (B, T_p+T_r)

        # per-prediction-position returns: terminal reward on response positions, 0 on prompt
        returns = torch.zeros(obs.shape[0], obs.shape[1] - 1)
        returns[:, t_p - 1:] = rew                                     # broadcast (B,1) -> response cols
        with torch.no_grad():
            action_logp, _ = policy.log_prob_with_entropy(obs)         # old == current (on-policy)
            ref_logp, _ = ref.log_prob_with_entropy(obs)
            values = value_head(obs)
            advantages = returns - values
        batch = _PpoBatch(
            obs=obs, action_logp=action_logp.detach(), rewards=rew,
            returns=returns, advantages=advantages.detach(), ref_logp=ref_logp.detach(),
        )
        metrics = trainer.step_update(batch)
        history.append(metrics)
        if on_step is not None:
            on_step(step, metrics)

    eval_gen = torch.Generator().manual_seed(cfg.seed + 7)
    reward_end = _mean_rollout_reward(policy, prompts, cfg, codebook, reward, scorer, eval_gen)
    return TrainResult(cfg.steps, metrics, history, reward_start, reward_end)


@dataclass
class _PpoBatch:
    obs: torch.Tensor
    action_logp: torch.Tensor
    rewards: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    ref_logp: torch.Tensor


# --------------------------------------------------------------------------- DPO


def train_dpo(cfg: TrainConfig = TrainConfig(), on_step: StepCallback | None = None) -> TrainResult:
    torch.manual_seed(cfg.seed)
    codebook, reward, scorer = _reward_path(cfg)
    policy, ref = build_policy_pair(_policy_config(cfg, codebook.vocab_size))
    trainer = DpoTrainer(policy=policy, ref_policy=ref, cfg=DpoConfig(lr=cfg.lr, total_steps=cfg.steps))

    gen = torch.Generator().manual_seed(cfg.seed + 1)
    prompts = torch.randint(0, codebook.vocab_size, (cfg.batch_prompts, cfg.prompt_len), generator=gen)
    eval_gen = torch.Generator().manual_seed(cfg.seed + 7)
    reward_start = _mean_rollout_reward(policy, prompts, cfg, codebook, reward, scorer, eval_gen)

    history: list[dict[str, float]] = []
    metrics: dict[str, float] = {}
    for step in range(cfg.steps):
        rollouts, _ = policy.generate(
            prompts, k=cfg.group_size, max_new_tokens=cfg.horizon,
            temperature=cfg.temperature, generator=gen,
        )
        rew, viol = _reward_matrix(rollouts, codebook, reward, scorer)  # (B, K)
        best = rew.argmax(dim=1)
        worst = rew.argmin(dim=1)
        bidx = torch.arange(rollouts.shape[0])
        chosen = rollouts[bidx, best]
        rejected = rollouts[bidx, worst]
        metrics = trainer.step_update(
            prompts, chosen, rejected,
            chosen_phys_violation=viol[bidx, best],
            rejected_phys_violation=viol[bidx, worst],
        )
        history.append(metrics)
        if on_step is not None:
            on_step(step, metrics)

    eval_gen = torch.Generator().manual_seed(cfg.seed + 7)
    reward_end = _mean_rollout_reward(policy, prompts, cfg, codebook, reward, scorer, eval_gen)
    return TrainResult(cfg.steps, metrics, history, reward_start, reward_end)


TRAINERS = {"grpo": train_grpo, "ppo": train_ppo, "dpo": train_dpo}


def train(algo: str, cfg: TrainConfig = TrainConfig(), on_step: StepCallback | None = None) -> TrainResult:
    if algo not in TRAINERS:
        raise ValueError(f"unknown algo {algo!r}; choose from {sorted(TRAINERS)}")
    return TRAINERS[algo](cfg, on_step)


__all__ = ["SMOKE", "TrainConfig", "TrainResult", "train", "train_dpo", "train_grpo", "train_ppo"]
