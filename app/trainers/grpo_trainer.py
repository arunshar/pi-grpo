"""Group Relative Policy Optimization (Shao et al., 2024 / DeepSeek-R1).

For each prompt we sample K rollouts under the current policy. The
advantage of each rollout is

    A_k = (R_k - mean_K(R)) / std_K(R)

where R_k is the (physics-aware) reward of rollout k. The loss is the
PPO-style clipped surrogate but with no value head:

    L = -E_k [ min( ratio_k * A_k, clip(ratio_k, 1-eps, 1+eps) * A_k ) ]
        + beta * KL(pi || pi_ref)

This is well-suited to short-horizon physics-reasoning prompts because
it avoids the value-head bias that hurts PPO when the reward is sparse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import structlog
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from app.trainers.base import AdaptiveKLController, clip_grad_norm, cosine_lr

log = structlog.get_logger(__name__)


@dataclass
class GrpoConfig:
    group_size: int = 8
    lr: float = 5e-7
    lr_min: float = 1e-7
    warmup_steps: int = 50
    total_steps: int = 3000
    clip_coef: float = 0.2
    target_kl: float = 4.0
    grad_clip: float = 1.0


@dataclass
class GrpoBatch:
    """A group: K rollouts of the same prompt."""

    prompt_ids: torch.Tensor                  # (B, T_p)
    rollout_ids: torch.Tensor                 # (B, K, T_r)
    action_logp_old: torch.Tensor             # (B, K, T_r)
    rewards: torch.Tensor                     # (B, K) -- physics-aware total reward
    ref_logp: torch.Tensor                    # (B, K, T_r)


class GrpoTrainer:
    def __init__(self, *, policy, ref_policy, cfg: GrpoConfig) -> None:
        self.policy = policy
        self.ref = ref_policy
        self.cfg = cfg
        self.opt = AdamW(policy.parameters(), lr=cfg.lr)
        self.kl = AdaptiveKLController(target=cfg.target_kl)
        self.step = 0

    def step_update(self, batch: GrpoBatch) -> dict[str, float]:
        B, K = batch.rewards.shape
        # advantages
        mean_R = batch.rewards.mean(dim=1, keepdim=True)
        std_R = batch.rewards.std(dim=1, keepdim=True).clamp_min(1e-6)
        advantages = (batch.rewards - mean_R) / std_R                           # (B, K)

        # forward current policy on each rollout
        new_logp = self.policy.log_prob_token(batch.prompt_ids, batch.rollout_ids)  # (B, K, T_r)
        ratio = (new_logp - batch.action_logp_old).exp()                            # (B, K, T_r)
        adv = advantages.unsqueeze(-1)                                              # (B, K, 1)

        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1 - self.cfg.clip_coef, 1 + self.cfg.clip_coef) * adv
        pg_loss = -torch.min(unclipped, clipped).mean()

        kl_per_token = (new_logp - batch.ref_logp)                                  # (B, K, T_r)
        kl = kl_per_token.mean()

        loss = pg_loss + self.kl.kl_coef * kl

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = clip_grad_norm(self.policy.parameters(), self.cfg.grad_clip)
        for g in self.opt.param_groups:
            g["lr"] = cosine_lr(
                self.step, warmup=self.cfg.warmup_steps, total=self.cfg.total_steps,
                lr_max=self.cfg.lr, lr_min=self.cfg.lr_min,
            )
        self.opt.step()
        self.kl.update(float(kl.detach().item()), n_steps=1)
        self.step += 1
        return {
            "loss": float(loss.item()),
            "pg_loss": float(pg_loss.item()),
            "kl": float(kl.item()),
            "kl_coef": float(self.kl.kl_coef),
            "grad_norm": gn,
            "lr": self.opt.param_groups[0]["lr"],
            "reward_mean": float(batch.rewards.mean().item()),
            "reward_std": float(batch.rewards.std().item()),
        }
