"""Proximal Policy Optimization (Schulman et al., 2017).

Clipped surrogate loss + value head + GAE. Implemented on top of TRL's
PPOTrainer with our adaptive KL controller and our PhysicsReward in
the reward path. We keep the trainer thin — the value comes from
matching token-level KL accounting and reward shaping correctly.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog
import torch
from torch.optim import AdamW

from app.trainers.base import AdaptiveKLController, clip_grad_norm, cosine_lr

log = structlog.get_logger(__name__)


@dataclass
class PpoConfig:
    lr: float = 1e-6
    lr_min: float = 1e-7
    warmup_steps: int = 100
    total_steps: int = 5000
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    grad_clip: float = 1.0
    target_kl: float = 6.0
    minibatch_size: int = 16
    rollout_batch_size: int = 64


@dataclass
class _PpoBatch:
    obs: torch.Tensor       # token ids of prompt + response
    action_logp: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    ref_logp: torch.Tensor


class PpoTrainer:
    def __init__(self, *, policy, ref_policy, value_head, cfg: PpoConfig) -> None:
        self.policy = policy
        self.ref = ref_policy
        self.vh = value_head
        self.cfg = cfg
        self.opt = AdamW(list(policy.parameters()) + list(value_head.parameters()), lr=cfg.lr)
        self.kl = AdaptiveKLController(target=cfg.target_kl)
        self.step = 0

    def step_update(self, batch: _PpoBatch) -> dict[str, float]:
        # forward through current policy
        new_logp, ent = self.policy.log_prob_with_entropy(batch.obs)
        ratio = (new_logp - batch.action_logp).exp()
        unclipped = ratio * batch.advantages
        clipped = torch.clamp(ratio, 1 - self.cfg.clip_coef, 1 + self.cfg.clip_coef) * batch.advantages
        pg_loss = -(torch.min(unclipped, clipped)).mean()

        v = self.vh(batch.obs)
        vf_loss = (v - batch.returns).pow(2).mean()

        kl = (batch.ref_logp - new_logp).mean()
        loss = (
            pg_loss
            + self.cfg.vf_coef * vf_loss
            - self.cfg.ent_coef * ent.mean()
            + self.kl.kl_coef * kl
        )

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = clip_grad_norm(list(self.policy.parameters()) + list(self.vh.parameters()), self.cfg.grad_clip)
        for g in self.opt.param_groups:
            g["lr"] = cosine_lr(
                self.step, warmup=self.cfg.warmup_steps, total=self.cfg.total_steps,
                lr_max=self.cfg.lr, lr_min=self.cfg.lr_min,
            )
        self.opt.step()

        self.kl.update(float(kl.detach().item()), n_steps=1)
        self.step += 1
        return {
            "loss": float(loss.detach().item()),
            "pg_loss": float(pg_loss.detach().item()),
            "vf_loss": float(vf_loss.detach().item()),
            "kl": float(kl.detach().item()),
            "kl_coef": float(self.kl.kl_coef),
            "grad_norm": gn,
            "lr": self.opt.param_groups[0]["lr"],
        }
