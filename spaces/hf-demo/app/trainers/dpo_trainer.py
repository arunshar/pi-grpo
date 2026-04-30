"""Direct Preference Optimization (Rafailov et al., 2023).

Loss:

    L_DPO = -E_(x, yw, yl) log sigma( beta * (
        log pi(yw|x) - log pi_ref(yw|x)
      - (log pi(yl|x) - log pi_ref(yl|x))
    ))

No reward model, no value head. We add a `gamma` term that mixes a
small physics-aware penalty into the implicit reward so the policy is
biased away from physics-violating outputs even when the human label
did not encode that signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from app.trainers.base import clip_grad_norm, cosine_lr

log = structlog.get_logger(__name__)


@dataclass
class DpoConfig:
    lr: float = 5e-7
    lr_min: float = 1e-7
    warmup_steps: int = 50
    total_steps: int = 2000
    beta: float = 0.1
    gamma_phys: float = 0.05
    grad_clip: float = 1.0


class DpoTrainer:
    def __init__(self, *, policy, ref_policy, cfg: DpoConfig) -> None:
        self.policy = policy
        self.ref = ref_policy
        self.cfg = cfg
        self.opt = AdamW(policy.parameters(), lr=cfg.lr)
        self.step = 0

    def step_update(
        self,
        prompt_ids: torch.Tensor,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        chosen_phys_violation: torch.Tensor,
        rejected_phys_violation: torch.Tensor,
    ) -> dict[str, float]:
        logp_chosen = self.policy.log_prob_seq(prompt_ids, chosen_ids)
        logp_rejected = self.policy.log_prob_seq(prompt_ids, rejected_ids)
        with torch.no_grad():
            ref_logp_chosen = self.ref.log_prob_seq(prompt_ids, chosen_ids)
            ref_logp_rejected = self.ref.log_prob_seq(prompt_ids, rejected_ids)

        chosen_logits = self.cfg.beta * (logp_chosen - ref_logp_chosen)
        rejected_logits = self.cfg.beta * (logp_rejected - ref_logp_rejected)
        # add a small physics-aware term: penalize rewards that prefer
        # physics-violating responses.
        chosen_logits = chosen_logits - self.cfg.gamma_phys * chosen_phys_violation
        rejected_logits = rejected_logits - self.cfg.gamma_phys * rejected_phys_violation

        loss = -F.logsigmoid(chosen_logits - rejected_logits).mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = clip_grad_norm(self.policy.parameters(), self.cfg.grad_clip)
        for g in self.opt.param_groups:
            g["lr"] = cosine_lr(
                self.step, warmup=self.cfg.warmup_steps, total=self.cfg.total_steps,
                lr_max=self.cfg.lr, lr_min=self.cfg.lr_min,
            )
        self.opt.step()
        self.step += 1
        with torch.no_grad():
            margin = (chosen_logits - rejected_logits).mean()
        return {
            "loss": float(loss.item()),
            "margin": float(margin.item()),
            "grad_norm": gn,
            "lr": self.opt.param_groups[0]["lr"],
        }
