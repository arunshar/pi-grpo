"""Causal-LM policy + value head shared by the PPO / DPO / GRPO trainers.

A decoder-only Transformer over a discrete *action* vocabulary. Each token
indexes a motion primitive (see `app.policy.decode`), so a rollout is a control
sequence that integrates through the S-KBM into a trajectory the
`PhysicsReward` can score. The class implements exactly the three log-prob
surfaces the trainers call, with consistent causal alignment:

    log_prob_token(prompt_ids, rollout_ids) -> (B, K, T_r)   # GRPO
    log_prob_with_entropy(obs)              -> (logp, ent)    # PPO, per-token (B, T-1)
    log_prob_seq(prompt_ids, response_ids)  -> (B,)           # DPO, sequence sum

plus `generate()` for on-policy rollouts and `frozen_clone()` for the frozen
reference policy. The model is deliberately tiny so a real training step runs on
CPU in tests; the same class scales by widening `PolicyConfig`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PolicyConfig:
    vocab_size: int = 32
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 128
    max_len: int = 64
    dropout: float = 0.0


class _Block(nn.Module):
    """Pre-norm Transformer decoder block (causal self-attention + MLP)."""

    def __init__(self, cfg: PolicyConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class CausalPolicy(nn.Module):
    """Decoder-only token policy. Logits over the motion-primitive vocabulary."""

    def __init__(self, cfg: PolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList(_Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------ core

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        # True entries are NOT allowed to attend (upper triangle, excl. diagonal).
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
        )

    def logits(self, ids: torch.Tensor) -> torch.Tensor:
        """(N, L) token ids -> (N, L, V) next-token logits."""
        length = ids.shape[1]
        if length > self.cfg.max_len:
            raise ValueError(f"sequence length {length} exceeds max_len {self.cfg.max_len}")
        pos = torch.arange(length, device=ids.device)
        x = self.tok(ids) + self.pos(pos).unsqueeze(0)
        mask = self._causal_mask(length, ids.device)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.ln_f(x))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:  # convenience alias
        return self.logits(ids)

    # ------------------------------------------------- GRPO log-prob surface

    def log_prob_token(self, prompt_ids: torch.Tensor, rollout_ids: torch.Tensor) -> torch.Tensor:
        """Per-token log-prob of each rollout token under the policy.

        prompt_ids (B, T_p), rollout_ids (B, K, T_r) -> (B, K, T_r).
        Rollout token j sits at concat position T_p + j and is predicted by the
        logits at position T_p + j - 1.
        """
        b, k, t_r = rollout_ids.shape
        t_p = prompt_ids.shape[1]
        prompt = prompt_ids.unsqueeze(1).expand(b, k, t_p)
        seq = torch.cat([prompt, rollout_ids], dim=2).reshape(b * k, t_p + t_r)
        logp_all = F.log_softmax(self.logits(seq), dim=-1)          # (B*K, T_p+T_r, V)
        pred_positions = torch.arange(t_p - 1, t_p + t_r - 1, device=seq.device)
        pred = logp_all[:, pred_positions, :]                       # (B*K, T_r, V)
        tok = rollout_ids.reshape(b * k, t_r).unsqueeze(-1)
        lp = pred.gather(-1, tok).squeeze(-1)                       # (B*K, T_r)
        return lp.view(b, k, t_r)

    # -------------------------------------------------- PPO log-prob surface

    def log_prob_with_entropy(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """obs (B, T) prompt+response ids -> (logp (B, T-1), entropy (B, T-1)).

        logp[:, t] is the log-prob of obs[:, t+1] under the categorical predicted
        from position t; entropy is that categorical's entropy.
        """
        logp_all = F.log_softmax(self.logits(obs), dim=-1)          # (B, T, V)
        pred = logp_all[:, :-1, :]                                  # (B, T-1, V)
        targets = obs[:, 1:].unsqueeze(-1)                          # (B, T-1, 1)
        logp = pred.gather(-1, targets).squeeze(-1)                 # (B, T-1)
        entropy = -(pred.exp() * pred).sum(dim=-1)                  # (B, T-1)
        return logp, entropy

    # -------------------------------------------------- DPO log-prob surface

    def log_prob_seq(self, prompt_ids: torch.Tensor, response_ids: torch.Tensor) -> torch.Tensor:
        """Sequence-summed log-prob of a response given a prompt. -> (B,)."""
        lp = self.log_prob_token(prompt_ids, response_ids.unsqueeze(1))  # (B, 1, T_r)
        return lp.squeeze(1).sum(dim=-1)                                 # (B,)

    # ------------------------------------------------------------ rollouts

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        *,
        k: int,
        max_new_tokens: int,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample K rollouts per prompt. -> (rollout_ids, logp_old), both (B, K, T_r)."""
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        b, t_p = prompt_ids.shape
        seq = prompt_ids.unsqueeze(1).expand(b, k, t_p).reshape(b * k, t_p).clone()
        toks: list[torch.Tensor] = []
        logps: list[torch.Tensor] = []
        for _ in range(max_new_tokens):
            step_logits = self.logits(seq)[:, -1, :] / temperature
            logp_all = F.log_softmax(step_logits, dim=-1)
            nxt = torch.multinomial(logp_all.exp(), num_samples=1, generator=generator)
            logps.append(logp_all.gather(-1, nxt).squeeze(-1))
            toks.append(nxt.squeeze(-1))
            seq = torch.cat([seq, nxt], dim=1)
        rollout = torch.stack(toks, dim=1).view(b, k, max_new_tokens)
        logp_old = torch.stack(logps, dim=1).view(b, k, max_new_tokens)
        return rollout, logp_old

    def frozen_clone(self) -> CausalPolicy:
        """A detached eval-mode copy for use as the reference policy."""
        ref = copy.deepcopy(self)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        return ref


class ValueHead(nn.Module):
    """State-value head for PPO. Maps prompt+response ids to a per-position value.

    Independent of the policy trunk (its own small embedding) so PPO can be
    exercised end to end without coupling the value estimate to the policy
    gradient. Output (B, T-1) aligns with the per-token returns/advantages.
    """

    def __init__(self, cfg: PolicyConfig) -> None:
        super().__init__()
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.net = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        length = obs.shape[1]
        pos = torch.arange(length, device=obs.device)
        x = self.tok(obs) + self.pos(pos).unsqueeze(0)
        v = self.net(x).squeeze(-1)            # (B, T)
        return v[:, :-1]                        # (B, T-1), aligns with predictions


def build_policy_pair(cfg: PolicyConfig) -> tuple[CausalPolicy, CausalPolicy]:
    """A trainable policy and its frozen reference (same initial weights)."""
    policy = CausalPolicy(cfg)
    ref = policy.frozen_clone()
    return policy, ref


__all__ = [
    "CausalPolicy",
    "PolicyConfig",
    "ValueHead",
    "build_policy_pair",
]
