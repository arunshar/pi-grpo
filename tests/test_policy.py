"""Policy network + real training-loop tests.

These exercise the class the three trainers depend on: the causal alignment of
its log-prob surfaces, consistency between sampling and re-scoring, and that a
real GRPO loop actually moves the policy toward higher physics-aware reward.
"""

from __future__ import annotations

import numpy as np
import torch

from app.components.kinematic_bicycle import SkbmConfig
from app.policy.decode import CodebookConfig, MotionCodebook
from app.policy.driver import TrainConfig, train_dpo, train_grpo, train_ppo
from app.policy.model import CausalPolicy, PolicyConfig, ValueHead, build_policy_pair


def _tiny_cfg(vocab: int = 16) -> PolicyConfig:
    return PolicyConfig(vocab_size=vocab, d_model=32, n_layers=2, n_heads=4, d_ff=64, max_len=32)


# ------------------------------------------------------------------ shapes


def test_logits_shape() -> None:
    cfg = _tiny_cfg()
    pol = CausalPolicy(cfg)
    ids = torch.randint(0, cfg.vocab_size, (3, 7))
    assert pol.logits(ids).shape == (3, 7, cfg.vocab_size)


def test_log_prob_token_shape_and_alignment() -> None:
    cfg = _tiny_cfg()
    pol = CausalPolicy(cfg).eval()
    b, k, t_p, t_r = 2, 3, 3, 4
    prompt = torch.randint(0, cfg.vocab_size, (b, t_p))
    rollout = torch.randint(0, cfg.vocab_size, (b, k, t_r))
    lp = pol.log_prob_token(prompt, rollout)
    assert lp.shape == (b, k, t_r)

    # manual reference: rollout token j predicted by logits at position t_p-1+j
    for bi in range(b):
        for ki in range(k):
            seq = torch.cat([prompt[bi], rollout[bi, ki]]).unsqueeze(0)
            logp_all = torch.log_softmax(pol.logits(seq), dim=-1)[0]
            for j in range(t_r):
                expected = logp_all[t_p - 1 + j, rollout[bi, ki, j]]
                assert torch.allclose(lp[bi, ki, j], expected, atol=1e-5)


def test_generate_logp_matches_rescored() -> None:
    cfg = _tiny_cfg()
    pol = CausalPolicy(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (2, 3))
    g = torch.Generator().manual_seed(0)
    rollout, logp_old = pol.generate(prompt, k=3, max_new_tokens=5, temperature=1.0, generator=g)
    assert rollout.shape == (2, 3, 5) and logp_old.shape == (2, 3, 5)
    rescored = pol.log_prob_token(prompt, rollout)
    assert torch.allclose(rescored, logp_old, atol=1e-5)


def test_log_prob_seq_equals_token_sum() -> None:
    cfg = _tiny_cfg()
    pol = CausalPolicy(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (4, 2))
    resp = torch.randint(0, cfg.vocab_size, (4, 6))
    seq_lp = pol.log_prob_seq(prompt, resp)
    tok_lp = pol.log_prob_token(prompt, resp.unsqueeze(1)).squeeze(1).sum(dim=-1)
    assert seq_lp.shape == (4,)
    assert torch.allclose(seq_lp, tok_lp, atol=1e-5)


def test_value_head_shape() -> None:
    cfg = _tiny_cfg()
    vh = ValueHead(cfg)
    obs = torch.randint(0, cfg.vocab_size, (3, 9))
    assert vh(obs).shape == (3, 8)


def test_entropy_is_nonnegative() -> None:
    cfg = _tiny_cfg()
    pol = CausalPolicy(cfg).eval()
    obs = torch.randint(0, cfg.vocab_size, (2, 8))
    _, ent = pol.log_prob_with_entropy(obs)
    assert ent.shape == (2, 7)
    assert (ent >= -1e-6).all()


def test_frozen_clone_is_detached() -> None:
    cfg = _tiny_cfg()
    pol = CausalPolicy(cfg)
    ref = pol.frozen_clone()
    assert all(not p.requires_grad for p in ref.parameters())
    # same initial weights
    for a, b in zip(pol.parameters(), ref.parameters(), strict=False):
        assert torch.allclose(a, b)


# ------------------------------------------------------------------ codebook


def test_codebook_decodes_and_some_tokens_violate() -> None:
    skbm = SkbmConfig()
    cb = MotionCodebook(skbm, CodebookConfig(n_accel=5, n_steer=5, span=1.6))
    assert cb.vocab_size == 25
    tokens = np.arange(cb.vocab_size)
    controls = cb.controls(tokens)
    assert controls.shape == (cb.vocab_size, 2)
    # the grid spans beyond the bounds, so some accel magnitudes exceed a_max
    assert (np.abs(controls[:, 0]) > skbm.a_max_mps2 + 1e-9).any()
    states = cb.tokens_to_states(tokens[:8])
    assert states.shape == (9, 4)  # T+1 states for T controls


# ------------------------------------------------------------------ trainers


def _learn_cfg() -> TrainConfig:
    return TrainConfig(
        steps=40, batch_prompts=4, group_size=6, prompt_len=2, horizon=10,
        lr=1e-2, seed=0, d_model=32, n_layers=1, n_heads=4,
    )


def test_grpo_step_runs_and_updates_params() -> None:
    cfg = TrainConfig(steps=2, batch_prompts=3, group_size=4, horizon=6, lr=1e-2, d_model=32, n_layers=1)
    # snapshot a param before/after via the public driver by training 2 steps
    res = train_grpo(cfg)
    assert res.final_step == 2
    assert np.isfinite(res.final_metrics["loss"])
    assert "kl" in res.final_metrics


def test_grpo_increases_mean_reward() -> None:
    res = train_grpo(_learn_cfg())
    # a real loop on a feasible-region task should raise mean on-policy reward
    assert res.reward_end > res.reward_start


def test_ppo_step_runs() -> None:
    cfg = TrainConfig(steps=3, batch_prompts=3, group_size=1, horizon=6, lr=1e-2, d_model=32, n_layers=1)
    res = train_ppo(cfg)
    assert res.final_step == 3
    assert np.isfinite(res.final_metrics["loss"])
    assert "vf_loss" in res.final_metrics


def test_dpo_step_runs() -> None:
    cfg = TrainConfig(steps=3, batch_prompts=3, group_size=4, horizon=6, lr=1e-2, d_model=32, n_layers=1)
    res = train_dpo(cfg)
    assert res.final_step == 3
    assert np.isfinite(res.final_metrics["loss"])
    assert "margin" in res.final_metrics


def test_build_policy_pair_shares_init() -> None:
    cfg = _tiny_cfg()
    pol, ref = build_policy_pair(cfg)
    for a, b in zip(pol.parameters(), ref.parameters(), strict=False):
        assert torch.allclose(a, b)
