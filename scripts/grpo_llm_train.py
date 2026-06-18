"""EXP-5: full-scale GRPO on a Qwen2-7B + LoRA policy under the physics reward.

This is the real LLM-scale trainer the paper's full-scale run needs. It reuses the
EXISTING physics reward path unchanged: the policy emits a trajectory as a sequence
of motion-primitive ids (the `MotionCodebook` vocabulary, a discrete accel x steer
grid), which `tokens_to_states` rolls out through the single-axle kinematic-bicycle
model into an (x, y, theta, v) state trajectory that `PhysicsReward` + `PiDpmScorer`
score. GRPO then optimizes the LoRA adapter with group-baseline advantages and a KL
penalty to the frozen reference (the base model, recovered by disabling the adapter),
exactly as `app.trainers.grpo_trainer` does for the tiny policy.

STATUS: this runs only inside the LLM container (docker/pigrpo_llm.def) on a GPU; it
is structurally complete and byte-compiles, but it has NOT been executed end to end
yet (the container and model download are the gating steps; see
docs/EXP5_LLM_SCALE.md). The rollout backend defaults to Hugging Face `generate`
(on-policy, correct); a vLLM backend is the throughput optimization and is left as a
documented hook because vLLM serves the frozen base weights, so on-policy LoRA
rollouts need a per-step adapter sync.

Reward is bit-compatible with the CPU experiments: same MotionCodebook + PhysicsReward
+ PiDpmScorer, so the trained-policy numbers fill tab:expected-trends on the same
reward the scaling/reward-hacking results use.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="grpo_llm_train", description=__doc__)
    ap.add_argument("--algo", choices=["grpo", "ppo", "dpo"], default="grpo")
    ap.add_argument("--base-model", default="Qwen/Qwen2-7B-Instruct")
    ap.add_argument("--vllm-url", default=None,
                    help="if set, use a vLLM OpenAI endpoint for rollouts (see note above)")
    ap.add_argument("--preferences", default=None, help="DPO triples (unused for grpo)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-prompts", type=int, default=8, help="B prompts per step")
    ap.add_argument("--group-size", type=int, default=8, help="K rollouts per prompt")
    ap.add_argument("--horizon", type=int, default=24, help="motion-primitive ids per trajectory")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--clip-coef", type=float, default=0.2)
    ap.add_argument("--target-kl", type=float, default=4.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="runs/llm_grpo")
    ap.add_argument("--log-every", type=int, default=1)
    return ap


# --------------------------------------------------------------- reward bridge


def _build_reward_path():
    """The same (codebook, reward, scorer) triple the CPU experiments use."""
    from app.components.kinematic_bicycle import SkbmConfig
    from app.components.physics_reward import EmpiricalEnvelope, PhysicsReward, RewardWeights
    from app.components.pidpm_scorer import PiDpmScorer
    from app.policy.decode import CodebookConfig, MotionCodebook

    skbm = SkbmConfig()
    codebook = MotionCodebook(skbm, CodebookConfig())
    reward = PhysicsReward(skbm, weights=RewardWeights(), envelope=EmpiricalEnvelope())
    scorer = PiDpmScorer(device="cuda")
    return codebook, reward, scorer


_INT_RE = re.compile(r"\d+")


def parse_motion_ids(text: str, vocab_size: int, horizon: int):
    """Parse the model's completion into up to `horizon` motion-primitive ids.

    The policy is prompted to emit space-separated integers in [0, vocab). We take
    the first `horizon` integers, clamp into range, and (if the model emitted too
    few) pad with the no-op-ish middle id so the trajectory is always scorable.
    Returns an int list of length `horizon`.
    """
    import numpy as np

    ids = [int(m.group()) for m in _INT_RE.finditer(text)][:horizon]
    ids = [min(max(0, i), vocab_size - 1) for i in ids]
    if len(ids) < horizon:
        ids += [vocab_size // 2] * (horizon - len(ids))
    return np.asarray(ids, dtype=np.int64)


def trajectory_reward(text, codebook, reward, scorer, vocab_size, horizon):
    """Completion text -> (reward_total, hard_violation_magnitude)."""
    ids = parse_motion_ids(text, vocab_size, horizon)
    states = codebook.tokens_to_states(ids)
    breakdown = reward.score(states, pi_dpm_log_prob=scorer.log_prob(states))
    return float(breakdown.total), float(-breakdown.hard)


def build_prompt(tokenizer, vocab_size: int, horizon: int) -> str:
    """Chat-templated instruction asking for a motion-primitive trajectory."""
    user = (
        f"Output a physically feasible vehicle trajectory as {horizon} integers in "
        f"[0,{vocab_size}), space-separated, each selecting a motion primitive "
        f"(acceleration and steering). Output only the integers."
    )
    msgs = [{"role": "user", "content": user}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# --------------------------------------------------------------- logprobs / GRPO


def completion_logprobs(model, input_ids, attn_mask, prompt_lens):
    """Sum of per-token log p(generated token) over the completion span.

    input_ids: (N, L) full prompt+completion. prompt_lens: (N,) prompt length per row.
    Returns (N,) summed completion logprob under `model` (adapter state as set by the
    caller). Shifted so position t predicts token t+1, standard causal LM scoring.
    """
    import torch

    out = model(input_ids=input_ids, attention_mask=attn_mask)
    logits = out.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (N, L-1)
    _, lm1 = tok_logp.shape
    pos = torch.arange(lm1, device=input_ids.device).unsqueeze(0)
    # completion tokens are positions >= prompt_len-1 in the shifted frame, masked by attn
    comp_mask = (pos >= (prompt_lens.unsqueeze(1) - 1)) & (attn_mask[:, 1:] > 0)
    return (tok_logp * comp_mask).sum(dim=1)


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.algo != "grpo":
        raise NotImplementedError(
            f"--algo {args.algo} not implemented in this entrypoint; grpo is the EXP-5 "
            "headline. PPO/DPO reuse the same reward bridge and policy; add them next."
        )

    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM",
        ),
    )
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    codebook, reward, scorer = _build_reward_path()
    vocab_size, horizon = codebook.vocab_size, args.horizon
    prompt = build_prompt(tok, vocab_size, horizon)
    kl_coef = 0.1

    for step in range(args.steps):
        # ---- rollouts: K samples for each of B identical prompts (group baseline) ----
        n = args.batch_prompts * args.group_size
        enc = tok([prompt] * n, return_tensors="pt", padding=True).to("cuda")
        prompt_len = int(enc["attention_mask"][0].sum().item())
        with torch.no_grad():
            gen = model.generate(
                **enc, do_sample=True, temperature=args.temperature,
                max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id,
            )
        full = gen  # (n, prompt_len + new)
        attn = (full != tok.pad_token_id).long()
        prompt_lens = torch.full((n,), prompt_len, device="cuda")
        texts = tok.batch_decode(gen[:, prompt_len:], skip_special_tokens=True)

        # ---- reward (reuse the physics path) ----
        rewards = np.zeros(n, dtype=np.float32)
        viols = np.zeros(n, dtype=np.float32)
        for i, t in enumerate(texts):
            rewards[i], viols[i] = trajectory_reward(t, codebook, reward, scorer, vocab_size, horizon)
        R = torch.tensor(rewards, device="cuda").view(args.batch_prompts, args.group_size)
        adv = (R - R.mean(dim=1, keepdim=True)) / (R.std(dim=1, keepdim=True) + 1e-6)
        adv = adv.view(n)

        # ---- old + reference logprobs (no grad) ----
        with torch.no_grad():
            logp_old = completion_logprobs(model, full, attn, prompt_lens)
            with model.disable_adapter():
                logp_ref = completion_logprobs(model, full, attn, prompt_lens)

        # ---- GRPO update (one epoch): clipped surrogate + KL to ref ----
        logp_new = completion_logprobs(model, full, attn, prompt_lens)
        ratio = (logp_new - logp_old).exp()
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef) * adv
        pg_loss = -torch.min(unclipped, clipped).mean()
        kl = (logp_new - logp_ref).mean()
        loss = pg_loss + kl_coef * kl
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        # adapt KL coefficient toward the target (same idea as AdaptiveKLController)
        kl_val = float(kl.detach())
        kl_coef *= 1.1 if kl_val > 2 * args.target_kl else (0.9 if kl_val < args.target_kl / 2 else 1.0)

        if step % args.log_every == 0:
            rec = {
                "step": step, "reward_mean": float(rewards.mean()),
                "hard_violation_rate": float((viols > 0).mean()),
                "kl": kl_val, "loss": float(loss.item()), "kl_coef": kl_coef,
            }
            print(json.dumps(rec), flush=True)
            (out_dir / "metrics.jsonl").open("a").write(json.dumps(rec) + "\n")

    model.save_pretrained(str(out_dir / "lora_adapter"))
    print(f"saved LoRA adapter to {out_dir / 'lora_adapter'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
