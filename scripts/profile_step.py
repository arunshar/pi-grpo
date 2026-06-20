"""HONEST per-op time + MFU profiler for ONE PC-RF / GRPO training step.

This wraps a single (or a few) training step(s) in `torch.profiler` (CPU + CUDA
activities, record_shapes, with_stack) and emits, for the paper's systems table:

  * top-N ops by self-CUDA time (the real bottleneck ranking),
  * total measured step wall-time (median over the timed steps),
  * an analytic MFU estimate = achieved_FLOPs_per_s / peak_TFLOPS, reusing the
    peak-TFLOPS table from mirror/scripts/gpu_estimator.py.

Two targets, both producing the SAME schema so the table is drop-in either way:

  --target llm-grpo  (the paper headline): ONE real GRPO step on the SAME path as
      scripts/grpo_llm_train.py -- Qwen2-7B-Instruct + LoRA, on-policy HF-generate
      rollouts, the physics reward bridge, and the clipped-surrogate + KL update.
      This is the measured result. Needs an 80GB H100 and the HF cache.

  --target toy  (cheap, CPU/GPU): the tiny CausalPolicy + GrpoTrainer.step_update
      from the repo, so the harness validates end to end in seconds on a 2-layer
      model. Used for the byte-compile + srun smoke; NOT a paper number.

WHAT IS MEASURED vs ESTIMATED (be explicit, this is peer review):
  * step time and the per-op CUDA-time ranking are MEASURED by torch.profiler.
  * achieved FLOPs is MEASURED via the profiler's FLOP counter when available
    (with_flops=True / key_averages flops), and otherwise FALLS BACK to an
    analytic 6*N*T*B estimate (clearly flagged as estimated in the output).
  * peak TFLOPS is the vendor analytic dense bf16 peak (from gpu_estimator.GPUS);
    MFU = achieved/peak is therefore an analytic upper-bound ratio, NOT a
    hardware-counter SM-utilization number. Reported as such.

Outputs: results/profile_step.json and results/profile_step.md (markdown table).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import statistics
import sys
import time
from pathlib import Path

# ----------------------------------------------------------------- peak TFLOPS
# Reuse the EXACT peak-TFLOPS table from the MIRROR gpu_estimator so the MFU
# denominator matches the rest of the paper's right-sizing. Loaded by path so we
# do not depend on mirror being importable; falls back to an inline copy.
_GPU_PEAK_TFLOPS = {
    "a100": 312.0, "a100-40": 312.0, "h100": 989.0, "v100": 125.0,
}


def _load_gpu_peaks() -> dict:
    est_path = Path.home() / "mirror" / "scripts" / "gpu_estimator.py"
    if not est_path.exists():
        return dict(_GPU_PEAK_TFLOPS)
    try:
        spec = importlib.util.spec_from_file_location("gpu_estimator", est_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        # GPUS values are (peak_tflops, mem_gb, avail); take the peak.
        return {k: v[0] for k, v in mod.GPUS.items()}
    except Exception:
        return dict(_GPU_PEAK_TFLOPS)


def _detect_gpu_key(peaks: dict) -> tuple[str, float]:
    """Map the live device name to a key in the peak table. Returns (key, peak_tflops)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu", 0.0
        name = torch.cuda.get_device_name(0).lower()
    except Exception:
        return "cpu", 0.0
    if "h100" in name:
        return "h100", peaks.get("h100", 989.0)
    if "a100" in name:
        return "a100", peaks.get("a100", 312.0)
    if "v100" in name:
        return "v100", peaks.get("v100", 125.0)
    # unknown CUDA device: report it but use a conservative A100 peak so MFU is a
    # (likely over-)estimate rather than a crash; flagged in the JSON.
    return name.replace(" ", "_"), peaks.get("a100", 312.0)


# ----------------------------------------------------------------- timing util


def _sync():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


# ----------------------------------------------------------------- toy target


def _build_toy_step(device: str, horizon: int, batch_prompts: int, group_size: int):
    """The tiny in-repo GRPO step: CausalPolicy + GrpoTrainer.step_update.

    Returns (run_one_step_callable, flops_per_step_estimate, meta).
    The FLOPs estimate is the analytic 6*N*T*B fallback (the toy is for smoke,
    not a paper number), with N = trainable params, T = total seq len, B = B*K.
    """
    import torch

    from app.policy.model import PolicyConfig, build_policy_pair
    from app.trainers.grpo_trainer import GrpoBatch, GrpoConfig, GrpoTrainer

    cfg = PolicyConfig(vocab_size=25, d_model=64, n_layers=2, n_heads=4, d_ff=128, max_len=64)
    policy, ref = build_policy_pair(cfg)
    policy.to(device)
    ref.to(device)
    trainer = GrpoTrainer(policy=policy, ref_policy=ref, cfg=GrpoConfig(group_size=group_size))

    b, k, t_p, t_r = batch_prompts, group_size, 4, horizon
    g = torch.Generator(device="cpu").manual_seed(0)
    prompt_ids = torch.randint(0, cfg.vocab_size, (b, t_p), generator=g).to(device)
    rollout_ids = torch.randint(0, cfg.vocab_size, (b, k, t_r), generator=g).to(device)
    with torch.no_grad():
        old = policy.log_prob_token(prompt_ids, rollout_ids).detach()
        rlp = ref.log_prob_token(prompt_ids, rollout_ids).detach()
    rewards = torch.randn(b, k, generator=g).to(device)

    def run_one_step():
        batch = GrpoBatch(
            prompt_ids=prompt_ids, rollout_ids=rollout_ids,
            action_logp_old=old, rewards=rewards, ref_logp=rlp,
        )
        return trainer.step_update(batch)

    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    seq_len = t_p + t_r
    n_seq = b * k
    # 6*N*T per token (fwd+bwd), times tokens = seq_len * n_seq. Forward passes in
    # the step: new_logp (1). old/ref are precomputed. So ~1 fwd+bwd over n_seq seqs.
    flops_est = 6.0 * n_params * seq_len * n_seq
    meta = {
        "n_trainable_params": n_params, "seq_len": seq_len, "n_sequences": n_seq,
        "model": f"CausalPolicy(d={cfg.d_model},L={cfg.n_layers})",
    }
    return run_one_step, flops_est, meta


# ----------------------------------------------------------------- llm target


def _build_llm_step(args, device: str):
    """ONE real GRPO LLM step, mirroring scripts/grpo_llm_train.py exactly.

    Returns (run_one_step_callable, flops_per_step_estimate, meta). The callable
    performs: K rollouts (HF generate) for B prompts, the physics reward bridge,
    old+ref+new completion-logprob forwards, and one clipped-surrogate+KL backward
    and optimizer step -- i.e. a complete training step, the thing the table is about.
    """
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Reuse the trainer's helpers so we profile the SAME code, not a re-implementation.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import grpo_llm_train as T  # the actual EXP-5 trainer module

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map=device
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

    codebook, reward, scorer = T._build_reward_path()
    vocab_size, horizon = codebook.vocab_size, args.horizon
    prompt = T.build_prompt(tok, vocab_size, horizon)
    n = args.batch_prompts * args.group_size

    # config-derived sizes for the analytic FLOPs fallback (forward only; the
    # profiler's measured FLOPs is preferred when available).
    cfg = model.config if hasattr(model, "config") else model.base_model.config
    hidden = getattr(cfg, "hidden_size", 3584)
    n_layer = getattr(cfg, "num_hidden_layers", 28)
    # dense transformer fwd ~ 2 * n_params_active * tokens; here we approximate the
    # FULL base forward (the dominant cost) as 2*P_base*tokens per forward pass.
    p_base = sum(p.numel() for p in model.parameters())

    def run_one_step():
        enc = tok([prompt] * n, return_tensors="pt", padding=True).to(device)
        prompt_len = int(enc["attention_mask"][0].sum().item())
        with torch.no_grad():
            gen = model.generate(
                **enc, do_sample=True, temperature=args.temperature, top_p=args.top_p,
                max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id,
            )
        full = gen
        attn = (full != tok.pad_token_id).long()
        prompt_lens = torch.full((n,), prompt_len, device=device)
        texts = tok.batch_decode(gen[:, prompt_len:], skip_special_tokens=True)

        rewards = np.zeros(n, dtype=np.float32)
        for i, t in enumerate(texts):
            rewards[i], _ = T.trajectory_reward(t, codebook, reward, scorer, vocab_size, horizon)
        R = torch.tensor(rewards, device=device).view(args.batch_prompts, args.group_size)
        adv = (R - R.mean(dim=1, keepdim=True)) / (R.std(dim=1, keepdim=True) + 1e-6)
        adv = adv.view(n)

        with torch.no_grad():
            logp_old = T.completion_logprobs(model, full, attn, prompt_lens)
            with model.disable_adapter():
                logp_ref = T.completion_logprobs(model, full, attn, prompt_lens)

        logp_new = T.completion_logprobs(model, full, attn, prompt_lens)
        ratio = (logp_new - logp_old).exp()
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef) * adv
        pg_loss = -torch.min(unclipped, clipped).mean()
        kl = (logp_new - logp_ref).mean()
        loss = pg_loss + args.kl_coef_init * kl
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        opt.zero_grad()
        # rough seq length for the FLOPs fallback: prompt + generated
        seq_len_full = int(full.shape[1])
        return {"loss": float(loss.item()), "seq_len_full": seq_len_full}

    # analytic FLOPs fallback for ONE training step (rollout fwd + 3 scoring fwds + 1 bwd).
    # We approximate with the measured full seq length captured at run time; here a
    # placeholder using config (refined after the timed run from the returned seq_len).
    def flops_fn(seq_len_full: int) -> float:
        tokens = n * seq_len_full
        # generation: max_new_tokens forwards but with KV cache ~ 2*P*tokens total;
        # plus 3 full-seq scoring forwards (old/ref/new) ~ 3*2*P*tokens; plus bwd ~2x the
        # graphed (new) forward ~ 2*2*P*tokens. Total ~ (2 + 6 + 4) * P * tokens = 12*P*tokens.
        return 12.0 * p_base * tokens

    meta = {
        "n_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "n_base_params": p_base, "hidden_size": hidden, "n_layers": n_layer,
        "n_sequences": n, "max_new_tokens": args.max_new_tokens,
        "model": args.base_model,
    }
    # stash the dynamic flops function on meta for the runner
    return run_one_step, flops_fn, meta


# ----------------------------------------------------------------- profiler run


def _profile(run_one_step, n_warmup: int, n_active: int, with_flops: bool, trace_path: str | None):
    """Run warmup + timed steps under torch.profiler. Returns (prof, per_step_times, last_out)."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    # warmup OUTSIDE the profiler (allocator, cudnn autotune, LoRA graph build).
    for _ in range(n_warmup):
        run_one_step()
    _sync()

    per_step_times: list[float] = []
    last_out = None
    prof_kwargs = dict(
        activities=activities, record_shapes=True, with_stack=True, profile_memory=True,
    )
    # with_flops is supported on recent torch; guard for older builds.
    try:
        prof_ctx = profile(with_flops=with_flops, **prof_kwargs)
    except TypeError:
        prof_ctx = profile(**prof_kwargs)

    with prof_ctx as prof:
        for _ in range(n_active):
            _sync()
            t0 = time.perf_counter()
            last_out = run_one_step()
            _sync()
            per_step_times.append(time.perf_counter() - t0)
    if trace_path:
        try:
            prof.export_chrome_trace(trace_path)
        except Exception:
            pass
    return prof, per_step_times, last_out


def _top_ops(prof, top_n: int) -> tuple[list[dict], float, float]:
    """Extract top-N ops by self CUDA time (falls back to self CPU time on CPU runs).

    Returns (rows, total_self_cuda_us, total_measured_flops). FLOPs is summed from
    the profiler's per-op flops if present (0.0 if the build lacks the counter).
    """
    ka = prof.key_averages()
    have_cuda = any(getattr(e, "self_cuda_time_total", 0) for e in ka)

    def keyfn(e):
        return getattr(e, "self_cuda_time_total", 0) if have_cuda else e.self_cpu_time_total

    ranked = sorted(ka, key=keyfn, reverse=True)
    total_cuda = sum(getattr(e, "self_cuda_time_total", 0) for e in ka)
    total_cpu = sum(e.self_cpu_time_total for e in ka)
    total_flops = 0.0
    for e in ka:
        f = getattr(e, "flops", 0) or 0
        total_flops += float(f)

    rows = []
    for e in ranked[:top_n]:
        rows.append({
            "op": e.key,
            "self_cuda_us": float(getattr(e, "self_cuda_time_total", 0)),
            "self_cpu_us": float(e.self_cpu_time_total),
            "cuda_total_us": float(getattr(e, "cuda_time_total", 0)),
            "cpu_total_us": float(e.cpu_time_total),
            "count": int(e.count),
            "flops": float(getattr(e, "flops", 0) or 0),
        })
    return rows, (total_cuda if have_cuda else total_cpu), total_flops


def _write_markdown(md_path: Path, payload: dict) -> None:
    g = payload["gpu"]
    s = payload["step"]
    m = payload["mfu"]
    lines = []
    lines.append(f"# Profiler pass: one {payload['target']} training step\n")
    lines.append(f"- device: `{g['device_name']}` (key `{g['gpu_key']}`, "
                 f"peak {g['peak_tflops']:.0f} TFLOPS bf16 dense)")
    lines.append(f"- torch {payload['env']['torch']}, "
                 f"CUDA available: {payload['env']['cuda_available']}")
    lines.append(f"- timed steps: {s['n_active']} (after {s['n_warmup']} warmup); "
                 f"median step time **{s['median_step_s']*1e3:.1f} ms** "
                 f"(min {s['min_step_s']*1e3:.1f} / max {s['max_step_s']*1e3:.1f} ms)")
    flops_src = m["flops_source"]
    lines.append(f"- achieved FLOPs/step: {m['flops_per_step']:.3e} (source: {flops_src})")
    lines.append(f"- achieved {m['achieved_tflops']:.1f} TFLOP/s -> "
                 f"**MFU {m['mfu']*100:.1f}%** (analytic peak denominator)\n")
    lines.append("## Top ops by self-CUDA time (MEASURED)\n")
    unit = "self CUDA us" if g["cuda_available"] else "self CPU us"
    lines.append(f"| rank | op | {unit} | calls | profiler FLOPs |")
    lines.append("|---|---|---:|---:|---:|")
    for i, r in enumerate(payload["top_ops"], 1):
        t = r["self_cuda_us"] if g["cuda_available"] else r["self_cpu_us"]
        lines.append(f"| {i} | `{r['op']}` | {t:,.0f} | {r['count']} | "
                     f"{r['flops']:.2e} |")
    lines.append("")
    lines.append("## Caveats\n")
    lines.append("- Step time and the op ranking are measured by `torch.profiler`.")
    lines.append("- MFU uses an ANALYTIC vendor peak (gpu_estimator) as the denominator; "
                 "it is an upper-bound ratio, not a hardware SM-utilization counter.")
    if flops_src.startswith("analytic"):
        lines.append("- Achieved FLOPs is an ANALYTIC estimate (this torch build did not "
                     "expose a per-op FLOP counter); treat MFU as order-of-magnitude.")
    md_path.write_text("\n".join(lines) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="profile_step", description=__doc__)
    ap.add_argument("--target", choices=["toy", "llm-grpo"], default="toy")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--active", type=int, default=3, help="timed steps to profile")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--out-dir", default=None, help="default: <repo>/results")
    ap.add_argument("--trace", action="store_true", help="also dump a chrome trace")
    # llm-grpo knobs (mirror grpo_llm_train.py defaults but smaller for a single step)
    ap.add_argument("--base-model", default="Qwen/Qwen2-7B-Instruct")
    ap.add_argument("--batch-prompts", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--lr", type=float, default=7e-6)
    ap.add_argument("--clip-coef", type=float, default=0.2)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.1)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--kl-coef-init", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    return ap


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    import torch

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    out_dir = Path(args.out_dir) if args.out_dir else (repo_root / "results")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    peaks = _load_gpu_peaks()
    gpu_key, peak_tflops = _detect_gpu_key(peaks)

    # build the step closure
    flops_fn = None
    if args.target == "toy":
        run_one_step, flops_est, meta = _build_toy_step(
            device, args.horizon, args.batch_prompts, args.group_size
        )
    else:
        run_one_step, flops_fn, meta = _build_llm_step(args, device)
        flops_est = None  # computed after the run from the measured seq length

    prof, per_step_times, last_out = _profile(
        run_one_step, args.warmup, args.active, with_flops=True,
        trace_path=str(out_dir / "profile_step_trace.json") if args.trace else None,
    )
    top_ops, total_self_us, measured_flops = _top_ops(prof, args.top_n)

    median_step = statistics.median(per_step_times)

    # ---- FLOPs: prefer profiler-measured, else analytic ----
    if measured_flops > 0:
        flops_per_step = measured_flops
        flops_source = "profiler (with_flops)"
    elif args.target == "llm-grpo" and flops_fn is not None and last_out:
        flops_per_step = flops_fn(int(last_out.get("seq_len_full", 0)) or 1)
        flops_source = "analytic 12*P*tokens (profiler FLOP counter unavailable)"
    else:
        flops_per_step = float(flops_est or 0.0)
        flops_source = "analytic 6*N*T*B (toy fallback)"

    achieved_flops_per_s = flops_per_step / max(median_step, 1e-9)
    achieved_tflops = achieved_flops_per_s / 1e12
    mfu = (achieved_tflops / peak_tflops) if peak_tflops > 0 else 0.0

    payload = {
        "target": args.target,
        "env": {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "python": platform.python_version(),
            "host": platform.node(),
        },
        "gpu": {
            "device_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
            "gpu_key": gpu_key,
            "peak_tflops": peak_tflops,
            "cuda_available": torch.cuda.is_available(),
        },
        "meta": meta,
        "step": {
            "n_warmup": args.warmup,
            "n_active": args.active,
            "median_step_s": median_step,
            "min_step_s": min(per_step_times),
            "max_step_s": max(per_step_times),
            "per_step_s": per_step_times,
            "total_self_op_us": total_self_us,
        },
        "mfu": {
            "flops_per_step": flops_per_step,
            "flops_source": flops_source,
            "achieved_tflops": achieved_tflops,
            "mfu": mfu,
        },
        "top_ops": top_ops,
        "caveats": [
            "step time + op ranking are MEASURED by torch.profiler",
            "MFU denominator is the analytic vendor bf16 dense peak (gpu_estimator), "
            "so MFU is an upper-bound ratio, not an SM-utilization counter",
            ("achieved FLOPs is analytic when the torch build lacks a per-op FLOP "
             "counter" if flops_source.startswith("analytic") else
             "achieved FLOPs is profiler-measured"),
        ],
    }

    json_path = out_dir / "profile_step.json"
    md_path = out_dir / "profile_step.md"
    json_path.write_text(json.dumps(payload, indent=2))
    _write_markdown(md_path, payload)

    print(json.dumps({
        "target": args.target, "device": payload["gpu"]["device_name"],
        "median_step_ms": median_step * 1e3, "mfu_pct": mfu * 100,
        "flops_source": flops_source, "top_op": (top_ops[0]["op"] if top_ops else None),
        "json": str(json_path), "md": str(md_path),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
