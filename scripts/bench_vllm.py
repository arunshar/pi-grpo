"""vLLM serving throughput / latency benchmark vs HF generate (PC-RF Ray paper).

Measures, on ONE GPU with an IDENTICAL model / prompt set / decoding config:

  1. vLLM offline-engine aggregate throughput (output tokens/sec) and per-request
     p50/p95 latency, swept over a few batch (concurrency) settings.
  2. A Hugging Face `AutoModelForCausalLM.generate` baseline: throughput and
     per-request latency, batched the way a naive serving loop would.
  3. The vLLM/HF speedup (tokens/sec ratio).

Honest-science contract for peer review:
  - SAME model weights (Qwen/Qwen2-7B-Instruct, bfloat16), SAME GPU, SAME prompt
    set, SAME decoding (greedy: temperature=0, fixed max_new_tokens). The only
    independent variable is the serving engine (and, within vLLM, the batch size).
  - We report aggregate output-token throughput and the latency distribution, not
    a single cherry-picked number. The prompt set and all knobs are logged into
    the JSON so the run is reproducible.
  - NOTHING here fabricates results: every number printed comes from a real
    forward pass on the allocated GPU. Run with `--smoke` to do a tiny correctness
    pass (few prompts, few tokens); the full run uses the defaults.

Usage (inside the LLM container, on an H100):
    python3 scripts/bench_vllm.py \
        --model Qwen/Qwen2-7B-Instruct \
        --num-prompts 128 --max-new-tokens 128 \
        --vllm-batch-sizes 1,8,32,128 \
        --out results/bench_vllm.json

    # cheap correctness smoke (still a real GPU forward, just tiny):
    python3 scripts/bench_vllm.py --smoke --out results/bench_vllm_smoke.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any


# --------------------------------------------------------------------------- #
# Prompt set: a fixed, deterministic bank so vLLM and HF see identical inputs.
# --------------------------------------------------------------------------- #
_PROMPT_BANK = [
    "Explain the difference between latency and throughput in one paragraph.",
    "Summarize the plot of a generic detective novel in five sentences.",
    "Write a short function in Python that returns the n-th Fibonacci number.",
    "List three trade-offs between batch size and tail latency when serving an LLM.",
    "Describe how a kinematic bicycle model approximates vehicle motion.",
    "What is prefix caching and why does it speed up repeated-prefix decoding?",
    "Give a concise explanation of reinforcement learning from human feedback.",
    "Compare continuous batching with static batching for inference servers.",
    "Explain why bfloat16 is often preferred over float16 for transformer training.",
    "Outline the steps of a single GRPO policy-update iteration.",
    "What is the role of the KV cache during autoregressive decoding?",
    "Describe one way to detect an abnormal gap in a GPS trajectory.",
    "Explain paged attention and the memory problem it solves.",
    "Write a haiku about distributed systems.",
    "Summarize the CAP theorem for a new engineer.",
    "Give three reasons tail latency matters more than mean latency in serving.",
]


def build_prompts(n: int) -> list[str]:
    """Deterministic prompt set of length ``n`` by cycling the bank with an index
    suffix, so every prompt is distinct but reproducible across engines/runs."""
    out: list[str] = []
    for i in range(n):
        base = _PROMPT_BANK[i % len(_PROMPT_BANK)]
        out.append(f"[req {i:04d}] {base}")
    return out


def apply_chat_template(tokenizer, prompts: list[str]) -> list[str]:
    """Render each user prompt through the model's chat template so vLLM and HF
    decode the SAME token stream. Falls back to the raw prompt if no template."""
    rendered: list[str] = []
    for p in prompts:
        try:
            rendered.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        except Exception:
            rendered.append(p)
    return rendered


def pct(samples: list[float], q: float) -> float:
    """Linear-interpolation percentile (q in [0,1]) over a list of floats."""
    if not samples:
        return float("nan")
    s = sorted(samples)
    if len(s) == 1:
        return s[0]
    idx = q * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


@dataclass
class EngineResult:
    engine: str
    batch_size: int
    num_prompts: int
    max_new_tokens: int
    total_output_tokens: int
    wall_s: float
    throughput_tok_s: float  # aggregate output tokens / wall second
    req_latency_p50_s: float
    req_latency_p95_s: float
    req_latency_mean_s: float
    notes: str = ""


@dataclass
class BenchReport:
    model: str
    dtype: str
    gpu: str
    num_prompts: int
    max_new_tokens: int
    temperature: float
    vllm_batch_sizes: list[int]
    hf_batch_size: int
    smoke: bool
    env: dict[str, Any] = field(default_factory=dict)
    vllm: list[dict[str, Any]] = field(default_factory=list)
    hf: dict[str, Any] | None = None
    speedup_best_vllm_over_hf: float | None = None


# --------------------------------------------------------------------------- #
# vLLM offline-engine benchmark.
# --------------------------------------------------------------------------- #
def bench_vllm(
    model: str,
    rendered_prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    batch_sizes: list[int],
    dtype: str,
    gpu_mem_util: float,
    max_model_len: int,
) -> list[EngineResult]:
    from vllm import LLM, SamplingParams

    # One engine instance reused across batch settings (loads weights once).
    llm = LLM(
        model=model,
        dtype=dtype,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        enforce_eager=False,
        trust_remote_code=True,
    )
    # Greedy decoding for a deterministic, fair comparison with HF greedy.
    sp = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_new_tokens,
        # ignore_eos so every request emits exactly max_new_tokens; this makes the
        # token-throughput comparison apples-to-apples across engines and prompts.
        ignore_eos=True,
    )

    results: list[EngineResult] = []
    for bs in batch_sizes:
        # vLLM does its own continuous batching internally; we control the number
        # of prompts submitted concurrently to expose the throughput/latency curve.
        # Warm up once at this batch size to amortize CUDA-graph capture.
        _ = llm.generate(rendered_prompts[: min(bs, len(rendered_prompts))], sp)

        per_req_lat: list[float] = []
        total_out_tokens = 0
        t0 = time.perf_counter()
        # Submit in chunks of size bs; within a chunk vLLM batches them together.
        for start in range(0, len(rendered_prompts), bs):
            chunk = rendered_prompts[start : start + bs]
            c0 = time.perf_counter()
            outs = llm.generate(chunk, sp)
            c_wall = time.perf_counter() - c0
            # Per-request latency: vLLM returns the whole chunk together, so the
            # honest per-request latency under this batch is the chunk wall time
            # (each request in the chunk waits for the slowest in its batch).
            for o in outs:
                per_req_lat.append(c_wall)
                total_out_tokens += len(o.outputs[0].token_ids)
        wall = time.perf_counter() - t0

        results.append(
            EngineResult(
                engine="vllm",
                batch_size=bs,
                num_prompts=len(rendered_prompts),
                max_new_tokens=max_new_tokens,
                total_output_tokens=total_out_tokens,
                wall_s=wall,
                throughput_tok_s=(total_out_tokens / wall) if wall > 0 else 0.0,
                req_latency_p50_s=pct(per_req_lat, 0.50),
                req_latency_p95_s=pct(per_req_lat, 0.95),
                req_latency_mean_s=statistics.mean(per_req_lat) if per_req_lat else 0.0,
                notes="offline LLM engine, ignore_eos, greedy",
            )
        )
    # Free the engine's GPU memory before HF loads the same weights.
    del llm
    try:
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass
    return results


# --------------------------------------------------------------------------- #
# Hugging Face generate baseline (same GPU, same prompts, same decoding).
# --------------------------------------------------------------------------- #
def bench_hf(
    model: str,
    rendered_prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    hf_batch_size: int,
    dtype: str,
) -> EngineResult:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        dtype, torch.bfloat16
    )
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # left-pad so generation aligns for batched decode
    mdl = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch_dtype, trust_remote_code=True
    ).to("cuda")
    mdl.eval()

    greedy = temperature == 0.0
    gen_kwargs: dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        do_sample=not greedy,
        pad_token_id=tok.pad_token_id,
        # match vLLM ignore_eos: keep generating to the fixed length
        eos_token_id=None,
    )
    if not greedy:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 1.0

    # Warmup (one tiny batch) so CUDA init / kernels are not charged to the timer.
    with torch.no_grad():
        w = tok(rendered_prompts[:1], return_tensors="pt", padding=True).to("cuda")
        _ = mdl.generate(**w, max_new_tokens=4, do_sample=False,
                         pad_token_id=tok.pad_token_id, eos_token_id=None)
    torch.cuda.synchronize()

    per_req_lat: list[float] = []
    total_out_tokens = 0
    t0 = time.perf_counter()
    for start in range(0, len(rendered_prompts), hf_batch_size):
        chunk = rendered_prompts[start : start + hf_batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True).to("cuda")
        in_len = enc["input_ids"].shape[1]
        c0 = time.perf_counter()
        with torch.no_grad():
            out = mdl.generate(**enc, **gen_kwargs)
        torch.cuda.synchronize()
        c_wall = time.perf_counter() - c0
        # Count only the NEW tokens per request (exclude left padding + prompt).
        new_tokens = out.shape[1] - in_len
        for _ in range(out.shape[0]):
            per_req_lat.append(c_wall)
            total_out_tokens += max(new_tokens, 0)
    wall = time.perf_counter() - t0

    return EngineResult(
        engine="hf_generate",
        batch_size=hf_batch_size,
        num_prompts=len(rendered_prompts),
        max_new_tokens=max_new_tokens,
        total_output_tokens=total_out_tokens,
        wall_s=wall,
        throughput_tok_s=(total_out_tokens / wall) if wall > 0 else 0.0,
        req_latency_p50_s=pct(per_req_lat, 0.50),
        req_latency_p95_s=pct(per_req_lat, 0.95),
        req_latency_mean_s=statistics.mean(per_req_lat) if per_req_lat else 0.0,
        notes="AutoModelForCausalLM.generate, left-pad, greedy, eos disabled",
    )


def gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "unknown"


def print_table(report: BenchReport) -> None:
    print("\n=== bench_vllm: Qwen2-7B serving throughput/latency ===")
    print(f"model={report.model} dtype={report.dtype} gpu={report.gpu}")
    print(
        f"num_prompts={report.num_prompts} max_new_tokens={report.max_new_tokens} "
        f"temperature={report.temperature} smoke={report.smoke}"
    )
    hdr = f"{'engine':12s} {'bs':>4s} {'tok/s':>10s} {'p50_s':>8s} {'p95_s':>8s} {'mean_s':>8s} {'wall_s':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for r in report.vllm:
        print(
            f"{'vllm':12s} {r['batch_size']:>4d} {r['throughput_tok_s']:>10.1f} "
            f"{r['req_latency_p50_s']:>8.3f} {r['req_latency_p95_s']:>8.3f} "
            f"{r['req_latency_mean_s']:>8.3f} {r['wall_s']:>8.2f}"
        )
    if report.hf is not None:
        r = report.hf
        print(
            f"{'hf_generate':12s} {r['batch_size']:>4d} {r['throughput_tok_s']:>10.1f} "
            f"{r['req_latency_p50_s']:>8.3f} {r['req_latency_p95_s']:>8.3f} "
            f"{r['req_latency_mean_s']:>8.3f} {r['wall_s']:>8.2f}"
        )
    if report.speedup_best_vllm_over_hf is not None:
        print(
            f"\nbest vLLM throughput / HF throughput = "
            f"{report.speedup_best_vllm_over_hf:.2f}x"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="vLLM vs HF serving benchmark")
    ap.add_argument("--model", default=os.environ.get("PIGRPO_BASE_MODEL", "Qwen/Qwen2-7B-Instruct"))
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0, help="0.0 = greedy")
    ap.add_argument("--vllm-batch-sizes", default="1,8,32,128",
                    help="comma-separated concurrency settings for vLLM")
    ap.add_argument("--hf-batch-size", type=int, default=8,
                    help="batch size for the HF generate baseline")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--skip-hf", action="store_true",
                    help="benchmark vLLM only (e.g. if HF baseline already recorded)")
    ap.add_argument("--skip-vllm", action="store_true")
    ap.add_argument("--out", default="results/bench_vllm.json")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny real-GPU correctness pass: 4 prompts, 16 new tokens, bs {1,4}")
    args = ap.parse_args()

    if args.smoke:
        args.num_prompts = min(args.num_prompts, 4)
        args.max_new_tokens = min(args.max_new_tokens, 16)
        args.vllm_batch_sizes = "1,4"
        args.hf_batch_size = min(args.hf_batch_size, 4)
        args.max_model_len = min(args.max_model_len, 1024)

    vllm_batch_sizes = [int(x) for x in args.vllm_batch_sizes.split(",") if x.strip()]

    prompts = build_prompts(args.num_prompts)
    # Render through the chat template ONCE, shared by both engines.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rendered = apply_chat_template(tok, prompts)

    report = BenchReport(
        model=args.model,
        dtype=args.dtype,
        gpu=gpu_name(),
        num_prompts=args.num_prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        vllm_batch_sizes=vllm_batch_sizes,
        hf_batch_size=args.hf_batch_size,
        smoke=args.smoke,
        env={
            "python": platform.python_version(),
            "hostname": platform.node(),
            "hf_home": os.environ.get("HF_HOME", ""),
        },
    )
    try:
        import torch
        report.env["torch"] = torch.__version__
        report.env["cuda"] = torch.version.cuda
    except Exception:
        pass
    try:
        import vllm
        report.env["vllm"] = vllm.__version__
    except Exception:
        pass
    try:
        import transformers
        report.env["transformers"] = transformers.__version__
    except Exception:
        pass

    if not args.skip_vllm:
        vllm_results = bench_vllm(
            model=args.model,
            rendered_prompts=rendered,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            batch_sizes=vllm_batch_sizes,
            dtype=args.dtype,
            gpu_mem_util=args.gpu_mem_util,
            max_model_len=args.max_model_len,
        )
        report.vllm = [asdict(r) for r in vllm_results]

    if not args.skip_hf:
        hf_result = bench_hf(
            model=args.model,
            rendered_prompts=rendered,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            hf_batch_size=args.hf_batch_size,
            dtype=args.dtype,
        )
        report.hf = asdict(hf_result)

    # Speedup: best vLLM throughput over the HF baseline throughput.
    if report.vllm and report.hf is not None:
        best_vllm = max(r["throughput_tok_s"] for r in report.vllm)
        hf_tps = report.hf["throughput_tok_s"]
        if hf_tps > 0:
            report.speedup_best_vllm_over_hf = best_vllm / hf_tps

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(asdict(report), f, indent=2)

    print_table(report)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
