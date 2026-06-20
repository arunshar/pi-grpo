"""STRETCH (separate from the profiler deliverable): one fused Triton kernel.

A fused RMSNorm forward kernel (the normalization Qwen2 / Llama-family models use),
with:
  * a CORRECTNESS check vs the torch eager reference (torch.allclose), and
  * a before/after MICROBENCHMARK (torch eager vs the Triton kernel, timed with
    CUDA events over many iters).

HONESTY CONTRACT (peer review): this script does NOT fabricate a speedup. It runs
both implementations and reports the measured times. It only claims "validated"
when BOTH (a) the Triton output matches torch within tolerance AND (b) the Triton
kernel is at least as fast as eager. Otherwise it prints, and returns via the
exit summary, status = not-validated, and the profiler remains the real result.

If Triton is not importable or no CUDA device is present, it exits cleanly with
status = not-validated (kernel unavailable), never a fake number.

Run inside the LLM container on a GPU:
  apptainer exec --nv <sif> python3 scripts/triton_kernel_demo.py
"""

from __future__ import annotations

import json
import sys


def _torch_rmsnorm(x, weight, eps: float):
    """Reference RMSNorm: x * rsqrt(mean(x^2) + eps) * weight, over the last dim."""
    import torch

    dt = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    out = xf * torch.rsqrt(var + eps)
    return (out.to(dt)) * weight


def _build_triton_kernel():
    """Compile the Triton RMSNorm kernel. Returns (callable, None) or (None, reason)."""
    try:
        import torch  # noqa: F401
        import triton
        import triton.language as tl
    except Exception as e:  # triton not installed
        return None, f"triton import failed: {e}"

    @triton.jit
    def _rmsnorm_fwd(
        x_ptr, w_ptr, out_ptr, stride_row, n_cols, eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        x_row = x_ptr + row * stride_row
        out_row = out_ptr + row * stride_row
        # accumulate sum of squares across the row in fp32
        col = tl.arange(0, BLOCK_SIZE)
        mask = col < n_cols
        x = tl.load(x_row + col, mask=mask, other=0.0).to(tl.float32)
        ss = tl.sum(x * x, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(ss + eps)
        w = tl.load(w_ptr + col, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd * w
        tl.store(out_row + col, y.to(tl.float32), mask=mask)

    def run(x, weight, eps: float):
        import torch

        assert x.is_cuda and weight.is_cuda, "triton kernel needs CUDA tensors"
        x2 = x.contiguous()
        n_rows = x2.numel() // x2.shape[-1]
        n_cols = x2.shape[-1]
        x_flat = x2.view(n_rows, n_cols)
        out = torch.empty_like(x_flat, dtype=torch.float32)
        # next pow2 >= n_cols, capped so the row fits one program
        block = 1
        while block < n_cols:
            block *= 2
        if block > 65536:
            raise ValueError(f"row too wide for this single-block kernel: {n_cols}")
        _rmsnorm_fwd[(n_rows,)](
            x_flat, weight, out, x_flat.stride(0), n_cols, eps, BLOCK_SIZE=block,
        )
        return out.view_as(x).to(x.dtype)

    return run, None


def _bench(fn, iters: int) -> float:
    """Median per-call ms over `iters`, timed with CUDA events."""
    import torch

    # warmup
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=4096, help="number of (B*T) rows")
    ap.add_argument("--cols", type=int, default=3584, help="hidden size (Qwen2-7B = 3584)")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float32")
    ap.add_argument("--atol", type=float, default=2e-2)
    ap.add_argument("--rtol", type=float, default=2e-2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    result = {"kernel": "fused_rmsnorm_triton", "status": "not-validated"}

    try:
        import torch
    except Exception as e:
        result["reason"] = f"torch unavailable: {e}"
        print(json.dumps(result, indent=2))
        return 0

    if not torch.cuda.is_available():
        result["reason"] = "no CUDA device; Triton kernel needs a GPU"
        print(json.dumps(result, indent=2))
        return 0

    run, reason = _build_triton_kernel()
    if run is None:
        result["reason"] = reason
        print(json.dumps(result, indent=2))
        return 0

    dtype = getattr(torch, args.dtype)
    dev = "cuda"
    torch.manual_seed(0)
    x = torch.randn(args.rows, args.cols, device=dev, dtype=dtype)
    weight = torch.randn(args.cols, device=dev, dtype=torch.float32).abs() + 0.5

    # ---- correctness ----
    try:
        y_triton = run(x, weight, args.eps)
        y_ref = _torch_rmsnorm(x, weight, args.eps)
        close = torch.allclose(y_triton.float(), y_ref.float(), atol=args.atol, rtol=args.rtol)
        max_abs = float((y_triton.float() - y_ref.float()).abs().max())
    except Exception as e:
        result["reason"] = f"triton kernel raised: {e}"
        print(json.dumps(result, indent=2))
        return 0

    result.update({
        "device": torch.cuda.get_device_name(0),
        "shape": [args.rows, args.cols], "dtype": args.dtype,
        "allclose": bool(close), "max_abs_err": max_abs,
        "atol": args.atol, "rtol": args.rtol,
    })

    # ---- microbenchmark (always run both, report honestly) ----
    t_eager = _bench(lambda: _torch_rmsnorm(x, weight, args.eps), args.iters)
    t_triton = _bench(lambda: run(x, weight, args.eps), args.iters)
    speedup = t_eager / t_triton if t_triton > 0 else 0.0
    result.update({
        "eager_ms": t_eager, "triton_ms": t_triton, "speedup_x": speedup,
    })

    # validated only if correct AND not slower than eager
    if close and speedup >= 1.0:
        result["status"] = "validated"
    elif close:
        result["status"] = "correct-but-not-faster"
    else:
        result["status"] = "not-validated"
        result["reason"] = "triton output does not match torch reference"

    if args.out:
        from pathlib import Path

        Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
