"""Multi-node FSDP / ZeRO-3 strong-scaling trainer for a ~7B causal LM.

This is the inter-node scaling centerpiece for the PC-RF (Ray) paper: it wraps a
HuggingFace causal LM (default Qwen/Qwen2-7B-Instruct) in
torch.distributed.fsdp.FullyShardedDataParallel with ShardingStrategy.FULL_SHARD
(the ZeRO-3 equivalent: parameters, gradients, and optimizer state are all sharded
across the world), mixed precision (bf16), and a transformer-block auto-wrap policy.
It then runs N optimizer steps over a SYNTHETIC token batch and measures honest
throughput (tokens/sec) and per-step wall time, writing results/fsdp_scaling_<world>.json.

The synthetic batch is deliberate: it isolates the distributed-training systems cost
(all-gather of shards in forward, reduce-scatter of grads in backward, inter-node
NCCL over InfiniBand) from any data-pipeline noise, so the 1-node vs 2-node strong-
scaling number reflects the FSDP communication overhead and nothing else. The number
is only a valid strong-scaling point if the 1-node and 2-node runs use the IDENTICAL
per-GPU batch, seq-len, and model (see fsdp_train_multinode.sbatch / fsdp_train_1node.sbatch).

Launch (single node, G GPUs), via torchrun standalone rendezvous:
  torchrun --standalone --nproc_per_node=G scripts/fsdp_train.py --model Qwen/Qwen2-7B-Instruct
Launch (K nodes x G GPUs), via the c10d rendezvous set up by the sbatch wrapper:
  torchrun --nnodes=K --nproc_per_node=G --rdzv_backend=c10d \
           --rdzv_endpoint=$MASTER_ADDR:$PORT scripts/fsdp_train.py --model ...

A tiny toy model (--model toy --toy-layers 2) lets the whole path (FSDP wrap +
auto-wrap policy + one optimizer step) be smoke-tested on CPU/gloo or 2 GPUs with no
network and no 7B weights, which is exactly what we validate before the real GPU run.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import time
from typing import NamedTuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)


# --------------------------------------------------------------------------- env
def env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v is not None and v != "" else default


def setup() -> tuple[int, int, int, int, str]:
    """Init the process group from the torchrun env.

    Returns (rank, world_size, local_rank, nodes, backend). Uses nccl on CUDA,
    gloo on CPU so the same script smoke-tests on a CPU-only login allocation.
    """
    cuda = torch.cuda.is_available()
    backend = "nccl" if cuda else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = env_int("LOCAL_RANK", 0)
    # Nodes = world / local-world-size; torchrun exports LOCAL_WORLD_SIZE.
    local_world = env_int("LOCAL_WORLD_SIZE", world)
    nodes = max(1, world // max(1, local_world))
    if cuda:
        torch.cuda.set_device(local)
    return rank, world, local, nodes, backend


# ------------------------------------------------------------------------- model
class ToyBlock(nn.Module):
    """A minimal transformer-like block so the auto-wrap policy has a unit to shard.

    Used only for the smoke (--model toy): no HF download, no 7B weights, but it
    exercises the exact FSDP code path (per-block wrap, mixed precision, a step).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + a)
        x = self.norm2(x + self.mlp(x))
        return x


class ToyOutput(NamedTuple):
    """HF-CausalLMOutput-like container for the toy forward.

    Crucially a NamedTuple (tuple subclass), NOT a bare object. FSDP registers its
    pre-backward hooks by walking the forward output with _apply_to_tensors, which
    traverses tuples / dicts / dataclasses but not an opaque custom class. A bare
    object hid loss/logits from that walk, so the root FSDP unit never transitioned to
    FORWARD_BACKWARD and the inner blocks' post-backward reduce asserted "expected
    FORWARD_BACKWARD but current state is IDLE". HF's real ModelOutput is an
    OrderedDict subclass, which is why the 7B path never hit this. .loss/.logits
    attribute access is preserved so one_step() is identical for toy and HF paths.
    """

    logits: torch.Tensor
    loss: torch.Tensor | None = None


class ToyLM(nn.Module):
    """Tiny causal LM: embedding -> N ToyBlocks -> tied-ish LM head."""

    def __init__(self, vocab: int, dim: int, layers: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([ToyBlock(dim) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, input_ids, labels=None):
        x = self.embed(input_ids)
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.norm(x))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
        return ToyOutput(logits=logits, loss=loss)


def build_model_and_wrap_classes(args, rank: int):
    """Return (unwrapped nn.Module, set-of-block-classes-for-auto-wrap, vocab).

    For the real path we load the HF model and pull its decoder-layer class out of
    the module tree so transformer_auto_wrap_policy shards one decoder block at a time
    (the standard, memory-correct FSDP unit for a 7B). For the toy path we use ToyBlock.
    """
    if args.model == "toy":
        vocab = 512
        model = ToyLM(vocab=vocab, dim=args.toy_dim, layers=args.toy_layers)
        return model, {ToyBlock}, vocab

    # Real HF model. Build on meta/CPU; FSDP shards it onto each GPU.
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(args.model)
    dtype = torch.bfloat16
    if args.random_init:
        # Construct from config WITHOUT downloading/loading the 15G checkpoint. Same
        # architecture, shapes, and FSDP cost; only the weights differ. Useful when the
        # scaling number (a systems measurement) does not depend on the trained values.
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, low_cpu_mem_usage=True
        )
    model.config.use_cache = False  # required for training / grad checkpointing

    # Discover the decoder-layer class to use as the FSDP wrap unit.
    block_classes = set()
    for module in model.modules():
        name = module.__class__.__name__
        if name.endswith("DecoderLayer") or name.endswith("Block"):
            block_classes.add(module.__class__)
    if not block_classes and rank == 0:
        print("[fsdp] WARNING: no *DecoderLayer/*Block class found; "
              "falling back to size-based auto-wrap", flush=True)
    vocab = int(getattr(cfg, "vocab_size", 32000))
    return model, block_classes, vocab


def make_auto_wrap_policy(block_classes):
    if block_classes:
        return functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls=block_classes
        )
    # Fallback: wrap any submodule above ~1e6 params.
    return functools.partial(size_based_auto_wrap_policy, min_num_params=int(1e6))


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="FSDP/ZeRO-3 strong-scaling trainer.")
    ap.add_argument("--model", default="Qwen/Qwen2-7B-Instruct",
                    help="HF model id, or 'toy' for the CPU/GPU smoke")
    ap.add_argument("--random-init", action="store_true",
                    help="construct the HF model from config (no checkpoint load); "
                         "the scaling number is a systems measurement, independent of weights")
    ap.add_argument("--steps", type=int, default=20, help="optimizer steps to time")
    ap.add_argument("--warmup", type=int, default=3,
                    help="untimed warmup steps (allocator + first all-gather)")
    ap.add_argument("--batch", type=int, default=1, help="PER-GPU micro-batch size")
    ap.add_argument("--seq-len", type=int, default=1024, dest="seq_len")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--toy-dim", type=int, default=128, dest="toy_dim")
    ap.add_argument("--toy-layers", type=int, default=2, dest="toy_layers")
    ap.add_argument("--grad-checkpointing", action="store_true", dest="grad_ckpt")
    ap.add_argument("--out-dir", default=None, dest="out_dir",
                    help="defaults to <repo>/results")
    a = ap.parse_args(argv)

    rank, world, local, nodes, backend = setup()
    cuda = torch.cuda.is_available()
    dev = torch.device("cuda", local) if cuda else torch.device("cpu")
    is_main = rank == 0

    if is_main:
        print(f"[fsdp] backend={backend} world={world} nodes={nodes} "
              f"local_rank={local} cuda={cuda} model={a.model} "
              f"per_gpu_batch={a.batch} seq_len={a.seq_len} "
              f"random_init={a.random_init}", flush=True)

    model, block_classes, vocab = build_model_and_wrap_classes(a, rank)
    n_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"[fsdp] model params={n_params/1e9:.3f}B  "
              f"wrap_blocks={sorted(c.__name__ for c in block_classes)}", flush=True)

    auto_wrap = make_auto_wrap_policy(block_classes)

    # Activation checkpointing must be enabled on the HF model BEFORE the FSDP wrap so
    # it applies to the inner decoder layers. use_reentrant=False composes with FSDP.
    if a.grad_ckpt and a.model != "toy":
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if is_main:
                print("[fsdp] gradient checkpointing enabled (use_reentrant=False)", flush=True)
        except Exception as e:
            if is_main:
                print(f"[fsdp] grad-checkpointing not enabled: {e}", flush=True)

    # Mixed precision: bf16 params/reduce/buffers. On CPU (smoke) FSDP mixed precision
    # with bf16 reduce is unsupported on gloo, so fall back to fp32 there.
    mp = None
    if cuda:
        mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3 equivalent
        mixed_precision=mp,
        device_id=local if cuda else None,
        use_orig_params=True,  # cleaner optimizer + works with grad checkpointing
        sync_module_states=False,
    )

    if is_main:
        print(f"[fsdp] wrapped OK; sharding=FULL_SHARD mp={'bf16' if mp else 'fp32'}",
              flush=True)

    opt = torch.optim.AdamW(fsdp_model.parameters(), lr=a.lr)

    # Synthetic token batch: same shape every step, seeded per-rank so ranks differ.
    g = torch.Generator(device="cpu").manual_seed(1234 + rank)
    input_ids = torch.randint(0, vocab, (a.batch, a.seq_len), generator=g).to(dev)
    labels = input_ids.clone()

    tokens_per_step = a.batch * a.seq_len * world  # global tokens per optimizer step

    def one_step():
        opt.zero_grad(set_to_none=True)
        out = fsdp_model(input_ids=input_ids, labels=labels)
        loss = out.loss
        loss.backward()
        opt.step()
        return float(loss.detach().float().item())

    # Warmup (untimed): first step pays the allocator + initial all-gather cost.
    for _ in range(max(0, a.warmup)):
        one_step()
    if cuda:
        torch.cuda.synchronize()
    dist.barrier()

    # Timed loop.
    step_times = []
    t_all0 = time.perf_counter()
    last_loss = float("nan")
    for s in range(a.steps):
        t0 = time.perf_counter()
        last_loss = one_step()
        if cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        step_times.append(dt)
        if is_main and (s % max(1, a.steps // 5) == 0 or s == a.steps - 1):
            print(f"[fsdp] step {s:3d}  loss {last_loss:.4f}  "
                  f"{dt*1e3:.1f} ms  {tokens_per_step/dt:,.0f} tok/s", flush=True)
    dist.barrier()
    total_time = time.perf_counter() - t_all0

    # Reduce per-step times to robust stats on rank 0.
    st = torch.tensor(step_times, dtype=torch.float64)
    median_step = float(st.median())
    mean_step = float(st.mean())
    p90_step = float(st.kthvalue(max(1, int(0.9 * len(st))))[0]) if len(st) > 1 else median_step
    # Throughput from the median step is the honest strong-scaling metric (robust to
    # a slow tail). Global tokens/sec = global tokens per step / median step time.
    tokens_per_sec = tokens_per_step / median_step

    peak_mem_gb = (torch.cuda.max_memory_allocated(dev) / 1e9) if cuda else 0.0

    if is_main:
        out_dir = a.out_dir
        if out_dir is None:
            here = os.path.dirname(os.path.abspath(__file__))
            out_dir = os.path.join(os.path.dirname(here), "results")
        os.makedirs(out_dir, exist_ok=True)
        rec = {
            "world_size": world,
            "nodes": nodes,
            "gpus_per_node": world // max(1, nodes),
            "backend": backend,
            "model": a.model,
            "random_init": bool(a.random_init),
            "n_params_billion": round(n_params / 1e9, 4),
            "per_gpu_batch": a.batch,
            "seq_len": a.seq_len,
            "global_batch": a.batch * world,
            "tokens_per_step_global": tokens_per_step,
            "steps_timed": a.steps,
            "warmup": a.warmup,
            "median_step_s": median_step,
            "mean_step_s": mean_step,
            "p90_step_s": p90_step,
            "total_time_s": total_time,
            "tokens_per_sec_global": tokens_per_sec,
            "tokens_per_sec_per_gpu": tokens_per_sec / world,
            "peak_mem_gb_rank0": round(peak_mem_gb, 3),
            "sharding_strategy": "FULL_SHARD",
            "mixed_precision": "bf16" if cuda else "fp32",
            "grad_checkpointing": bool(a.grad_ckpt),
            "final_loss": last_loss,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        }
        path = os.path.join(out_dir, f"fsdp_scaling_{world}.json")
        with open(path, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"[fsdp] world={world} nodes={nodes} "
              f"median_step={median_step*1e3:.1f}ms "
              f"tok/s(global)={tokens_per_sec:,.0f} "
              f"peak_mem={peak_mem_gb:.1f}GB -> {path}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
