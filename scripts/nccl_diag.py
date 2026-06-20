"""Minimal 2-node NCCL all-reduce diagnostic.

Isolates an inter-node NCCL hang from any model/FSDP code: it does nothing but
init the process group and all-reduce a 4 MB tensor a few times. Run under
torchrun with NCCL_DEBUG=INFO to see transport selection (IB vs SOCKET) and, with
TORCH_NCCL_BLOCKING_WAIT=1 + a short init timeout, fail fast with a traceback
instead of hanging to the wall-clock limit.

Outcomes:
- prints "[diag] DONE allreduce correct=True"  -> the IB fabric + NCCL config are
  fine, so a hang in the real run is in the model/FSDP path, not the fabric.
- hangs or raises in init_process_group / all_reduce -> the fabric/NCCL transport
  is the culprit; the NCCL_DEBUG=INFO lines above the failure name the cause.
"""

from __future__ import annotations

import datetime
import os
import time

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    host = os.uname().nodename

    # Short timeout so a stalled rendezvous/collective aborts with a traceback
    # under TORCH_NCCL_BLOCKING_WAIT rather than hanging to the Slurm time limit.
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=90))
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.cuda.current_device()
    print(
        f"[diag] rank={rank}/{world} local_rank={local_rank} host={host} "
        f"dev={dev} {torch.cuda.get_device_name(dev)}",
        flush=True,
    )

    x = torch.ones(1 << 20, device=dev) * (rank + 1)  # 4 MB per rank
    for i in range(5):
        t0 = time.time()
        dist.all_reduce(x)
        torch.cuda.synchronize()
        if rank == 0:
            print(
                f"[diag] iter={i} all_reduce ok x0={x[0].item():.1f} "
                f"ms={(time.time() - t0) * 1e3:.2f}",
                flush=True,
            )
        x = torch.ones(1 << 20, device=dev) * (rank + 1)

    # One final clean reduce to verify correctness end to end.
    dist.all_reduce(x)
    torch.cuda.synchronize()
    if rank == 0:
        expected = world * (world + 1) / 2.0
        print(
            f"[diag] DONE all_reduce correct={abs(x[0].item() - expected) < 1e-3} "
            f"got={x[0].item():.1f} expected={expected:.1f}",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
