"""Train Pi-DPM on the normal-trajectory manifold.

CPU-runnable for a smoke test; uses CUDA + AMP automatically when available and
scales to multi-GPU under `accelerate launch` (the optimizer/step are framework
agnostic). Saves a checkpoint consumable by PiDPM.from_checkpoint and by the
pi-grpo PiDpmScorer reward bridge.

    python -m app.components.pidpm.train --epochs 5 --n 2048 --out /tmp/pidpm.pt
"""

from __future__ import annotations

import argparse
import copy

import torch
from torch.utils.data import DataLoader

from .config import PiDPMConfig
from .data import TrajectoryDataset
from .scoring import PiDPM


def train(cfg: PiDPMConfig, n: int = 4096, out: str | None = None, device: str | None = None) -> PiDPM:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg.seed)

    ds = TrajectoryDataset(cfg, n=n, anomaly_ratio=0.0, seed=cfg.seed)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=0)

    model = PiDPM(cfg).to(dev)
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    use_amp = cfg.amp and dev.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.train()
    for epoch in range(cfg.epochs):
        running = {"loss": 0.0, "eps": 0.0, "phys": 0.0}
        for xb, _ in dl:
            xb = xb.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                out_d = model.diffusion.loss(xb)
            scaler.scale(out_d["loss"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters(), strict=False):
                    pe.mul_(cfg.ema_decay).add_(pm, alpha=1 - cfg.ema_decay)
            for k in running:
                running[k] += float(out_d[k])
        nb = len(dl)
        print(f"epoch {epoch + 1:3d}/{cfg.epochs}  "
              f"loss={running['loss'] / nb:.4f}  eps={running['eps'] / nb:.4f}  phys={running['phys'] / nb:.4f}")

    ema.cfg = cfg
    if out:
        ema.save(out)
        print(f"saved checkpoint -> {out}")
    return ema


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    a = ap.parse_args()
    base = PiDPMConfig()
    cfg = PiDPMConfig(
        epochs=a.epochs or base.epochs,
        batch_size=a.batch_size or base.batch_size,
        seq_len=a.seq_len or base.seq_len,
    )
    train(cfg, n=a.n, out=a.out, device=a.device)


if __name__ == "__main__":
    main()
