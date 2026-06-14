"""Evaluate Pi-DPM anomaly detection on a labelled synthetic split.

Reports REAL numbers measured on the run (AUROC, average precision, and
precision/recall at a lambda set from the normal-set 95th percentile). These are
honest synthetic-benchmark figures, not leaderboard claims.

    python -m app.components.pidpm.eval --ckpt /tmp/pidpm.pt --n 1024
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from .config import PiDPMConfig
from .data import TrajectoryDataset
from .scoring import PiDPM


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney U); no sklearn dependency."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    n_pos = int(labels.sum())
    return float((precision * y).sum() / n_pos) if n_pos else float("nan")


def evaluate(model: PiDPM, cfg: PiDPMConfig, n: int = 1024, device: str | None = None) -> dict[str, float]:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev).eval()

    # normal-only split to calibrate lambda, then a mixed labelled split
    cal = TrajectoryDataset(cfg, n=n // 2, anomaly_ratio=0.0, seed=cfg.seed + 1)
    test = TrajectoryDataset(cfg, n=n, anomaly_ratio=0.5, seed=cfg.seed + 2)

    def scores_for(ds: TrajectoryDataset) -> np.ndarray:
        out = []
        for i in range(0, len(ds), cfg.batch_size):
            xb = torch.from_numpy(ds.x[i : i + cfg.batch_size]).to(dev)
            out.append(model.score(xb).score)
        return np.concatenate(out)

    cal_scores = scores_for(cal)
    lam = float(np.quantile(cal_scores, 0.95))
    test_scores = scores_for(test)
    labels = test.y

    auroc = _auroc(labels, test_scores)
    ap = _average_precision(labels, test_scores)
    pred = (test_scores >= lam).astype(np.int64)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "auroc": auroc, "average_precision": ap, "lambda": lam,
        "precision@lambda": precision, "recall@lambda": recall, "f1@lambda": f1,
        "n_test": float(len(labels)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None, help="checkpoint; omit to score with an untrained model")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--device", type=str, default=None)
    a = ap.parse_args()
    if a.ckpt:
        model = PiDPM.from_checkpoint(a.ckpt)
        cfg = model.cfg
    else:
        cfg = PiDPMConfig()
        model = PiDPM(cfg)
    metrics = evaluate(model, cfg, n=a.n, device=a.device)
    print("Pi-DPM synthetic anomaly-detection metrics (measured this run):")
    for k, v in metrics.items():
        print(f"  {k:20s} {v:.4f}")


if __name__ == "__main__":
    main()
