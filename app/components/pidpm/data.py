"""Synthetic trajectory data for Pi-DPM.

Normal tracks are smooth, speed-bounded constant-curvature arcs (the kind the
S-KBM envelope admits). Three anomaly families mirror the paper's case studies:

    teleport       one sample jumps far off the path (GPS spoofing).
    excess_speed   an abrupt reversal at implausible speed (the Cathay Phoenix
                   circular-spoofing pattern in Sec. 1).
    freeze         the track stalls then resumes (signal-denial gap).

Everything is generated on the fly, so train.py / eval.py run anywhere with no
downloads. To train on real AIS instead, replace TrajectoryDataset with a loader
over MarineCadastre / Danish AIS CSVs that yields (T, in_dim) lon-lat windows
normalised the same way (see normalise()); the rest of the pipeline is unchanged.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import PiDPMConfig


def _normal_arc(rng: np.random.Generator, T: int, v_scale: float) -> np.ndarray:
    omega = rng.uniform(-0.12, 0.12)
    v = rng.uniform(0.4, 1.0) * v_scale
    th0 = rng.uniform(-np.pi, np.pi)
    t = np.arange(T, dtype=np.float64)
    heading = th0 + omega * t
    step = v
    xy = np.cumsum(np.stack([step * np.cos(heading), step * np.sin(heading)], axis=1), axis=0)
    xy += rng.normal(0, 0.02 * v_scale, xy.shape)
    return xy - xy[0]


def _inject(rng: np.random.Generator, xy: np.ndarray, kind: str) -> np.ndarray:
    xy = xy.copy()
    T = xy.shape[0]
    if kind == "teleport":
        j = rng.integers(T // 4, 3 * T // 4)
        xy[j] += rng.uniform(6, 12) * rng.choice([-1.0, 1.0], size=2)
    elif kind == "excess_speed":
        # sustained implausible speed (the Cathay Phoenix pattern): each step
        # advances by d metres, so speed ~ d*sqrt(2) must exceed the v_max
        # envelope (12.86 m/s) to be a genuine kinematic violation
        h = T // 2
        d = rng.uniform(20.0, 35.0)
        xy[h:] = xy[h - 1] + np.outer(np.arange(1, T - h + 1), rng.choice([-1.0, 1.0], size=2) * d)
    elif kind == "freeze":
        a, b = sorted(rng.choice(np.arange(1, T), size=2, replace=False))
        xy[a:b] = xy[a]
    return xy


class TrajectoryDataset(Dataset):
    """Generates normal (label 0) and anomalous (label 1) tracks.

    anomaly_ratio controls the share of anomalies; for training set it to 0 so
    the diffusion learns the normal manifold, and use a labelled split (ratio>0)
    only for evaluation.
    """

    KINDS = ("teleport", "excess_speed", "freeze")

    def __init__(self, cfg: PiDPMConfig, n: int = 4096, anomaly_ratio: float = 0.0, seed: int = 0) -> None:
        self.cfg = cfg
        self.n = n
        rng = np.random.default_rng(seed)
        v_scale = cfg.v_max_mps * 0.6
        xs, ys = [], []
        for _ in range(n):
            base = _normal_arc(rng, cfg.seq_len, v_scale)
            if rng.random() < anomaly_ratio:
                xy = _inject(rng, base, rng.choice(self.KINDS))
                ys.append(1)
            else:
                xy = base
                ys.append(0)
            xs.append(xy[:, : cfg.in_dim])
        arr = np.stack(xs).astype(np.float32)
        # model space = (physical metres - mean) / pos_scale, a single isotropic
        # scale so velocities/curvatures stay consistent for the physics term
        self.mean = arr.reshape(-1, cfg.in_dim).mean(0)
        self.pos_scale = cfg.pos_scale
        self.x = ((arr - self.mean) / self.pos_scale).astype(np.float32)
        self.y = np.asarray(ys, dtype=np.int64)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.x[i]), torch.tensor(self.y[i])


def normalise(xy: np.ndarray, mean: np.ndarray, pos_scale: float) -> np.ndarray:
    """Map physical-metre positions into model space (mean-centred, /pos_scale)."""
    return (xy - mean) / pos_scale
