"""Bridge to a Pi-DPM checkpoint for reconstruction-error scoring.

Wraps a torchscripted Pi-DPM module that returns the negative log-
likelihood of a trajectory under the trained physics-informed
diffusion. The scorer is treated as a frozen black box so we can
swap in newer Pi-DPM checkpoints without touching the trainer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import structlog
import torch

log = structlog.get_logger(__name__)


class PiDpmScorer:
    def __init__(self, checkpoint_path: str | None = None, device: str = "cuda") -> None:
        self.device = device
        self.module: torch.jit.ScriptModule | None = None
        if checkpoint_path and Path(checkpoint_path).exists():
            try:
                self.module = torch.jit.load(checkpoint_path, map_location=device)
                self.module.eval()
            except Exception as exc:  # pragma: no cover
                log.warning("pidpm_load_failed", err=str(exc))

    def log_prob(self, trajectory: np.ndarray) -> float:
        """Returns log p(trajectory). Negative reconstruction errors map to higher rewards."""

        if self.module is None:
            # offline / test stub: use a smooth proxy of -|jerk| as a stand-in
            if trajectory.shape[0] < 3:
                return 0.0
            if trajectory.shape[1] >= 4:
                v = trajectory[:, 3]
            else:
                v = np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1)
            jerk = np.diff(np.diff(v))
            return float(-np.mean(np.abs(jerk)))
        with torch.no_grad():
            t = torch.from_numpy(trajectory).float().to(self.device).unsqueeze(0)
            return float(self.module(t).item())
