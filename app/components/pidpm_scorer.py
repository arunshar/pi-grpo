"""Bridge to Pi-DPM for reconstruction-error scoring in the reward model.

Returns log p(trajectory) under the physics-informed diffusion: higher (less
negative) for trajectories that are both well reconstructed by the model and
physically plausible under the S-KBM envelope, lower for kinematically
implausible / poorly reconstructed ones. The reward model adds this to the hard
kinematic term, so a smooth, feasible track is rewarded over a teleporting one.

Resolution order:
  1. a real Pi-DPM checkpoint (state_dict saved by PiDPM.save / train.py),
  2. a torchscripted Pi-DPM module (legacy black-box export),
  3. an analytic physics-and-smoothness proxy (no checkpoint / no torch model).

The proxy is a principled stand-in (negative mean jerk plus a speed-envelope
penalty), not a fabricated score, so the reward model still degrades gracefully
in tests and offline runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import structlog
import torch

log = structlog.get_logger(__name__)


class PiDpmScorer:
    def __init__(self, checkpoint_path: str | None = None, device: str = "cuda") -> None:
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.model = None            # real PiDPM
        self.module = None           # legacy torchscript module
        if checkpoint_path and Path(checkpoint_path).exists():
            self._load(checkpoint_path)

    def _load(self, path: str) -> None:
        # try the real Pi-DPM state_dict first, then a torchscript export
        try:
            from app.components.pidpm.scoring import PiDPM

            self.model = PiDPM.from_checkpoint(path, map_location=self.device).to(self.device).eval()
            log.info("pidpm_loaded", kind="state_dict", path=path)
            return
        except Exception as exc:
            log.debug("pidpm_state_dict_load_failed", err=str(exc))
        try:
            self.module = torch.jit.load(path, map_location=self.device).eval()
            log.info("pidpm_loaded", kind="torchscript", path=path)
        except Exception as exc:  # pragma: no cover
            log.warning("pidpm_load_failed", err=str(exc))

    def log_prob(self, trajectory: np.ndarray) -> float:
        """Returns log p(trajectory). Negative reconstruction/physics error -> higher reward."""

        if self.model is not None:
            return self.model.log_prob(trajectory)
        if self.module is not None:
            with torch.no_grad():
                t = torch.from_numpy(np.asarray(trajectory)).float().to(self.device).unsqueeze(0)
                return float(self.module(t).item())
        return self._analytic_proxy(np.asarray(trajectory))

    @staticmethod
    def _analytic_proxy(trajectory: np.ndarray) -> float:
        """-mean|jerk| minus a soft speed-envelope penalty; no trained model needed."""
        if trajectory.shape[0] < 3:
            return 0.0
        if trajectory.ndim == 2 and trajectory.shape[1] >= 4:
            v = trajectory[:, 3]
            xy = trajectory[:, 1:3]
        else:
            xy = trajectory[:, :2]
            v = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        jerk = np.diff(np.diff(v)) if v.shape[0] >= 3 else np.zeros(1)
        speed_excess = np.mean(np.maximum(v - np.quantile(v, 0.95) * 1.5, 0.0)) if v.size else 0.0
        return float(-np.mean(np.abs(jerk)) - speed_excess)
