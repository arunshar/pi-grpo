"""Generate the demo dataset files shipped with the Pi-GRPO Hugging Face Space.

Three files:

- trajectories.jsonl  — 30 trajectories across 5 classes, each scored by the
  PhysicsReward (R_total / R_hard / R_soft / R_data) so the Streamlit app can
  display a reward distribution.
- training_curve.jsonl — 300 step records for each of three trainers
  (PPO, DPO with gamma_phys, GRPO) on a synthetic but realistic reward path:
  warm-up, exponential rise to ~0, noisy plateau.
- preferences.jsonl   — 50 (prompt, chosen, rejected) DPO triples derived from
  pairing high-reward trajectories against violation-heavy ones.

The numbers are deterministic (seeded). Re-run this script to regenerate the
files; they're checked in so the Space loads them directly without needing
any extra deps at startup.

Run:
    python spaces/hf-demo/data/_generate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent.parent.parent
sys.path.insert(0, str(_PROJECT))

from app.components.kinematic_bicycle import SkbmConfig, evaluate  # noqa: E402
from app.components.physics_reward import (  # noqa: E402
    EmpiricalEnvelope,
    PhysicsReward,
    RewardWeights,
)


_SEED = 42
_CFG = SkbmConfig()
_REWARD = PhysicsReward(_CFG, weights=RewardWeights(), envelope=EmpiricalEnvelope())


# ----------------------------------------------------------------------- traj


def _clean(n: int, v: float) -> np.ndarray:
    s = np.zeros((n, 4))
    s[:, 0] = np.arange(n) * v
    s[:, 3] = v
    return s


def _speeding(n: int, v: float) -> np.ndarray:
    s = np.zeros((n, 4))
    s[:, 0] = np.arange(n) * v
    s[:, 3] = v
    return s


def _ramping(n: int, v0: float, v1: float) -> np.ndarray:
    v = np.linspace(v0, v1, n)
    s = np.zeros((n, 4))
    s[1:, 0] = np.cumsum(v[:-1])
    s[:, 3] = v
    return s


def _jerk_spike(n: int, v_base: float, jerk_at: int) -> np.ndarray:
    v = np.full(n, v_base)
    width = max(1, n // 8)
    for i, k in enumerate(range(jerk_at, min(jerk_at + width, n))):
        v[k] = v_base + 8.0 * (1 + i / width)
    s = np.zeros((n, 4))
    s[1:, 0] = np.cumsum(v[:-1])
    s[:, 3] = v
    return s


def _curving(n: int, v: float, total_turn_rad: float) -> np.ndarray:
    theta = np.linspace(0, total_turn_rad, n)
    s = np.zeros((n, 4))
    for i in range(1, n):
        s[i, 0] = s[i - 1, 0] + v * np.cos(theta[i - 1])
        s[i, 1] = s[i - 1, 1] + v * np.sin(theta[i - 1])
    s[:, 2] = theta
    s[:, 3] = v
    return s


def gen_trajectories() -> list[dict]:
    rng = np.random.default_rng(_SEED)
    items: list[dict] = []
    idx = 0

    # 8 clean
    for _ in range(8):
        v = float(4 + 4 * rng.random())
        n = int(rng.integers(15, 30))
        states = _clean(n, v)
        idx += 1
        items.append(_make(idx, "clean", f"steady {v:.1f} m/s, {n} steps", states))

    # 6 speeding
    for _ in range(6):
        v = float(15 + 25 * rng.random())  # 15..40 m/s, all over v_max=12.86
        n = int(rng.integers(15, 30))
        states = _speeding(n, v)
        idx += 1
        items.append(_make(idx, "speeding", f"sustained {v:.1f} m/s vs v_max=12.86", states))

    # 5 ramping (some cross v_max, some don't)
    for _ in range(5):
        v0 = float(2 + 4 * rng.random())
        v1 = float(8 + 25 * rng.random())
        n = int(rng.integers(20, 30))
        states = _ramping(n, v0, v1)
        idx += 1
        items.append(_make(idx, "ramping", f"{v0:.1f} -> {v1:.1f} m/s over {n} steps", states))

    # 5 jerk-spike
    for _ in range(5):
        v_base = float(4 + 4 * rng.random())
        n = int(rng.integers(20, 30))
        spike_at = int(rng.integers(n // 4, 3 * n // 4))
        states = _jerk_spike(n, v_base, spike_at)
        idx += 1
        items.append(_make(idx, "jerk-spike", f"base {v_base:.1f} m/s with sharp spike at step {spike_at}", states))

    # 6 curving
    for _ in range(6):
        v = float(3 + 4 * rng.random())
        n = int(rng.integers(20, 30))
        turn = float(rng.uniform(np.pi / 8, np.pi))
        states = _curving(n, v, turn)
        idx += 1
        items.append(_make(idx, "curving", f"v={v:.1f} m/s, turn {np.degrees(turn):.0f} deg", states))

    return items


def _make(idx: int, label: str, description: str, states: np.ndarray) -> dict:
    rb = _REWARD.score(states)
    v = evaluate(states, cfg=_CFG)
    return {
        "id": f"t-{idx:03d}",
        "label": label,
        "description": description,
        "n_steps": int(states.shape[0]),
        "reward": {
            "total": float(rb.total),
            "hard": float(rb.hard),
            "soft": float(rb.soft),
            "data": float(rb.data),
            "pref": float(rb.pref),
        },
        "violations": {
            "speed_max_pct": float(v.speed_max_pct),
            "accel_max_pct": float(v.accel_max_pct),
            "steer_max_pct": float(v.steer_max_pct),
            "curvature_p95": float(v.curvature_p95),
            "jerk_p95": float(v.jerk_p95),
        },
        "states": states.tolist(),
    }


# -------------------------------------------------------------- training curve


def _curve(algo: str, n: int = 300, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    steps = np.arange(n)

    if algo == "PPO":
        # PPO with adaptive KL controller; reward rises smoothly, KL hovers near target
        reward = -2.0 + 2.5 * (1 - np.exp(-steps / 80)) + rng.normal(0, 0.18, n)
        kl = 4.0 + rng.normal(0, 0.6, n)
        loss = 0.55 * np.exp(-steps / 100) + 0.05 + rng.normal(0, 0.02, n)
        kl_coef = 0.2 + 0.05 * np.sin(steps / 30)
        violation_rate = np.clip(0.18 * np.exp(-steps / 60) + rng.normal(0, 0.005, n), 0, None)
    elif algo == "DPO+gamma_phys":
        # DPO + small physics-aware gamma; converges fastest, low violation rate
        reward = -1.0 + 1.5 * (1 - np.exp(-steps / 50)) + rng.normal(0, 0.10, n)
        kl = 1.5 + rng.normal(0, 0.3, n)  # KL is implicit; tracks the implicit reward beta
        loss = 0.45 * np.exp(-steps / 60) + 0.04 + rng.normal(0, 0.015, n)
        kl_coef = np.full(n, 0.05)  # gamma_phys, not adaptive
        violation_rate = np.clip(0.05 * np.exp(-steps / 40) + rng.normal(0, 0.002, n), 0, None)
    elif algo == "GRPO":
        # GRPO group-baseline; medium pace, no value head, slightly noisier
        reward = -2.5 + 3.0 * (1 - np.exp(-steps / 70)) + rng.normal(0, 0.20, n)
        kl = 4.0 + rng.normal(0, 0.7, n)
        loss = 0.60 * np.exp(-steps / 90) + 0.06 + rng.normal(0, 0.025, n)
        kl_coef = 0.18 + 0.04 * np.cos(steps / 40)
        violation_rate = np.clip(0.12 * np.exp(-steps / 80) + rng.normal(0, 0.004, n), 0, None)
    else:
        raise ValueError(algo)

    return [
        {
            "algo": algo,
            "step": int(steps[i]),
            "reward_mean": float(reward[i]),
            "kl": float(kl[i]),
            "loss": float(loss[i]),
            "kl_coef": float(kl_coef[i]),
            "violation_rate": float(violation_rate[i]),
        }
        for i in range(n)
    ]


def gen_training_curves() -> list[dict]:
    return _curve("PPO", seed=1) + _curve("DPO+gamma_phys", seed=2) + _curve("GRPO", seed=3)


# ----------------------------------------------------------------- preferences


def gen_preferences(trajectories: list[dict]) -> list[dict]:
    """Pair clean trajectories (chosen) against violation-heavy ones (rejected)."""

    rng = np.random.default_rng(_SEED + 1)
    chosen_pool = [t for t in trajectories if t["reward"]["hard"] >= 0]
    rejected_pool = [t for t in trajectories if t["reward"]["hard"] < 0]
    pairs: list[dict] = []
    for i in range(50):
        c = chosen_pool[int(rng.integers(0, len(chosen_pool)))]
        r = rejected_pool[int(rng.integers(0, len(rejected_pool)))]
        margin = c["reward"]["total"] - r["reward"]["total"]
        pairs.append({
            "id": f"p-{i:03d}",
            "prompt": (
                "Rate the physical plausibility of this vessel trajectory and "
                "flag any sustained-speeding or hard-jerk violations. Choose the "
                "trajectory that better satisfies the single-axle kinematic-bicycle "
                "envelope (v_max=12.86 m/s, a_max=0.5 m/s^2)."
            ),
            "chosen_id": c["id"],
            "chosen_label": c["label"],
            "chosen_reward": c["reward"]["total"],
            "rejected_id": r["id"],
            "rejected_label": r["label"],
            "rejected_reward": r["reward"]["total"],
            "margin": float(margin),
            "source": "synthetic",
        })
    return pairs


# ----------------------------------------------------------------------- main


def main() -> None:
    out_dir = _HERE
    trajectories = gen_trajectories()
    curves = gen_training_curves()
    preferences = gen_preferences(trajectories)

    (out_dir / "trajectories.jsonl").write_text(
        "\n".join(json.dumps(t, separators=(",", ":")) for t in trajectories) + "\n"
    )
    (out_dir / "training_curve.jsonl").write_text(
        "\n".join(json.dumps(c, separators=(",", ":")) for c in curves) + "\n"
    )
    (out_dir / "preferences.jsonl").write_text(
        "\n".join(json.dumps(p, separators=(",", ":")) for p in preferences) + "\n"
    )

    print(f"trajectories      : {len(trajectories)}  -> {out_dir/'trajectories.jsonl'}")
    print(f"training_curve    : {len(curves)}        -> {out_dir/'training_curve.jsonl'}")
    print(f"preferences       : {len(preferences)}   -> {out_dir/'preferences.jsonl'}")


if __name__ == "__main__":
    main()
