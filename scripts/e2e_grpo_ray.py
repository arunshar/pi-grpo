"""End-to-end Ray-GRPO smoke driver (CPU, self-contained).

This is the Python half of ``scripts/e2e_grpo_ray.sbatch``. It runs three checks
against the REAL, already-landed code in this repo and writes a single PASS/FAIL
verdict so a Slurm dependency (or a human) can gate on it without a login-node
poller:

  (1) reward-actor sweep   W = 1, 2, 4 driving the real
      ``app.policy.ray_driver.train_grpo_ray`` once per W with a
      ``RayRewardPool(n_actors=W)`` (serial fallback when Ray is absent or
      W == 1). For each W we record reward-eval throughput (trajectories / sec)
      and assert it is RECORDED (a finite, positive number). Whether throughput
      INCREASES with W is reported honestly: on a tiny synthetic task, or with
      no real Ray cluster, the curve is expected to be flat or even to regress
      from actor / serialization overhead, and we say so rather than asserting a
      speedup that the hardware did not produce.

  (2) short training run    a single ``train_grpo_ray`` and the assertion
      ``reward_end >= reward_start`` (GRPO must not move mean reward DOWN on the
      tiny task; this mirrors tests/test_ray_driver.py's convergence-parity gate).

  (3) step-decomposition CSV emitted for the whole sweep (one row per (W, step))
      plus the per-W scaling summary CSV.

WHY a separate driver rather than just calling ``scripts/grpo_scale.py``: the
sweep entrypoint in ``grpo_scale.run_sweep`` is written against a FUTURE sibling
contract (``RayTrainConfig`` + ``train_grpo_ray(cfg, num_reward_actors=...)``)
that the current ``app/policy/ray_driver.py`` does not yet expose (its real
signature is ``train_grpo_ray(cfg: TrainConfig, reward_pool=None, ...)`` and it
emits ``t_sample`` / ``t_score`` / ``t_learn`` per step, not ``sample_s`` /
``reward_s`` / ``learn_s``). Rather than modify the repo (forbidden) or wait on
the sibling task, this driver calls the real driver directly and REUSES the pure
CSV / throughput / speedup helpers from ``grpo_scale.py`` after a small,
explicit key remap. When the sibling contract lands, ``grpo_scale.main`` becomes
the canonical path and this driver can shrink to a thin wrapper around it.

Pure, importable, no-side-effect helpers (``remap_step_record``,
``classify_scaling``, ``parse_actor_counts_arg``, ``verdict_line``) are unit
tested in ``tests/test_e2e_grpo_ray.py`` with NO torch / Ray / GPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- repo wiring

# This file lives at <repo>/scripts/e2e_grpo_ray.py. The sbatch also sets
# PYTHONPATH to the repo root, but we add it here too so the driver is runnable
# standalone (``python3 scripts/e2e_grpo_ray.py``) without env priming.
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_grpo_scale():
    """Path-import ``scripts/grpo_scale.py`` (the scripts dir is not a package).

    We register the module in ``sys.modules`` BEFORE ``exec_module`` because
    grpo_scale uses a frozen dataclass with a ``default_factory`` under
    ``from __future__ import annotations``; the forward-ref resolution in
    ``dataclasses._process_class`` looks the module up by name and would hit
    ``None`` otherwise (same pattern as tests/test_grpo_scale.py).
    """
    path = _THIS.parent / "grpo_scale.py"
    spec = importlib.util.spec_from_file_location("grpo_scale", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load grpo_scale from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- pure helpers


# The real ray_driver emits these per-step timing keys; grpo_scale's CSV helpers
# expect the *_s names. This is the single, explicit bridge between them.
_TIMING_REMAP: dict[str, str] = {
    "t_sample": "sample_s",
    "t_score": "reward_s",
    "t_learn": "learn_s",
}


def remap_step_record(metrics: dict[str, float], n_traj: float) -> dict[str, float]:
    """Translate one ``train_grpo_ray`` per-step metrics dict into the shape
    ``grpo_scale.extract_step_records`` produces.

    Maps ``t_sample/t_score/t_learn`` -> ``sample_s/reward_s/learn_s``, carries
    ``reward_mean/loss/kl`` through when present, and stamps ``n_traj`` (the
    B*K trajectories scored that step). Missing keys become ``nan`` so a partial
    metrics dict still yields a valid, total record rather than a KeyError.
    """
    out: dict[str, float] = {
        "sample_s": float("nan"),
        "reward_s": float("nan"),
        "learn_s": float("nan"),
        "n_traj": float(n_traj),
        "reward_mean": float(metrics.get("reward_mean", float("nan"))),
        "loss": float(metrics.get("loss", float("nan"))),
        "kl": float(metrics.get("kl", float("nan"))),
    }
    for src, dst in _TIMING_REMAP.items():
        if src in metrics and metrics[src] is not None:
            out[dst] = float(metrics[src])
    return out


def classify_scaling(scaling_rows: Sequence[dict[str, Any]]) -> str:
    """Describe the throughput-vs-W curve honestly.

    Returns one of ``"increasing"``, ``"flat"``, ``"regressing"``, or
    ``"unknown"`` based on the largest-W throughput relative to the W=1 baseline.
    A >10% gain is ``increasing``; a >10% loss is ``regressing``; within +/-10%
    is ``flat``; a missing / non-finite baseline or single point is ``unknown``.
    This is a REPORT, never an assertion: on a tiny CPU task we fully expect
    ``flat`` or ``regressing`` from actor overhead, and that is not a failure.
    """
    by_w: dict[int, float] = {}
    for r in scaling_rows:
        thr = r.get("throughput_traj_per_s")
        if thr is not None and not math.isnan(thr):
            by_w[int(r["num_reward_actors"])] = float(thr)
    if 1 not in by_w or len(by_w) < 2:
        return "unknown"
    base = by_w[1]
    if base <= 0.0:
        return "unknown"
    top_w = max(by_w)
    ratio = by_w[top_w] / base
    if ratio > 1.10:
        return "increasing"
    if ratio < 0.90:
        return "regressing"
    return "flat"


def parse_actor_counts_arg(spec: str) -> tuple[int, ...]:
    """Parse a ``"1,2,4"`` actor-count spec into a tuple of positive ints.

    Thin, dependency-free mirror of ``grpo_scale.parse_actor_counts`` so the e2e
    arg parsing is unit-testable without importing the (torch-touching) sweep
    module. Rejects empty input and non-positive entries.
    """
    counts: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        w = int(part)
        if w <= 0:
            raise ValueError(f"reward-actor count must be positive; got {w}")
        counts.append(w)
    if not counts:
        raise ValueError(f"no reward-actor counts parsed from {spec!r}")
    return tuple(counts)


def verdict_line(*, throughput_ok: bool, reward_ok: bool, csv_ok: bool) -> str:
    """Compose the single PASS/FAIL headline from the three check booleans.

    PASS iff all three checks held. The string is stable and greppable so a
    downstream dependency can ``grep -q '^PASS' e2e_summary.txt``.
    """
    status = "PASS" if (throughput_ok and reward_ok and csv_ok) else "FAIL"
    return (
        f"{status} throughput_recorded={throughput_ok} "
        f"reward_nondecreasing={reward_ok} csv_emitted={csv_ok}"
    )


# --------------------------------------------------------------------------- live run


@dataclass
class E2EConfig:
    actor_counts: tuple[int, ...] = (1, 2, 4)
    steps: int = 8
    batch_prompts: int = 4
    group_size: int = 6
    horizon: int = 10
    seed: int = 0
    out_dir: Path = Path("e2e_out")


def _build_reward_pool(num_reward_actors: int):
    """Build a ``RayRewardPool(n_actors=W)`` if importable, else the serial stub.

    ``RayRewardPool`` itself degrades to the serial path when ``n_actors <= 1``
    or Ray is unavailable, so this is safe on a CPU-only node; W just stops
    changing the wall time in that case (which ``classify_scaling`` will report
    as ``flat``).
    """
    from app.policy.ray_driver import default_reward_pool

    return default_reward_pool(num_reward_actors)


def run_sweep_real(cfg: E2EConfig, gs) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drive the REAL train_grpo_ray once per W; return (scaling_rows, decomp_rows).

    Uses grpo_scale's pure builders (``build_scaling_row``, ``build_decomp_rows``,
    ``attach_speedups``) after remapping each step's timing keys.
    """
    from app.policy.driver import TrainConfig
    from app.policy.ray_driver import train_grpo_ray

    train_cfg = TrainConfig(
        steps=cfg.steps,
        batch_prompts=cfg.batch_prompts,
        group_size=cfg.group_size,
        horizon=cfg.horizon,
        seed=cfg.seed,
    )
    n_per_step = cfg.batch_prompts * cfg.group_size

    # A SweepConfig only to feed build_scaling_row's run-shape fields.
    sweep_cfg = gs.SweepConfig(
        actor_counts=cfg.actor_counts,
        steps=cfg.steps,
        batch_prompts=cfg.batch_prompts,
        group_size=cfg.group_size,
        horizon=cfg.horizon,
        seed=cfg.seed,
        out_dir=cfg.out_dir,
    )

    scaling_rows: list[dict[str, Any]] = []
    decomp_rows: list[dict[str, Any]] = []
    for w in cfg.actor_counts:
        pool = _build_reward_pool(w)
        t0 = time.perf_counter()
        result = train_grpo_ray(train_cfg, reward_pool=pool)
        wall = time.perf_counter() - t0
        records = [remap_step_record(m, n_per_step) for m in result.history]
        scaling_rows.append(
            gs.build_scaling_row(num_reward_actors=w, cfg=sweep_cfg, records=records)
        )
        decomp_rows.extend(gs.build_decomp_rows(w, records))
        print(
            f"[e2e] W={w} wall={wall:.3f}s "
            f"reward_start={result.reward_start:.4f} reward_end={result.reward_end:.4f}"
        )
    scaling_rows = gs.attach_speedups(scaling_rows)
    return scaling_rows, decomp_rows


def _throughput_recorded(scaling_rows: Sequence[dict[str, Any]]) -> bool:
    """True iff EVERY W row has a finite, positive recorded throughput."""
    if not scaling_rows:
        return False
    for r in scaling_rows:
        thr = r.get("throughput_traj_per_s")
        if thr is None or math.isnan(thr) or thr <= 0.0:
            return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="e2e_grpo_ray",
        description="Self-contained end-to-end Ray-GRPO smoke (sweep + train + CSV).",
    )
    ap.add_argument("--actor-counts", default="1,2,4", help="comma W list (default 1,2,4)")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--batch-prompts", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="e2e_out")
    ap.add_argument(
        "--summary",
        default=None,
        help="optional path to also write the PASS/FAIL + DONE verdict file",
    )
    args = ap.parse_args(argv)

    cfg = E2EConfig(
        actor_counts=parse_actor_counts_arg(args.actor_counts),
        steps=args.steps,
        batch_prompts=args.batch_prompts,
        group_size=args.group_size,
        horizon=args.horizon,
        seed=args.seed,
        out_dir=Path(args.out_dir),
    )

    gs = _load_grpo_scale()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # (1) + (3): sweep and CSV emission.
    scaling_rows, decomp_rows = run_sweep_real(cfg, gs)
    scaling_csv = out / "scaling_results.csv"
    decomp_csv = out / "step_decomposition.csv"
    gs.write_csv(scaling_csv, gs.SCALING_COLUMNS, scaling_rows)
    gs.write_csv(decomp_csv, gs.DECOMP_COLUMNS, decomp_rows)
    csv_ok = scaling_csv.is_file() and decomp_csv.is_file() and len(decomp_rows) > 0

    throughput_ok = _throughput_recorded(scaling_rows)
    scaling_kind = classify_scaling(scaling_rows)

    # (2): a short train run with the reward-nondecreasing gate. We reuse the
    # serial-fallback pool (W=1) so this is a clean, Ray-free convergence check.
    from app.policy.driver import TrainConfig
    from app.policy.ray_driver import train_grpo_ray

    train_cfg = TrainConfig(
        steps=cfg.steps,
        batch_prompts=cfg.batch_prompts,
        group_size=cfg.group_size,
        horizon=cfg.horizon,
        seed=cfg.seed,
    )
    res = train_grpo_ray(train_cfg, reward_pool=_build_reward_pool(1))
    reward_ok = res.reward_end >= res.reward_start

    print()
    print(gs.format_summary(scaling_rows))
    print()
    print(f"[e2e] scaling_curve={scaling_kind} (report only, not a gate)")
    print(f"[e2e] reward_start={res.reward_start:.6f} reward_end={res.reward_end:.6f}")
    print(f"[e2e] wrote {scaling_csv}")
    print(f"[e2e] wrote {decomp_csv} ({len(decomp_rows)} rows)")

    line = verdict_line(throughput_ok=throughput_ok, reward_ok=reward_ok, csv_ok=csv_ok)
    print()
    print(line)

    if args.summary:
        spath = Path(args.summary)
        spath.parent.mkdir(parents=True, exist_ok=True)
        with spath.open("w") as fh:
            fh.write(line + "\n")
            fh.write(f"scaling_curve={scaling_kind}\n")
            fh.write(f"reward_start={res.reward_start:.6f}\n")
            fh.write(f"reward_end={res.reward_end:.6f}\n")
            fh.write(f"scaling_csv={scaling_csv}\n")
            fh.write(f"decomp_csv={decomp_csv}\n")
            fh.write(f"decomp_rows={len(decomp_rows)}\n")
            fh.write("\n")
            fh.write(gs.format_summary(scaling_rows) + "\n")
            fh.write("\nDONE\n")
        print(f"[e2e] wrote verdict -> {spath}")

    return 0 if line.startswith("PASS") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "E2EConfig",
    "remap_step_record",
    "classify_scaling",
    "parse_actor_counts_arg",
    "verdict_line",
    "run_sweep_real",
    "main",
]
