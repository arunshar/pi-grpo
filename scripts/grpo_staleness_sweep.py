"""EXP-4: async / off-policy staleness sweep for the GRPO Ray path.

This sweeps the bounded-queue staleness admission knob ``max_staleness`` (s) over
``s = 0, 1, 2, 4, 8`` and, for each s, runs the in-process async/off-policy harness
``app.policy.ray_staleness.train_grpo_async_staleness`` for a fixed GRPO shape. The
only thing that changes between rows is s, so the resulting curve isolates the
staleness/off-policy cost.

For each s it records:

* the staleness accounting: mean / max consumed lag and the on-policy fraction
  (with s = 0 the harness reduces to the synchronous loop, lag 0, fraction 1.0);
* the off-policy KL cost: the mean per-step GRPO KL-to-reference over the consumed
  steps (this is the quantity staleness inflates), and the final reward;
* a systems-demo throughput (learner steps / wall second) and the queue admission
  counters (admitted / rejected-stale / evicted).

HONEST SCOPE. On the tiny motion-primitive policy the throughput number is a
systems demonstration, not a speedup claim (see the ``ray_staleness`` module
docstring): sampling / scoring / learn are all sub-millisecond, so async
bookkeeping dominates. The meaningful, real signal is the staleness/KL/reward
trade: as s grows, mean lag and the off-policy KL rise and reward can degrade. The
"knee" s* reported here is the largest s whose final reward is still within a
tolerance of the s = 0 baseline (the most off-policy the run can go before reward
pays for it); it is a heuristic read of the curve, not a throughput-optimal point.

DESIGN NOTE (additive / opt-in, mirrors ``grpo_scale.py``). The pure helpers here
(row builders, the delta / knee math, arg parsing, CSV writers) have NO dependency
on Ray or torch and are unit-tested in ``tests/test_grpo_staleness_sweep.py``. The
live sweep (``run_sweep``) lazily imports the torch-backed staleness harness, so
importing this module on a CPU-only login node or in the test env never pulls in
torch.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- columns

# Per-s summary row. Order is load-bearing: the CSV header is written from this.
STALENESS_COLUMNS: tuple[str, ...] = (
    "max_staleness",            # s (the swept knob)
    "steps",                    # GRPO learner steps requested
    "batch_prompts",            # B
    "group_size",               # K
    "horizon",                  # T_r (control tokens per rollout)
    "producer_ahead",           # batches the producer runs ahead per step (s>0)
    "refresh_every",            # learner updates between producer weight refreshes
    "consumed_steps",           # learner steps that actually consumed a batch
    "mean_lag",                 # mean consumed staleness lag
    "max_lag",                  # max consumed staleness lag
    "on_policy_fraction",       # fraction of consumed batches with lag == 0
    "mean_kl",                  # mean per-step GRPO KL-to-ref over consumed steps
    "reward_start",             # mean rollout reward before training
    "reward_end",               # mean rollout reward after training
    "reward_gain",              # reward_end - reward_start
    "reward_gap_vs_s0",         # reward_end(s) - reward_end(s=0); filled by attach
    "throughput_steps_per_s",   # learner steps / wall second (systems demo only)
    "throughput_ratio_vs_s0",   # throughput(s) / throughput(s=0); filled by attach
    "wall_time_s",              # end-to-end seconds for the run
    "queue_admitted",           # batches admitted to the staleness queue
    "queue_rejected_stale",     # batches rejected at enqueue for being too stale
    "queue_evicted",            # batches evicted (DROP_OLDEST / drain_stale)
)

# Default staleness sweep. 0 is the synchronous on-policy baseline.
DEFAULT_STALENESS: tuple[int, ...] = (0, 1, 2, 4, 8)


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class SweepConfig:
    """Knobs for one staleness sweep.

    ``staleness_values`` is the list of s values to sweep. The GRPO run shape
    (steps / batch_prompts / group_size / horizon) and the async schedule
    (``producer_ahead`` / ``refresh_every`` for s > 0) are held FIXED across s so
    the only thing that changes between rows is the staleness bound, which is what
    makes the staleness/KL/reward curve interpretable. ``s == 0`` always uses the
    synchronous reduction (queue_maxsize 1, producer_ahead 1), matching the
    baseline arm of ``ray_staleness.run_staleness_comparison``.
    """

    staleness_values: tuple[int, ...] = DEFAULT_STALENESS
    steps: int = 60
    batch_prompts: int = 8
    group_size: int = 8
    horizon: int = 12
    seed: int = 42
    queue_maxsize: int = 4       # capacity for the s > 0 runs
    producer_ahead: int = 2      # batches produced ahead per step for s > 0
    refresh_every: int = 2       # learner updates between producer weight refreshes
    out_dir: Path = field(default_factory=lambda: Path("staleness_out"))


# --------------------------------------------------------------------------- pure helpers


def parse_staleness(spec: str) -> tuple[int, ...]:
    """Parse '0,1,2,4,8' -> (0, 1, 2, 4, 8). Rejects empty / negative entries.

    Unlike the reward-actor counts in ``grpo_scale``, 0 is a VALID and important
    value here: it is the synchronous on-policy baseline the rest of the sweep is
    measured against. Only negative values are rejected.
    """
    values: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        s = int(part)
        if s < 0:
            raise ValueError(f"max_staleness must be >= 0; got {s}")
        values.append(s)
    if not values:
        raise ValueError(f"no staleness values parsed from {spec!r}")
    return tuple(values)


def _mean_finite(values: Sequence[float]) -> float:
    """Mean that ignores ``nan`` entries; returns ``nan`` if all are nan / empty."""
    finite = [v for v in values if v is not None and not math.isnan(float(v))]
    if not finite:
        return float("nan")
    return math.fsum(float(v) for v in finite) / len(finite)


def mean_kl_from_history(history: Sequence[dict]) -> float:
    """Mean per-step GRPO KL-to-reference over the consumed steps.

    Each history record is the GRPO metrics dict from ``trainer.step_update`` (it
    carries a ``"kl"`` key) plus the consumed batch ``"lag"``. A record missing
    ``"kl"`` contributes ``nan`` and is ignored by the nan-tolerant mean, so a
    partial history still yields a number rather than crashing.
    """
    return _mean_finite([rec.get("kl", float("nan")) for rec in history])


def throughput_steps_per_s(final_step: float, wall_time: float) -> float:
    """Learner steps / wall second. ``nan`` for a degenerate (no time / no work) run."""
    if wall_time is None or wall_time <= 0.0 or final_step is None or final_step <= 0:
        return float("nan")
    return float(final_step) / float(wall_time)


def build_staleness_row(*, max_staleness: int, cfg: SweepConfig, result: Any) -> dict[str, Any]:
    """Build one ``staleness_sweep.csv`` row from an ``AsyncTrainResult``-like object.

    ``result`` is duck-typed: it must expose ``final_step``, ``wall_time``,
    ``reward_start``, ``reward_end``, ``history`` (list of GRPO metrics dicts),
    ``queue_admitted`` / ``queue_rejected_stale`` / ``queue_evicted``, and a
    ``staleness`` summary with ``mean_lag`` / ``max_lag`` / ``on_policy_fraction``.
    ``reward_gap_vs_s0`` and ``throughput_ratio_vs_s0`` are left ``nan`` here and
    filled by ``attach_baseline_deltas`` once the whole sweep is collected.
    """
    st = result.staleness
    final_step = float(result.final_step)
    wall_time = float(result.wall_time)
    reward_start = float(result.reward_start)
    reward_end = float(result.reward_end)
    return {
        "max_staleness": int(max_staleness),
        "steps": int(cfg.steps),
        "batch_prompts": int(cfg.batch_prompts),
        "group_size": int(cfg.group_size),
        "horizon": int(cfg.horizon),
        "producer_ahead": 1 if max_staleness == 0 else int(cfg.producer_ahead),
        "refresh_every": 1 if max_staleness == 0 else int(cfg.refresh_every),
        "consumed_steps": int(final_step),
        "mean_lag": float(st.mean_lag),
        "max_lag": int(st.max_lag),
        "on_policy_fraction": float(st.on_policy_fraction),
        "mean_kl": mean_kl_from_history(result.history),
        "reward_start": reward_start,
        "reward_end": reward_end,
        "reward_gain": reward_end - reward_start,
        "reward_gap_vs_s0": float("nan"),
        "throughput_steps_per_s": throughput_steps_per_s(final_step, wall_time),
        "throughput_ratio_vs_s0": float("nan"),
        "wall_time_s": wall_time,
        "queue_admitted": int(result.queue_admitted),
        "queue_rejected_stale": int(result.queue_rejected_stale),
        "queue_evicted": int(result.queue_evicted),
    }


def attach_baseline_deltas(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fill ``reward_gap_vs_s0`` and ``throughput_ratio_vs_s0`` against the s = 0 row.

    The s = 0 row is the synchronous on-policy baseline. If there is no s = 0 row
    (or its throughput is not a positive finite number) the ratio is left ``nan``
    rather than silently rebasing onto the smallest s. Rows are returned as new
    dicts; the input is not mutated.
    """
    base_reward = None
    base_throughput = None
    for r in rows:
        if int(r["max_staleness"]) == 0:
            base_reward = r["reward_end"]
            base_throughput = r["throughput_steps_per_s"]
            break
    valid_thr_base = (
        base_throughput is not None
        and not math.isnan(base_throughput)
        and base_throughput > 0.0
    )
    out: list[dict[str, Any]] = []
    for r in rows:
        new = dict(r)
        if base_reward is not None and not math.isnan(base_reward):
            new["reward_gap_vs_s0"] = float(r["reward_end"]) - float(base_reward)
        else:
            new["reward_gap_vs_s0"] = float("nan")
        thr = r["throughput_steps_per_s"]
        if valid_thr_base and thr is not None and not math.isnan(thr):
            new["throughput_ratio_vs_s0"] = float(thr) / float(base_throughput)
        else:
            new["throughput_ratio_vs_s0"] = float("nan")
        out.append(new)
    return out


def find_knee(rows: Sequence[dict[str, Any]], *, reward_tol: float = 0.05) -> int | None:
    """Heuristic knee s*: the largest s whose reward is still within ``reward_tol``.

    Defined as the largest staleness whose ``reward_end`` has not dropped below the
    s = 0 baseline by more than ``reward_tol`` of the baseline magnitude. This is
    the most off-policy the run can go before final reward starts paying for the
    staleness. Returns ``None`` if there is no s = 0 baseline. This is a read of the
    curve, NOT a throughput-optimal claim (throughput is a systems demo on the tiny
    model); the full per-s table is the real deliverable.
    """
    base_reward = None
    for r in rows:
        if int(r["max_staleness"]) == 0:
            base_reward = r["reward_end"]
            break
    if base_reward is None or math.isnan(base_reward):
        return None
    tol = reward_tol * abs(base_reward)
    knee = 0
    for r in sorted(rows, key=lambda x: int(x["max_staleness"])):
        s = int(r["max_staleness"])
        re = r["reward_end"]
        if re is None or math.isnan(re):
            continue
        # reward_end is "good enough" if it has not dropped more than tol below baseline.
        if float(re) >= float(base_reward) - tol:
            knee = s
        else:
            break
    return knee


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    """Write ``rows`` to ``path`` with a fixed ``columns`` header (creates dirs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_summary(rows: Sequence[dict[str, Any]], knee: int | None) -> str:
    """Human-readable staleness table for the DONE / summary marker."""
    lines = ["async/off-policy staleness sweep (EXP-4)", ""]
    header = (
        f"{'s':>3} {'mean_lag':>9} {'on_pol':>7} {'mean_kl':>10} "
        f"{'reward_end':>11} {'rew_gap':>9} {'thr/s':>9}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for r in sorted(rows, key=lambda x: int(x["max_staleness"])):
        lines.append(
            f"{int(r['max_staleness']):>3} "
            f"{r['mean_lag']:>9.3f} "
            f"{r['on_policy_fraction']:>7.2f} "
            f"{r['mean_kl']:>10.5f} "
            f"{r['reward_end']:>11.4f} "
            f"{r['reward_gap_vs_s0']:>9.4f} "
            f"{r['throughput_steps_per_s']:>9.2f}"
        )
    lines.append("")
    lines.append(
        f"knee s* (largest s within reward tolerance of s=0): {knee}"
        if knee is not None
        else "knee s*: undefined (no s=0 baseline row)"
    )
    lines.append(
        "NOTE: throughput is a systems demo on the tiny policy, not a speedup claim; "
        "the staleness/KL/reward trade is the real signal."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- arg parsing


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="grpo_staleness_sweep",
        description="Sweep the async staleness bound s and record the staleness/KL/reward trade.",
    )
    ap.add_argument(
        "--staleness",
        type=str,
        default=",".join(str(s) for s in DEFAULT_STALENESS),
        help="comma-separated max_staleness values s (default: 0,1,2,4,8; 0 = on-policy baseline)",
    )
    ap.add_argument("--steps", type=int, default=60, help="GRPO learner steps per s (default: 60)")
    ap.add_argument("--batch-prompts", type=int, default=8, help="B prompts per step (default: 8)")
    ap.add_argument("--group-size", type=int, default=8, help="K rollouts per prompt (default: 8)")
    ap.add_argument("--horizon", type=int, default=12, help="control tokens per rollout (default: 12)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    ap.add_argument("--queue-maxsize", type=int, default=4, help="staleness queue capacity for s>0 (default: 4)")
    ap.add_argument("--producer-ahead", type=int, default=2, help="producer batches ahead/step, s>0 (def 2)")
    ap.add_argument("--refresh-every", type=int, default=2, help="updates between producer refreshes (def 2)")
    ap.add_argument(
        "--out-dir",
        type=str,
        default="staleness_out",
        help="directory for staleness_sweep.csv",
    )
    return ap


def config_from_args(args: argparse.Namespace) -> SweepConfig:
    return SweepConfig(
        staleness_values=parse_staleness(args.staleness),
        steps=args.steps,
        batch_prompts=args.batch_prompts,
        group_size=args.group_size,
        horizon=args.horizon,
        seed=args.seed,
        queue_maxsize=args.queue_maxsize,
        producer_ahead=args.producer_ahead,
        refresh_every=args.refresh_every,
        out_dir=Path(args.out_dir),
    )


# --------------------------------------------------------------------------- sweep (live)


def _run_one(max_staleness: int, cfg: SweepConfig):
    """Run the async staleness harness once for a single s.

    Lazily imports the torch-backed harness so importing this module never
    requires torch. s == 0 uses the synchronous reduction (queue_maxsize 1,
    producer_ahead 1), exactly the baseline arm of ``run_staleness_comparison``;
    s > 0 uses the swept queue / producer schedule held fixed across the sweep.
    """
    try:
        from app.policy.driver import TrainConfig
        from app.policy.ray_staleness import StalenessConfig, train_grpo_async_staleness
    except ImportError as exc:  # pragma: no cover - exercised only in a torch env
        raise RuntimeError(
            "staleness harness not importable. This sweep requires "
            "app/policy/ray_staleness.py + app/policy/driver.py and a torch install. "
            "Run in the project env (pcrf) on a compute node."
        ) from exc

    cfg_obj = TrainConfig(
        steps=cfg.steps,
        batch_prompts=cfg.batch_prompts,
        group_size=cfg.group_size,
        horizon=cfg.horizon,
        seed=cfg.seed,
    )
    if max_staleness == 0:
        sc = StalenessConfig(max_staleness=0, queue_maxsize=1, producer_ahead=1, refresh_every=1)
    else:
        sc = StalenessConfig(
            max_staleness=max_staleness,
            queue_maxsize=cfg.queue_maxsize,
            producer_ahead=cfg.producer_ahead,
            refresh_every=cfg.refresh_every,
        )
    return train_grpo_async_staleness(cfg_obj, sc)


def run_sweep(cfg: SweepConfig) -> tuple[list[dict[str, Any]], int | None]:
    """Run the full staleness sweep; return (rows, knee_staleness).

    Live path only (imports torch). Pure-helper tests do not call this; they test
    the row / delta / knee helpers directly with stub results.
    """
    rows: list[dict[str, Any]] = []
    for s in cfg.staleness_values:
        result = _run_one(s, cfg)
        rows.append(build_staleness_row(max_staleness=s, cfg=cfg, result=result))
    rows = attach_baseline_deltas(rows)
    return rows, find_knee(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = config_from_args(args)
    rows, knee = run_sweep(cfg)
    out = Path(cfg.out_dir)
    write_csv(out / "staleness_sweep.csv", STALENESS_COLUMNS, rows)
    print(format_summary(rows, knee))
    print(f"\nwrote {out / 'staleness_sweep.csv'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
