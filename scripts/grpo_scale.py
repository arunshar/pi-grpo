"""Reward-actor scaling sweep for the Ray GRPO path.

This script measures how the parallel reward-evaluation path scales with the
number of Ray reward actors ``W``. For each ``W`` in a sweep it runs a few
GRPO steps via ``train_grpo_ray`` and records:

* reward-eval throughput in trajectories/sec (the metric that the Ray path is
  meant to improve over the serial ``_reward_matrix`` bottleneck in
  ``app/policy/driver.py``), and
* the mean per-step time decomposition (sample / reward / learn seconds).

It writes two CSV files:

* ``scaling_results.csv``  one row per ``W`` (throughput + speedup vs W=1), and
* ``step_decomposition.csv`` one row per (W, step) with the timing breakdown.

DESIGN NOTE (additive / opt-in). The pure helpers in this module (row builders,
throughput / speedup math, arg parsing, CSV writers) have NO dependency on Ray
or torch and are unit-tested in ``tests/test_grpo_scale.py``. The actual sweep
(``run_sweep``) lazily imports the Ray GRPO entrypoint, so importing this module
on a CPU-only login node (or in the test env) never pulls in Ray.

TRAIN_GRPO_RAY CONTRACT (provided by the sibling Ray-driver task). ``run_sweep``
expects a callable ``train_grpo_ray(cfg, *, num_reward_actors)`` that runs a
short GRPO run on ``num_reward_actors`` Ray reward actors and returns a result
exposing per-step timing. We read it through ``extract_step_records`` which
tolerates either:

  result.history  -> list[dict] with optional keys
                     {"sample_s", "reward_s", "learn_s", "n_traj",
                      "reward_mean", "loss", "kl"}

or a plain ``list[dict]`` of the same shape. Any missing timing key is recorded
as ``float("nan")`` so a partial Ray implementation still produces a valid CSV
rather than crashing. See the TODOs near ``run_sweep`` for the exact fields the
Ray driver must populate for the throughput numbers to be meaningful.
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

# Per-W summary row. Order is load-bearing: the CSV header is written from this.
SCALING_COLUMNS: tuple[str, ...] = (
    "num_reward_actors",      # W
    "steps",                  # GRPO steps run for this W
    "batch_prompts",          # B
    "group_size",             # K
    "horizon",                # T_r (control tokens per rollout)
    "n_trajectories",         # total trajectories scored across the run = B*K*steps
    "reward_eval_seconds",    # summed reward-eval wall time across the run
    "throughput_traj_per_s",  # n_trajectories / reward_eval_seconds
    "speedup_vs_w1",          # throughput(W) / throughput(W=1)
    "mean_sample_s",          # mean per-step sampling time
    "mean_reward_s",          # mean per-step reward-eval time
    "mean_learn_s",           # mean per-step learn (step_update) time
    "mean_step_s",            # mean per-step total time
)

# Per-(W, step) decomposition row.
DECOMP_COLUMNS: tuple[str, ...] = (
    "num_reward_actors",
    "step",
    "sample_s",
    "reward_s",
    "learn_s",
    "step_s",
    "n_traj",
    "reward_mean",
    "loss",
    "kl",
)

# Default sweep over reward-actor counts.
DEFAULT_ACTOR_COUNTS: tuple[int, ...] = (1, 2, 4, 8, 16)


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class SweepConfig:
    """Knobs for one scaling sweep.

    ``actor_counts`` is the list of W values to sweep. The GRPO run shape
    (steps / batch_prompts / group_size / horizon) is held FIXED across W so the
    only thing that changes between rows is the number of reward actors, which
    is what makes the throughput curve interpretable.
    """

    actor_counts: tuple[int, ...] = DEFAULT_ACTOR_COUNTS
    steps: int = 5
    batch_prompts: int = 8
    group_size: int = 8
    horizon: int = 12
    seed: int = 42
    out_dir: Path = field(default_factory=lambda: Path("scaling_out"))


# --------------------------------------------------------------------------- pure helpers


def n_trajectories(steps: int, batch_prompts: int, group_size: int) -> int:
    """Total trajectories scored across a run: B*K per step, ``steps`` steps."""
    return int(steps) * int(batch_prompts) * int(group_size)


def trajectories_per_sec(n_traj: float, seconds: float) -> float:
    """Reward-eval throughput. Returns ``nan`` when no time elapsed / no work.

    Guards against a zero / non-positive denominator (a degenerate run with no
    measured reward time) so the sweep never divides by zero.
    """
    if seconds is None or seconds <= 0.0 or n_traj <= 0:
        return float("nan")
    return float(n_traj) / float(seconds)


def _mean(values: Sequence[float]) -> float:
    """Mean that ignores ``nan`` entries; returns ``nan`` if all are nan/empty."""
    finite = [v for v in values if v is not None and not math.isnan(v)]
    if not finite:
        return float("nan")
    return math.fsum(finite) / len(finite)


def extract_step_records(result: Any) -> list[dict[str, float]]:
    """Normalize a ``train_grpo_ray`` return into a list of per-step dicts.

    Accepts either an object with a ``.history`` list-of-dicts attribute or a
    bare list of dicts. Every record is filled out with all DECOMP timing keys
    (missing ones become ``nan``) so downstream row building is total.
    """
    if hasattr(result, "history"):
        raw = result.history
    else:
        raw = result
    if not isinstance(raw, (list, tuple)):
        raise TypeError(
            "train_grpo_ray result must expose a list of per-step dicts via "
            f".history or be such a list; got {type(raw)!r}"
        )
    records: list[dict[str, float]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise TypeError(f"each step record must be a dict; got {type(item)!r}")
        records.append(
            {
                # ray_driver emits t_sample / t_score / t_learn per step; the
                # pure-helper tests use sample_s / reward_s / learn_s stubs.
                # Accept BOTH so the harness reads the real driver result and the
                # unit-test fixtures. reward_s (the scored / reward-eval phase) is
                # what the throughput curve is built on, so the t_score fallback
                # is the one that makes a live sweep non-NaN.
                "sample_s": float(item.get("sample_s", item.get("t_sample", float("nan")))),
                "reward_s": float(item.get("reward_s", item.get("t_score", float("nan")))),
                "learn_s": float(item.get("learn_s", item.get("t_learn", float("nan")))),
                "n_traj": float(item.get("n_traj", float("nan"))),
                "reward_mean": float(item.get("reward_mean", float("nan"))),
                "loss": float(item.get("loss", float("nan"))),
                "kl": float(item.get("kl", float("nan"))),
            }
        )
    return records


def step_total_seconds(record: dict[str, float]) -> float:
    """Per-step wall time = sample + reward + learn (nan-tolerant sum)."""
    return _mean_sum(record.get("sample_s"), record.get("reward_s"), record.get("learn_s"))


def _mean_sum(*parts: float | None) -> float:
    """Sum of the finite parts; ``nan`` only if every part is missing."""
    finite = [p for p in parts if p is not None and not math.isnan(p)]
    if not finite:
        return float("nan")
    return math.fsum(finite)


def build_decomp_rows(
    num_reward_actors: int, records: Sequence[dict[str, float]]
) -> list[dict[str, Any]]:
    """One DECOMP row per step for a single W."""
    rows: list[dict[str, Any]] = []
    for step, rec in enumerate(records):
        rows.append(
            {
                "num_reward_actors": int(num_reward_actors),
                "step": step,
                "sample_s": rec["sample_s"],
                "reward_s": rec["reward_s"],
                "learn_s": rec["learn_s"],
                "step_s": step_total_seconds(rec),
                "n_traj": rec["n_traj"],
                "reward_mean": rec["reward_mean"],
                "loss": rec["loss"],
                "kl": rec["kl"],
            }
        )
    return rows


def build_scaling_row(
    *,
    num_reward_actors: int,
    cfg: SweepConfig,
    records: Sequence[dict[str, float]],
) -> dict[str, Any]:
    """Build one ``scaling_results.csv`` summary row for a single W.

    ``speedup_vs_w1`` is left as ``nan`` here; it is filled in by
    ``attach_speedups`` once the whole sweep is collected and the W=1 baseline is
    known.
    """
    n_traj = n_trajectories(cfg.steps, cfg.batch_prompts, cfg.group_size)
    reward_seconds = _mean_sum(*[r["reward_s"] for r in records])
    throughput = trajectories_per_sec(n_traj, reward_seconds)
    return {
        "num_reward_actors": int(num_reward_actors),
        "steps": int(cfg.steps),
        "batch_prompts": int(cfg.batch_prompts),
        "group_size": int(cfg.group_size),
        "horizon": int(cfg.horizon),
        "n_trajectories": n_traj,
        "reward_eval_seconds": reward_seconds,
        "throughput_traj_per_s": throughput,
        "speedup_vs_w1": float("nan"),
        "mean_sample_s": _mean([r["sample_s"] for r in records]),
        "mean_reward_s": _mean([r["reward_s"] for r in records]),
        "mean_learn_s": _mean([r["learn_s"] for r in records]),
        "mean_step_s": _mean([step_total_seconds(r) for r in records]),
    }


def attach_speedups(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fill ``speedup_vs_w1`` for every row using the W=1 throughput baseline.

    Speedup is throughput(W) / throughput(W=1). If there is no W=1 row, or its
    throughput is not a positive finite number, every speedup is ``nan`` (we do
    not silently rebase onto the smallest W, because that would misreport the
    scaling curve). Rows are returned as new dicts; the input is not mutated.
    """
    baseline = None
    for r in rows:
        if int(r["num_reward_actors"]) == 1:
            baseline = r["throughput_traj_per_s"]
            break
    out: list[dict[str, Any]] = []
    valid_base = (
        baseline is not None
        and not math.isnan(baseline)
        and baseline > 0.0
    )
    for r in rows:
        new = dict(r)
        thr = r["throughput_traj_per_s"]
        if valid_base and thr is not None and not math.isnan(thr):
            new["speedup_vs_w1"] = float(thr) / float(baseline)
        else:
            new["speedup_vs_w1"] = float("nan")
        out.append(new)
    return out


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    """Write ``rows`` to ``path`` with a fixed ``columns`` header (creates dirs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_summary(rows: Sequence[dict[str, Any]]) -> str:
    """Human-readable scaling table for the DONE/summary marker."""
    lines = ["reward-actor scaling sweep", ""]
    header = f"{'W':>4} {'traj/s':>12} {'speedup':>9} {'mean_step_s':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        lines.append(
            f"{int(r['num_reward_actors']):>4} "
            f"{r['throughput_traj_per_s']:>12.2f} "
            f"{r['speedup_vs_w1']:>9.2f} "
            f"{r['mean_step_s']:>12.4f}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- arg parsing


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="grpo_scale",
        description="Sweep Ray reward-actor count and record GRPO reward-eval throughput.",
    )
    ap.add_argument(
        "--actor-counts",
        type=str,
        default=",".join(str(w) for w in DEFAULT_ACTOR_COUNTS),
        help="comma-separated reward-actor counts W (default: 1,2,4,8,16)",
    )
    ap.add_argument("--steps", type=int, default=5, help="GRPO steps per W (default: 5)")
    ap.add_argument("--batch-prompts", type=int, default=8, help="B prompts per step (default: 8)")
    ap.add_argument("--group-size", type=int, default=8, help="K rollouts per prompt (default: 8)")
    ap.add_argument("--horizon", type=int, default=12, help="control tokens per rollout (default: 12)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    ap.add_argument(
        "--out-dir",
        type=str,
        default="scaling_out",
        help="directory for scaling_results.csv + step_decomposition.csv",
    )
    return ap


def parse_actor_counts(spec: str) -> tuple[int, ...]:
    """Parse '1,2,4' -> (1, 2, 4). Rejects empty / non-positive entries."""
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


def config_from_args(args: argparse.Namespace) -> SweepConfig:
    return SweepConfig(
        actor_counts=parse_actor_counts(args.actor_counts),
        steps=args.steps,
        batch_prompts=args.batch_prompts,
        group_size=args.group_size,
        horizon=args.horizon,
        seed=args.seed,
        out_dir=Path(args.out_dir),
    )


# --------------------------------------------------------------------------- sweep (live)


def _run_one(num_reward_actors: int, cfg: SweepConfig) -> list[dict[str, float]]:
    """Run one short GRPO run on ``num_reward_actors`` Ray reward actors.

    Lazily imports the Ray GRPO entrypoint so that importing this module never
    requires Ray. The entrypoint and its config are provided by app.policy.

    Wired to the real ray_driver API: train_grpo_ray(cfg, reward_pool=None,
    on_step=None) where cfg is app.policy.driver.TrainConfig. The reward-actor
    count W is set by the reward pool, default_reward_pool(num_workers=W) returns
    a serial pool for W<=1 and a W-actor RayRewardPool otherwise, NOT by a
    num_reward_actors kwarg (which the stubbed version wrongly assumed). The
    result exposes per-step timing via .history, each dict carrying t_sample /
    t_score / t_learn; extract_step_records maps those onto sample_s / reward_s /
    learn_s.
    """
    try:
        from app.policy.ray_driver import train_grpo_ray, default_reward_pool
        from app.policy.driver import TrainConfig
    except ImportError as exc:  # pragma: no cover - exercised only on a live cluster
        raise RuntimeError(
            "Ray GRPO entrypoint not importable. This sweep requires "
            "app/policy/ray_driver.py (train_grpo_ray + default_reward_pool), "
            "app/policy/driver.py (TrainConfig), and a Ray install. Run on a "
            "compute node in a ray-enabled env."
        ) from exc

    cfg_obj = TrainConfig(
        steps=cfg.steps,
        batch_prompts=cfg.batch_prompts,
        group_size=cfg.group_size,
        horizon=cfg.horizon,
        seed=cfg.seed,
    )
    pool = default_reward_pool(num_workers=num_reward_actors)
    result = train_grpo_ray(cfg_obj, reward_pool=pool)
    return extract_step_records(result)


def run_sweep(cfg: SweepConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the full sweep and return (scaling_rows, decomposition_rows).

    Live path only (imports Ray). Pure-helper tests do not call this; they test
    the row/throughput/speedup helpers directly with stub records.
    """
    scaling_rows: list[dict[str, Any]] = []
    decomp_rows: list[dict[str, Any]] = []
    for w in cfg.actor_counts:
        records = _run_one(w, cfg)
        scaling_rows.append(build_scaling_row(num_reward_actors=w, cfg=cfg, records=records))
        decomp_rows.extend(build_decomp_rows(w, records))
    scaling_rows = attach_speedups(scaling_rows)
    return scaling_rows, decomp_rows


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = config_from_args(args)
    scaling_rows, decomp_rows = run_sweep(cfg)
    out = Path(cfg.out_dir)
    write_csv(out / "scaling_results.csv", SCALING_COLUMNS, scaling_rows)
    write_csv(out / "step_decomposition.csv", DECOMP_COLUMNS, decomp_rows)
    print(format_summary(scaling_rows))
    print(f"\nwrote {out / 'scaling_results.csv'}")
    print(f"wrote {out / 'step_decomposition.csv'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SCALING_COLUMNS",
    "DECOMP_COLUMNS",
    "DEFAULT_ACTOR_COUNTS",
    "SweepConfig",
    "n_trajectories",
    "trajectories_per_sec",
    "extract_step_records",
    "step_total_seconds",
    "build_decomp_rows",
    "build_scaling_row",
    "attach_speedups",
    "write_csv",
    "format_summary",
    "build_arg_parser",
    "parse_actor_counts",
    "config_from_args",
    "run_sweep",
    "main",
]
