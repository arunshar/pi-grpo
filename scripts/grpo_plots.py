"""Plots for the Ray-parallel GRPO reward-evaluation scaling study.

This module turns a list of per-worker-count benchmark records into three
publication figures for the PI-GRPO paper:

1. ``plot_throughput_vs_workers`` reward-eval throughput (rollouts scored per
   second) as a function of the number of Ray reward workers ``W``, with the
   ideal-linear reference line (W * single-worker throughput).
2. ``plot_scaling_efficiency`` parallel efficiency ``T_W / (W * T_1)`` where
   ``T_W`` is reward-eval throughput at ``W`` workers (1.0 == perfect scaling).
3. ``plot_step_decomposition`` a stacked per-step decomposition of wall time
   into sample / score / learn, showing the score (reward-eval) bar shrinking
   as a fraction of the step as ``W`` grows: the bottleneck moves off
   reward-eval.

Design contract
---------------
* Pure standard library + matplotlib. No seaborn, no pandas, no numpy.
* matplotlib runs head-less via the ``Agg`` backend (selected before pyplot
  is imported), so every plot writes a PNG with no display attached.
* Opt-in / additive: each plot only draws the columns that are actually
  present in the records. A record missing ``sample_s`` / ``score_s`` /
  ``learn_s`` is simply skipped by the decomposition plot rather than raising,
  so partially-instrumented benchmark runs still render the figures they can.

Input record schema (a record is a plain ``dict``)
--------------------------------------------------
Required for every plot:
    ``workers``         int   number of Ray reward workers W (>= 1)

Throughput / efficiency plots use, in order of preference:
    ``throughput``      float rollouts scored per second at this W, OR
    ``score_s`` (+ ``n_rollouts``) to derive throughput = n_rollouts / score_s.

Decomposition plot uses the per-step wall-time breakdown:
    ``sample_s``        float seconds spent sampling K rollouts (policy.generate)
    ``score_s``         float seconds spent scoring rollouts (the reward matrix)
    ``learn_s``         float seconds spent in the trainer step_update

All time fields are seconds. The module never mutates the input records.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib

matplotlib.use("Agg")  # head-less: must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402  (intentional: backend set first)

# The three phases of one GRPO step, in execution order. Used by the
# decomposition plot and its fraction helper.
PHASES: tuple[str, ...] = ("sample", "score", "learn")
_PHASE_KEYS: dict[str, str] = {p: f"{p}_s" for p in PHASES}


# --------------------------------------------------------------------------- math helpers


def throughput_of(record: Mapping[str, Any]) -> float | None:
    """Return rollouts-scored-per-second for one benchmark record, or None.

    Prefers an explicit ``throughput`` column. Otherwise derives it from
    ``n_rollouts / score_s`` when both are present and ``score_s > 0``.
    Returns ``None`` when neither form is available so callers can skip the
    point (opt-in behaviour).
    """

    if "throughput" in record and record["throughput"] is not None:
        val = float(record["throughput"])
        return val if val > 0.0 else None
    n = record.get("n_rollouts")
    score_s = record.get("score_s")
    if n is not None and score_s is not None and float(score_s) > 0.0:
        return float(n) / float(score_s)
    return None


def speedup_series(
    workers: Sequence[float], throughput: Sequence[float]
) -> list[float]:
    """Speedup ``T_W / T_1`` for each worker count.

    ``T_1`` is the throughput at the smallest worker count in the series
    (the single-worker / baseline point). The two sequences must be aligned
    and non-empty, and the baseline throughput must be positive.
    """

    if len(workers) != len(throughput):
        raise ValueError("workers and throughput must have equal length")
    if not workers:
        raise ValueError("need at least one (W, throughput) point")
    base_idx = min(range(len(workers)), key=lambda i: workers[i])
    t1 = float(throughput[base_idx])
    if t1 <= 0.0:
        raise ValueError("baseline throughput T_1 must be positive")
    return [float(t) / t1 for t in throughput]


def efficiency_series(
    workers: Sequence[float], throughput: Sequence[float]
) -> list[float]:
    """Parallel efficiency ``T_W / (W * T_1)`` for each worker count.

    Efficiency 1.0 is perfect linear scaling; values below 1.0 quantify the
    parallel overhead. The baseline worker count is taken as the minimum W in
    the series (normally 1) and must be positive.
    """

    if len(workers) != len(throughput):
        raise ValueError("workers and throughput must have equal length")
    if not workers:
        raise ValueError("need at least one (W, throughput) point")
    base_idx = min(range(len(workers)), key=lambda i: workers[i])
    w1 = float(workers[base_idx])
    if w1 <= 0.0:
        raise ValueError("baseline worker count W_1 must be positive")
    speedups = speedup_series(workers, throughput)
    out: list[float] = []
    for w, s in zip(workers, speedups, strict=True):
        if float(w) <= 0.0:
            raise ValueError("worker counts must be positive")
        # speedup is normalised to T_1; efficiency normalises by W/W_1 too.
        out.append(s / (float(w) / w1))
    return out


def ideal_linear_throughput(
    workers: Sequence[float], t1: float
) -> list[float]:
    """Ideal-linear reference: ``(W / W_1) * T_1`` for each worker count.

    ``W_1`` is the minimum worker count in the series (the point at which the
    measured throughput equals ``T_1``).
    """

    if not workers:
        raise ValueError("need at least one worker count")
    w1 = float(min(workers))
    if w1 <= 0.0:
        raise ValueError("baseline worker count must be positive")
    return [(float(w) / w1) * float(t1) for w in workers]


def decomposition_fractions(record: Mapping[str, Any]) -> dict[str, float] | None:
    """Per-step time split into sample/score/learn fractions summing to 1.0.

    Returns ``None`` when the record carries none of the phase timings (so the
    caller can skip it). When at least one phase is present, any missing phase
    is treated as zero time. The total must be positive.
    """

    seconds: dict[str, float] = {}
    present = False
    for phase, key in _PHASE_KEYS.items():
        val = record.get(key)
        if val is None:
            seconds[phase] = 0.0
        else:
            seconds[phase] = float(val)
            present = True
    if not present:
        return None
    total = sum(seconds.values())
    if total <= 0.0:
        raise ValueError("total step time must be positive")
    return {phase: seconds[phase] / total for phase in PHASES}


def _sorted_by_workers(
    records: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Records sorted ascending by worker count; raises if any lacks W."""

    for r in records:
        if "workers" not in r:
            raise KeyError("every record must carry a 'workers' field")
    return sorted(records, key=lambda r: float(r["workers"]))


# --------------------------------------------------------------------------- plots


def plot_throughput_vs_workers(
    records: Sequence[Mapping[str, Any]], out_path: str
) -> str:
    """Throughput vs W with the ideal-linear reference line. Writes a PNG.

    Only records that yield a throughput (see :func:`throughput_of`) are
    plotted. Returns the written path.
    """

    rows = _sorted_by_workers(records)
    pts: list[tuple[float, float]] = []
    for r in rows:
        thr = throughput_of(r)
        if thr is not None:
            pts.append((float(r["workers"]), thr))
    if not pts:
        raise ValueError("no records yielded a throughput to plot")

    ws = [w for w, _ in pts]
    ts = [t for _, t in pts]
    t1 = ts[0]  # rows are sorted; first is the smallest W (baseline)
    ideal = ideal_linear_throughput(ws, t1)

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.plot(ws, ts, marker="o", linewidth=1.8, label="measured")
    ax.plot(ws, ideal, linestyle="--", linewidth=1.4, color="0.5",
            label="ideal linear")
    ax.set_xlabel("reward workers W")
    ax.set_ylabel("reward-eval throughput (rollouts/s)")
    ax.set_title("GRPO reward-eval throughput vs workers")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_scaling_efficiency(
    records: Sequence[Mapping[str, Any]], out_path: str
) -> str:
    """Parallel efficiency ``T_W / (W * T_1)`` vs W. Writes a PNG.

    A horizontal reference at efficiency 1.0 marks perfect scaling. Returns
    the written path.
    """

    rows = _sorted_by_workers(records)
    pts: list[tuple[float, float]] = []
    for r in rows:
        thr = throughput_of(r)
        if thr is not None:
            pts.append((float(r["workers"]), thr))
    if not pts:
        raise ValueError("no records yielded a throughput to plot")

    ws = [w for w, _ in pts]
    ts = [t for _, t in pts]
    eff = efficiency_series(ws, ts)

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.plot(ws, eff, marker="s", linewidth=1.8, label="efficiency")
    ax.axhline(1.0, linestyle="--", linewidth=1.4, color="0.5",
               label="ideal (1.0)")
    ax.set_xlabel("reward workers W")
    ax.set_ylabel(r"efficiency  $T_W / (W \cdot T_1)$")
    ax.set_ylim(0.0, 1.1)
    ax.set_title("GRPO reward-eval scaling efficiency")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_step_decomposition(
    records: Sequence[Mapping[str, Any]], out_path: str
) -> str:
    """Stacked per-step time fractions (sample/score/learn) vs W. Writes a PNG.

    Each bar shows the fraction of one GRPO step spent in each phase; as W
    grows the ``score`` (reward-eval) band shrinks and the bottleneck moves
    onto sample/learn. Only records carrying at least one phase timing are
    drawn. Returns the written path.
    """

    rows = _sorted_by_workers(records)
    ws: list[float] = []
    fracs: list[dict[str, float]] = []
    for r in rows:
        fr = decomposition_fractions(r)
        if fr is not None:
            ws.append(float(r["workers"]))
            fracs.append(fr)
    if not fracs:
        raise ValueError("no records carried sample/score/learn timings")

    x = list(range(len(ws)))
    bottoms = [0.0] * len(ws)
    colors = {"sample": "#4C72B0", "score": "#DD8452", "learn": "#55A868"}

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for phase in PHASES:
        heights = [f[phase] for f in fracs]
        ax.bar(x, heights, bottom=bottoms, width=0.7,
               label=phase, color=colors[phase])
        bottoms = [b + h for b, h in zip(bottoms, heights, strict=True)]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(w)}" for w in ws])
    ax.set_xlabel("reward workers W")
    ax.set_ylabel("fraction of per-step wall time")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Per-step decomposition: reward-eval bottleneck vs W")
    ax.legend(loc="upper right", frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- CLI


def _load_records(path: str) -> list[dict[str, Any]]:
    """Load a JSON file that is either a list of records or {"records": [...]}"""

    import json

    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "records" in data:
        data = data["records"]
    if not isinstance(data, list):
        raise ValueError("expected a JSON list of records or {'records': [...]}")
    return [dict(r) for r in data]


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records_json",
                        help="JSON file: list of per-W benchmark records")
    parser.add_argument("--outdir", default=".",
                        help="directory to write the PNGs into")
    parser.add_argument("--prefix", default="grpo_",
                        help="filename prefix for the written PNGs")
    args = parser.parse_args(argv)

    import os

    records = _load_records(args.records_json)
    os.makedirs(args.outdir, exist_ok=True)

    def out(name: str) -> str:
        return os.path.join(args.outdir, f"{args.prefix}{name}.png")

    written: list[str] = []
    # Each plot is opt-in: skip (with a note) when no record supports it.
    try:
        written.append(plot_throughput_vs_workers(records, out("throughput")))
    except ValueError as exc:
        print(f"[skip] throughput plot: {exc}")
    try:
        written.append(plot_scaling_efficiency(records, out("efficiency")))
    except ValueError as exc:
        print(f"[skip] efficiency plot: {exc}")
    try:
        written.append(plot_step_decomposition(records, out("decomposition")))
    except ValueError as exc:
        print(f"[skip] decomposition plot: {exc}")

    for p in written:
        print(f"wrote {p}")
    return 0


__all__ = [
    "PHASES",
    "throughput_of",
    "speedup_series",
    "efficiency_series",
    "ideal_linear_throughput",
    "decomposition_fractions",
    "plot_throughput_vs_workers",
    "plot_scaling_efficiency",
    "plot_step_decomposition",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
