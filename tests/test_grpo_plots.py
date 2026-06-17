"""CPU-only, Agg-backend tests for scripts/grpo_plots.py.

These tests exercise (a) the pure math helpers on hand-computed values and
(b) that each plot writes a non-empty PNG into a tmp dir without raising. No
seaborn, no GPU, no Ray; matplotlib runs head-less via the Agg backend that
grpo_plots selects on import.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("matplotlib")  # skip cleanly on a matplotlib-less lane instead of failing collection

from scripts import grpo_plots as gp


# --------------------------------------------------------------------------- helpers


def test_throughput_of_prefers_explicit_column() -> None:
    rec = {"workers": 4, "throughput": 123.0, "n_rollouts": 10, "score_s": 5.0}
    assert gp.throughput_of(rec) == 123.0


def test_throughput_of_derives_from_score_time() -> None:
    # 20 rollouts scored in 4 s -> 5 rollouts/s
    rec = {"workers": 2, "n_rollouts": 20, "score_s": 4.0}
    assert gp.throughput_of(rec) == pytest.approx(5.0)


def test_throughput_of_returns_none_when_unavailable() -> None:
    assert gp.throughput_of({"workers": 1}) is None
    # non-positive throughput is treated as missing
    assert gp.throughput_of({"workers": 1, "throughput": 0.0}) is None
    assert gp.throughput_of({"workers": 1, "n_rollouts": 5, "score_s": 0.0}) is None


def test_speedup_series_hand_values() -> None:
    # baseline T_1 = 10; series should normalise to itself / 10
    workers = [1, 2, 4]
    thr = [10.0, 18.0, 32.0]
    assert gp.speedup_series(workers, thr) == pytest.approx([1.0, 1.8, 3.2])


def test_speedup_series_uses_min_worker_as_baseline() -> None:
    # unsorted input; baseline is W=1 (throughput 10) regardless of order
    workers = [4, 1, 2]
    thr = [32.0, 10.0, 18.0]
    assert gp.speedup_series(workers, thr) == pytest.approx([3.2, 1.0, 1.8])


def test_efficiency_series_hand_values() -> None:
    # efficiency = (T_W / T_1) / (W / W_1)
    workers = [1, 2, 4]
    thr = [10.0, 18.0, 32.0]
    # speedups 1.0, 1.8, 3.2 ; divide by W -> 1.0, 0.9, 0.8
    assert gp.efficiency_series(workers, thr) == pytest.approx([1.0, 0.9, 0.8])


def test_efficiency_perfect_scaling_is_one() -> None:
    workers = [1, 2, 4, 8]
    thr = [5.0, 10.0, 20.0, 40.0]
    assert gp.efficiency_series(workers, thr) == pytest.approx([1.0, 1.0, 1.0, 1.0])


def test_ideal_linear_throughput_hand_values() -> None:
    assert gp.ideal_linear_throughput([1, 2, 4], 10.0) == pytest.approx(
        [10.0, 20.0, 40.0]
    )


def test_speedup_series_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        gp.speedup_series([], [])
    with pytest.raises(ValueError):
        gp.speedup_series([1, 2], [10.0])  # length mismatch
    with pytest.raises(ValueError):
        gp.speedup_series([1, 2], [0.0, 5.0])  # non-positive baseline


def test_decomposition_fractions_sum_to_one() -> None:
    rec = {"workers": 1, "sample_s": 1.0, "score_s": 8.0, "learn_s": 1.0}
    fr = gp.decomposition_fractions(rec)
    assert fr is not None
    assert set(fr) == {"sample", "score", "learn"}
    assert sum(fr.values()) == pytest.approx(1.0)
    assert fr["score"] == pytest.approx(0.8)


def test_decomposition_fractions_missing_phase_is_zero() -> None:
    # only score present -> score fraction 1.0, others 0.0, still sums to 1
    rec = {"workers": 8, "score_s": 2.0}
    fr = gp.decomposition_fractions(rec)
    assert fr is not None
    assert fr["score"] == pytest.approx(1.0)
    assert fr["sample"] == 0.0 and fr["learn"] == 0.0
    assert sum(fr.values()) == pytest.approx(1.0)


def test_decomposition_fractions_none_when_no_phases() -> None:
    assert gp.decomposition_fractions({"workers": 1, "throughput": 5.0}) is None


def test_decomposition_fractions_rejects_nonpositive_total() -> None:
    with pytest.raises(ValueError):
        gp.decomposition_fractions({"workers": 1, "score_s": 0.0})


def test_sorted_by_workers_requires_workers_field() -> None:
    with pytest.raises(KeyError):
        gp._sorted_by_workers([{"throughput": 1.0}])


# --------------------------------------------------------------------------- plots write PNGs


def _records() -> list[dict]:
    # bottleneck moves off score as W grows: score fraction 0.8 -> 0.5 -> 0.25
    return [
        {"workers": 1, "throughput": 10.0,
         "sample_s": 1.0, "score_s": 8.0, "learn_s": 1.0},
        {"workers": 2, "throughput": 18.0,
         "sample_s": 1.0, "score_s": 4.0, "learn_s": 1.0},
        {"workers": 4, "throughput": 32.0,
         "sample_s": 1.0, "score_s": 2.0, "learn_s": 1.0},
    ]


def _assert_png(path: str) -> None:
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0
    with open(path, "rb") as fh:
        assert fh.read(8) == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes


def test_plot_throughput_vs_workers_writes_png(tmp_path) -> None:
    out = str(tmp_path / "throughput.png")
    ret = gp.plot_throughput_vs_workers(_records(), out)
    assert ret == out
    _assert_png(out)


def test_plot_scaling_efficiency_writes_png(tmp_path) -> None:
    out = str(tmp_path / "efficiency.png")
    ret = gp.plot_scaling_efficiency(_records(), out)
    assert ret == out
    _assert_png(out)


def test_plot_step_decomposition_writes_png(tmp_path) -> None:
    out = str(tmp_path / "decomposition.png")
    ret = gp.plot_step_decomposition(_records(), out)
    assert ret == out
    _assert_png(out)


def test_plots_are_opt_in_skip_unsupported(tmp_path) -> None:
    # records with NO throughput and NO phase timings -> each plot raises
    # ValueError (the CLI catches these to skip; here we assert the contract).
    bare = [{"workers": 1}, {"workers": 2}]
    with pytest.raises(ValueError):
        gp.plot_throughput_vs_workers(bare, str(tmp_path / "a.png"))
    with pytest.raises(ValueError):
        gp.plot_scaling_efficiency(bare, str(tmp_path / "b.png"))
    with pytest.raises(ValueError):
        gp.plot_step_decomposition(bare, str(tmp_path / "c.png"))


def test_decomposition_plots_only_present_columns(tmp_path) -> None:
    # mixed: one record has timings, one only throughput. Decomposition must
    # render using just the timed record (opt-in on present columns).
    mixed = [
        {"workers": 1, "throughput": 10.0},                 # no phase timings
        {"workers": 2, "sample_s": 1.0, "score_s": 1.0, "learn_s": 2.0},
    ]
    out = str(tmp_path / "mixed.png")
    ret = gp.plot_step_decomposition(mixed, out)
    assert ret == out
    _assert_png(out)


def test_main_cli_writes_present_plots(tmp_path) -> None:
    import json

    recs_path = tmp_path / "recs.json"
    recs_path.write_text(json.dumps(_records()), encoding="utf-8")
    outdir = tmp_path / "figs"
    rc = gp.main([str(recs_path), "--outdir", str(outdir), "--prefix", "t_"])
    assert rc == 0
    _assert_png(str(outdir / "t_throughput.png"))
    _assert_png(str(outdir / "t_efficiency.png"))
    _assert_png(str(outdir / "t_decomposition.png"))
