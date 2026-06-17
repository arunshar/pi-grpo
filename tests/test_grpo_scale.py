"""Unit tests for the reward-actor scaling sweep helpers (scripts/grpo_scale.py).

Pure CPU, no Ray and no torch: every test exercises the pure split/merge/aggregate
helpers with hand-built stub step records, never a live sweep. The Ray entrypoint
is reached only through ``run_sweep`` / ``_run_one`` which these tests do not call.

Run on a compute node with:
    PYTHONPATH=. pytest -q tests/test_grpo_scale.py
"""

from __future__ import annotations

import csv
import math

import pytest

# scripts/ is not a package; import by path-insertion so the test is runnable
# both from the repo root (PYTHONPATH=.) and standalone.
import importlib.util
from pathlib import Path

_SCALE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "grpo_scale.py"
_spec = importlib.util.spec_from_file_location("grpo_scale", _SCALE_PATH)
assert _spec is not None and _spec.loader is not None
gs = importlib.util.module_from_spec(_spec)
# Register in sys.modules BEFORE exec_module so the frozen-dataclass forward-ref
# (a default_factory under `from __future__ import annotations`) resolves; without
# it dataclasses._process_class hits sys.modules.get(name) is None -> AttributeError
# at collection time.
import sys
sys.modules[_spec.name] = gs
_spec.loader.exec_module(gs)


# --------------------------------------------------------------------------- stubs


def _rec(sample_s: float, reward_s: float, learn_s: float, **extra: float) -> dict[str, float]:
    base = {
        "sample_s": sample_s,
        "reward_s": reward_s,
        "learn_s": learn_s,
        "n_traj": extra.get("n_traj", 64.0),
        "reward_mean": extra.get("reward_mean", 0.0),
        "loss": extra.get("loss", 0.0),
        "kl": extra.get("kl", 0.0),
    }
    return base


# --------------------------------------------------------------------------- n_trajectories / throughput


def test_n_trajectories_is_b_k_steps() -> None:
    # 5 steps * 8 prompts * 8 rollouts = 320
    assert gs.n_trajectories(steps=5, batch_prompts=8, group_size=8) == 320


def test_trajectories_per_sec_hand_value() -> None:
    # 320 trajectories in 2.0 s -> 160 traj/s
    assert gs.trajectories_per_sec(320, 2.0) == pytest.approx(160.0)


def test_trajectories_per_sec_guards_zero_and_negative() -> None:
    assert math.isnan(gs.trajectories_per_sec(320, 0.0))
    assert math.isnan(gs.trajectories_per_sec(320, -1.0))
    assert math.isnan(gs.trajectories_per_sec(0, 2.0))


def test_step_total_seconds_sums_parts() -> None:
    assert gs.step_total_seconds(_rec(0.1, 0.2, 0.3)) == pytest.approx(0.6)


def test_step_total_seconds_nan_tolerant() -> None:
    # a missing reward time should not nuke the whole step total
    assert gs.step_total_seconds(_rec(0.1, float("nan"), 0.3)) == pytest.approx(0.4)


# --------------------------------------------------------------------------- extract_step_records


def test_extract_from_history_attribute() -> None:
    class _Result:
        history = [{"sample_s": 0.1, "reward_s": 0.5, "learn_s": 0.2, "n_traj": 64}]

    recs = gs.extract_step_records(_Result())
    assert len(recs) == 1
    assert recs[0]["reward_s"] == pytest.approx(0.5)
    assert recs[0]["n_traj"] == pytest.approx(64.0)


def test_extract_from_bare_list_fills_missing_keys_with_nan() -> None:
    recs = gs.extract_step_records([{"reward_s": 0.5}])
    assert recs[0]["reward_s"] == pytest.approx(0.5)
    # sample_s / learn_s were absent -> nan, not a crash
    assert math.isnan(recs[0]["sample_s"])
    assert math.isnan(recs[0]["learn_s"])


def test_extract_rejects_non_list() -> None:
    with pytest.raises(TypeError):
        gs.extract_step_records(object())


def test_extract_rejects_non_dict_record() -> None:
    with pytest.raises(TypeError):
        gs.extract_step_records([1, 2, 3])


# --------------------------------------------------------------------------- decomp rows


def test_build_decomp_rows_columns_and_step_index() -> None:
    records = [_rec(0.1, 0.2, 0.3), _rec(0.15, 0.25, 0.35)]
    rows = gs.build_decomp_rows(num_reward_actors=4, records=records)
    assert len(rows) == 2
    # exact column set matches the declared schema
    for row in rows:
        assert set(row.keys()) == set(gs.DECOMP_COLUMNS)
    assert [r["step"] for r in rows] == [0, 1]
    assert all(r["num_reward_actors"] == 4 for r in rows)
    assert rows[1]["step_s"] == pytest.approx(0.75)


# --------------------------------------------------------------------------- scaling row


def test_build_scaling_row_columns_and_throughput() -> None:
    cfg = gs.SweepConfig(steps=2, batch_prompts=4, group_size=8, horizon=12)
    # 2 steps, reward_s 0.5 + 0.5 = 1.0 s; n_traj = 2*4*8 = 64 -> 64 traj/s
    records = [_rec(0.1, 0.5, 0.2), _rec(0.1, 0.5, 0.2)]
    row = gs.build_scaling_row(num_reward_actors=2, cfg=cfg, records=records)
    assert set(row.keys()) == set(gs.SCALING_COLUMNS)
    assert row["n_trajectories"] == 64
    assert row["reward_eval_seconds"] == pytest.approx(1.0)
    assert row["throughput_traj_per_s"] == pytest.approx(64.0)
    assert row["mean_reward_s"] == pytest.approx(0.5)
    assert row["mean_step_s"] == pytest.approx(0.8)
    # speedup not yet filled at this stage
    assert math.isnan(row["speedup_vs_w1"])


# --------------------------------------------------------------------------- speedup


def test_attach_speedups_relative_to_w1() -> None:
    # Hand-constructed throughputs: W=1 -> 100, W=2 -> 180, W=4 -> 320.
    rows = [
        {"num_reward_actors": 1, "throughput_traj_per_s": 100.0},
        {"num_reward_actors": 2, "throughput_traj_per_s": 180.0},
        {"num_reward_actors": 4, "throughput_traj_per_s": 320.0},
    ]
    out = gs.attach_speedups(rows)
    by_w = {r["num_reward_actors"]: r["speedup_vs_w1"] for r in out}
    assert by_w[1] == pytest.approx(1.0)
    assert by_w[2] == pytest.approx(1.8)
    assert by_w[4] == pytest.approx(3.2)
    # input not mutated
    assert "speedup_vs_w1" not in rows[0]


def test_attach_speedups_nan_when_no_w1_baseline() -> None:
    rows = [
        {"num_reward_actors": 2, "throughput_traj_per_s": 180.0},
        {"num_reward_actors": 4, "throughput_traj_per_s": 320.0},
    ]
    out = gs.attach_speedups(rows)
    assert all(math.isnan(r["speedup_vs_w1"]) for r in out)


def test_attach_speedups_nan_when_baseline_throughput_is_nan() -> None:
    rows = [
        {"num_reward_actors": 1, "throughput_traj_per_s": float("nan")},
        {"num_reward_actors": 2, "throughput_traj_per_s": 180.0},
    ]
    out = gs.attach_speedups(rows)
    assert all(math.isnan(r["speedup_vs_w1"]) for r in out)


# --------------------------------------------------------------------------- CSV writing


def test_write_csv_header_and_rows(tmp_path) -> None:
    cfg = gs.SweepConfig(steps=1, batch_prompts=2, group_size=2, horizon=4)
    rows = [gs.build_scaling_row(num_reward_actors=1, cfg=cfg, records=[_rec(0.1, 0.1, 0.1)])]
    rows = gs.attach_speedups(rows)
    path = tmp_path / "scaling_results.csv"
    gs.write_csv(path, gs.SCALING_COLUMNS, rows)

    with path.open() as fh:
        reader = csv.DictReader(fh)
        assert tuple(reader.fieldnames) == gs.SCALING_COLUMNS
        read_rows = list(reader)
    assert len(read_rows) == 1
    assert int(read_rows[0]["num_reward_actors"]) == 1
    assert int(read_rows[0]["n_trajectories"]) == 4


def test_write_csv_decomp_roundtrip(tmp_path) -> None:
    rows = gs.build_decomp_rows(8, [_rec(0.1, 0.2, 0.3)])
    path = tmp_path / "step_decomposition.csv"
    gs.write_csv(path, gs.DECOMP_COLUMNS, rows)
    with path.open() as fh:
        reader = csv.DictReader(fh)
        assert tuple(reader.fieldnames) == gs.DECOMP_COLUMNS
        read_rows = list(reader)
    assert int(read_rows[0]["num_reward_actors"]) == 8
    assert int(read_rows[0]["step"]) == 0


# --------------------------------------------------------------------------- arg parsing


def test_arg_parser_defaults() -> None:
    args = gs.build_arg_parser().parse_args([])
    assert args.actor_counts == "1,2,4,8,16"
    assert args.steps == 5
    assert args.batch_prompts == 8
    assert args.group_size == 8
    assert args.horizon == 12
    assert args.seed == 42
    assert args.out_dir == "scaling_out"


def test_config_from_args_defaults_match_sweep_default() -> None:
    args = gs.build_arg_parser().parse_args([])
    cfg = gs.config_from_args(args)
    assert cfg.actor_counts == gs.DEFAULT_ACTOR_COUNTS
    assert cfg.steps == 5
    assert cfg.batch_prompts == 8


def test_parse_actor_counts_hand_values() -> None:
    assert gs.parse_actor_counts("1,2,4") == (1, 2, 4)
    assert gs.parse_actor_counts(" 8 , 16 ") == (8, 16)


def test_parse_actor_counts_rejects_nonpositive_and_empty() -> None:
    with pytest.raises(ValueError):
        gs.parse_actor_counts("0,2")
    with pytest.raises(ValueError):
        gs.parse_actor_counts("-4")
    with pytest.raises(ValueError):
        gs.parse_actor_counts(" , ")


def test_arg_parser_overrides() -> None:
    args = gs.build_arg_parser().parse_args(
        ["--actor-counts", "1,2", "--steps", "3", "--batch-prompts", "4",
         "--group-size", "2", "--horizon", "6", "--seed", "7", "--out-dir", "/tmp/x"]
    )
    cfg = gs.config_from_args(args)
    assert cfg.actor_counts == (1, 2)
    assert cfg.steps == 3
    assert cfg.group_size == 2
    assert str(cfg.out_dir) == "/tmp/x"


# --------------------------------------------------------------------------- summary


def test_format_summary_lists_every_w() -> None:
    rows = [
        {"num_reward_actors": 1, "throughput_traj_per_s": 100.0, "speedup_vs_w1": 1.0, "mean_step_s": 0.5},
        {"num_reward_actors": 2, "throughput_traj_per_s": 180.0, "speedup_vs_w1": 1.8, "mean_step_s": 0.3},
    ]
    text = gs.format_summary(rows)
    assert "reward-actor scaling sweep" in text
    # one data line per W
    assert text.count("\n") >= 4
    assert "100.00" in text
    assert "1.80" in text
