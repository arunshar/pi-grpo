"""Unit tests for the pure helpers in scripts/e2e_grpo_ray.py.

Pure CPU, no torch / Ray / GPU: every test exercises a side-effect-free helper
(``remap_step_record``, ``classify_scaling``, ``parse_actor_counts_arg``,
``verdict_line``) with hand-built inputs. The live sweep (``run_sweep_real`` /
``main``) touches torch and the real GRPO driver and is NOT called here; it is
covered by the sbatch end-to-end run on a compute node.

Run on a compute node (or any box with python3.11) with:
    PYTHONPATH=. pytest -q tests/test_e2e_grpo_ray.py
or standalone (no pytest):
    python3.11 tests/test_e2e_grpo_ray.py
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

# scripts/ is not a package; path-import the driver module by file location so the
# test runs both under pytest (PYTHONPATH=.) and standalone.
_E2E_PATH = Path(__file__).resolve().parent.parent / "scripts" / "e2e_grpo_ray.py"
_spec = importlib.util.spec_from_file_location("e2e_grpo_ray", _E2E_PATH)
assert _spec is not None and _spec.loader is not None
e2e = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = e2e
_spec.loader.exec_module(e2e)


# --------------------------------------------------------------------------- remap_step_record


def test_remap_translates_timing_keys() -> None:
    rec = e2e.remap_step_record(
        {"t_sample": 0.1, "t_score": 0.2, "t_learn": 0.3, "reward_mean": 1.5,
         "loss": 0.9, "kl": 0.05},
        n_traj=24,
    )
    assert rec["sample_s"] == 0.1
    assert rec["reward_s"] == 0.2
    assert rec["learn_s"] == 0.3
    assert rec["n_traj"] == 24.0
    assert rec["reward_mean"] == 1.5
    assert rec["loss"] == 0.9
    assert rec["kl"] == 0.05


def test_remap_missing_timing_is_nan_not_keyerror() -> None:
    # A partial metrics dict (only reward_mean) must still yield a total record.
    rec = e2e.remap_step_record({"reward_mean": 0.0}, n_traj=10)
    assert math.isnan(rec["sample_s"])
    assert math.isnan(rec["reward_s"])
    assert math.isnan(rec["learn_s"])
    assert math.isnan(rec["loss"])
    assert math.isnan(rec["kl"])
    assert rec["reward_mean"] == 0.0
    assert rec["n_traj"] == 10.0


# --------------------------------------------------------------------------- classify_scaling


def _row(w: int, thr: float) -> dict[str, object]:
    return {"num_reward_actors": w, "throughput_traj_per_s": thr}


def test_classify_increasing() -> None:
    rows = [_row(1, 100.0), _row(2, 150.0), _row(4, 220.0)]
    assert e2e.classify_scaling(rows) == "increasing"


def test_classify_flat_within_ten_percent() -> None:
    rows = [_row(1, 100.0), _row(2, 101.0), _row(4, 95.0)]
    assert e2e.classify_scaling(rows) == "flat"


def test_classify_regressing() -> None:
    # Top-W throughput well below the W=1 baseline (actor overhead dominates).
    rows = [_row(1, 100.0), _row(2, 70.0), _row(4, 50.0)]
    assert e2e.classify_scaling(rows) == "regressing"


def test_classify_unknown_without_baseline() -> None:
    rows = [_row(2, 100.0), _row(4, 200.0)]
    assert e2e.classify_scaling(rows) == "unknown"


def test_classify_unknown_single_point() -> None:
    assert e2e.classify_scaling([_row(1, 100.0)]) == "unknown"


def test_classify_ignores_nan_rows() -> None:
    rows = [_row(1, 100.0), _row(2, float("nan")), _row(4, 130.0)]
    # Only finite W=1 and W=4 count; 130/100 = 1.3 -> increasing.
    assert e2e.classify_scaling(rows) == "increasing"


# --------------------------------------------------------------------------- parse_actor_counts_arg


def test_parse_actor_counts_basic() -> None:
    assert e2e.parse_actor_counts_arg("1,2,4") == (1, 2, 4)


def test_parse_actor_counts_strips_and_skips_blanks() -> None:
    assert e2e.parse_actor_counts_arg(" 1 , ,2 ,4 ") == (1, 2, 4)


def test_parse_actor_counts_rejects_nonpositive() -> None:
    try:
        e2e.parse_actor_counts_arg("1,0,4")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for non-positive count")


def test_parse_actor_counts_rejects_empty() -> None:
    try:
        e2e.parse_actor_counts_arg("   ")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for empty spec")


# --------------------------------------------------------------------------- verdict_line


def test_verdict_pass_only_when_all_true() -> None:
    line = e2e.verdict_line(throughput_ok=True, reward_ok=True, csv_ok=True)
    assert line.startswith("PASS")


def test_verdict_fail_if_any_false() -> None:
    for a, b, c in [(False, True, True), (True, False, True), (True, True, False)]:
        line = e2e.verdict_line(throughput_ok=a, reward_ok=b, csv_ok=c)
        assert line.startswith("FAIL")


def test_verdict_line_is_greppable() -> None:
    line = e2e.verdict_line(throughput_ok=True, reward_ok=False, csv_ok=True)
    assert "reward_nondecreasing=False" in line
    assert "throughput_recorded=True" in line
    assert "csv_emitted=True" in line


# --------------------------------------------------------------------------- standalone runner


def _run_all() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {exc!r}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
