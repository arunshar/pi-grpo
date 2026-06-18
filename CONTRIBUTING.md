# Contributing to Pi-GRPO

This is a research-engineering project for physics-informed RL. Contributions in any of the following areas are welcome:

- Reward terms (additions / refinements to `app/components/physics_reward.py`).
- New trainers under `app/trainers/`.
- Rollout backends in `app/rollouts/`.
- Preference-data tooling in `app/components/preference_builder.py`.
- Evaluation, observability, or documentation.

## Before you open a PR

```bash
ruff check .
mypy app
pytest -q
```

If you change the reward, run `tests/test_reward_sanity.py` and confirm it passes; the trainer refuses to start if the sanity test fails.

## Before pushing

Run the local CI gate before every push. It mirrors `.github/workflows/ci.yml`
(the `lint-type-test` job) exactly: the same `ruff check`, `mypy app || true`,
and `pytest -q -p no:warnings` commands, in the `pcrf` conda env.

```bash
make ci          # or: bash scripts/ci_local.sh
```

The gate prints `CI-LOCAL: PASS` (exit 0) or `CI-LOCAL: FAIL (<reason>)` (exit
non-zero). `ruff` and `pytest` are hard gates; `mypy` is non-blocking, exactly
as in CI. If some tests cannot be collected here because of a missing optional
dependency (for example `redis` or `rtree`), the gate prints an `ENV-GAP`
warning, runs the collectable subset, and still hard-fails on any real test
failure (it never reports a gap as a pass).

On a fresh clone, enable the pre-push hook once so the gate runs automatically
and blocks a failing push:

```bash
make hooks       # or: git config core.hooksPath .githooks
```

`core.hooksPath` is a local git config (not committed), so each clone opts in.
To bypass the hook in an emergency: `git push --no-verify` (not recommended).

## Hard invariants

- Reference model is frozen for the entirety of a run; verified by SHA at run start and end.
- KL is computed token-wise against the reference, not against the previous policy.
- The `AdaptiveKLController` is bounded `[clip_min, clip_max]`. Never disable bounds.
- Hard physics violations dominate any preference signal; verified by `tests/test_physics_reward.py`.
- Checkpoints are content-addressed: `runs/<run_id>/step_<n>/<sha>.bin` with `MANIFEST.jsonl`.

## Adding a trainer flag

A new flag must come with: a row in `configs/safe_ranges.yaml`, a sanity test, and a docs/training.md entry under "failure modes".
