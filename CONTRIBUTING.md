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

## Hard invariants (see `CLAUDE.md`)

- Reference model is frozen for the entirety of a run; verified by SHA at run start and end.
- KL is computed token-wise against the reference, not against the previous policy.
- The `AdaptiveKLController` is bounded `[clip_min, clip_max]`. Never disable bounds.
- Hard physics violations dominate any preference signal; verified by `tests/test_physics_reward.py`.
- Checkpoints are content-addressed: `runs/<run_id>/step_<n>/<sha>.bin` with `MANIFEST.jsonl`.

## Adding a trainer flag

A new flag must come with: a row in `configs/safe_ranges.yaml`, a sanity test, and a docs/training.md entry under "failure modes".
