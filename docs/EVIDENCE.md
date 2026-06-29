# pi-grpo — evidence

_Test inventory generated 2026-06-29T10:37:18Z. The RL stack runs locally with the repo deps (PyTorch + Ray); the suite below was not re-run in this environment. The headline ~18% -> ~0% hard-constraint-violation reduction is the workshop paper's reported result ("Physics as a Hard Reward Floor", MOSS @ COLM 2026), not re-run at scale here._

## Test suite
```
__init__.py
test_api_integration.py
test_e2e_grpo_ray.py
test_grpo_advantage.py
test_grpo_plots.py
test_grpo_scale.py
test_kinematic_bicycle.py
test_kl_controller.py
test_physics_reward.py
test_pidpm.py
test_policy.py
test_ray_driver.py
test_ray_grpo_docs.py
test_ray_reward_pool.py
test_ray_rollout.py
test_ray_staleness.py
test_reward_properties.py
test_reward_sanity.py
```

These tests cover the physics reward, GRPO advantage, KL controller, kinematic-bicycle prior, reward properties/sanity, the Ray rollout/driver/reward-pool, and end-to-end GRPO, the runnable parts behind the paper's claims.

## Reproduced on MSI (2026-06-29T17:37:16Z)

Ran `scripts/verify_reward_hacking_mechanism.py` on MSI (Agate). The physics reward floor is reproduced at the reward level: against a hacked +10 preference signal that "likes" infeasible (3x-over-speed) trajectories,

- preference-only (w_hard=0): hard-violation catch rate = **0%** (0/5 rejected) - reward hacking succeeds.
- physics-grounded (w_hard=5, default): catch rate = **100%** (5/5 rejected) - the unbounded hard S-KBM term overrides the hacked preference.

This reproduces the *mechanism* behind the paper's reward-hacking claim. It does NOT reproduce the end-to-end trained-policy 18%->0% number: per the script's own note, that needs a DPO-trained policy with a `log_prob_seq` method that is not in this repo, so it stays the paper's reported result, not reproduced here.
