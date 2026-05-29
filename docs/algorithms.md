# Algorithms

## PPO (Schulman et al., 2017)

Clipped surrogate loss with a separate value head. KL to a frozen reference is added as a soft penalty managed by an adaptive controller (`AdaptiveKLController`). Implementation: `app/trainers/ppo_trainer.py`.

## DPO (Rafailov et al., 2023)

Direct optimization of the implicit reward `beta · log(pi(y|x) / pi_ref(y|x))` against pairwise preferences. We add a small `gamma_phys` term that injects a physics-aware penalty into the implicit reward so the policy is biased away from physics-violating outputs even when the human label did not encode that signal. Implementation: `app/trainers/dpo_trainer.py`.

## GRPO (Shao et al., 2024 / DeepSeek-R1)

For each prompt we sample `K` rollouts. The advantage of each rollout is `(R - mean_K(R)) / std_K(R)`. Loss is a clipped surrogate plus KL to the reference; no value head. Particularly effective for short-horizon physics-reasoning prompts where a critic is hard to fit. Implementation: `app/trainers/grpo_trainer.py`.

## Reward model

`PhysicsReward` is the shared reward across all three trainers:

```
R = w_hard * R_hard + w_soft * R_soft + w_data * R_data + w_pref * R_pref
```

Configured by `configs/physics_reward.yaml`. Validated by `tests/test_reward_sanity.py` before every PPO run.
