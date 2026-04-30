# Pi-GRPO: Architecture

## Why three trainers in one stack

| | PPO | DPO | GRPO |
|---|---|---|---|
| Needs reward model | yes | no | yes (or critic-free reward) |
| Needs value head | yes | no | no |
| Needs preferences | optional | required | optional |
| Stability | medium (KL controller, value head) | high | medium (group baseline) |
| Best for | online improvement when a fresh reward model exists | offline tuning from human preferences | reasoning policies with sparse rewards |

The reward path is shared (`app/components/physics_reward.py`) so all three trainers see the same signal. Switching between them is a config flag, not a rewrite.

## Pipeline

```
   build preferences (HITL or synthetic)         data/preferences/v<n>.jsonl
        │
        ▼
   POST /v1/runs ──▶ RunOrchestrator ──▶ TrainerAgent ──▶ {ppo|dpo|grpo}_trainer.py
                              │                                │
                              ▼                                ▼
                       Postgres `runs`                    PhysicsReward
                       W&B run                              ├ S-KBM hard floor
                       OTEL spans                           ├ soft envelope
                                                            ├ Pi-DPM scorer
                                                            └ pref classifier
                              ▼
                       checkpoints/<run_id>/step_<n>/<sha>.bin
                              │
                              ▼
                       Evaluator ──▶ evaluation/eval_results/<ts>.json
```

## Reward shape

```
R(traj) = w_hard * R_hard
        + w_soft * R_soft
        + w_data * R_data
        + w_pref * R_pref
```

- `R_hard` is the sum of relative excess over S-KBM bounds. Unbounded above so any single hard violation dominates.
- `R_soft` penalizes the 95th-percentile curvature and jerk relative to the empirical envelope fit on Porto / Harbin / MarineCadastre AIS.
- `R_data` is the Pi-DPM reconstruction-error-tail surrogate (negative log-likelihood under the diffusion prior).
- `R_pref` is the cross-encoder pref-classifier output, used only when configured.

## Why GRPO is the headliner for reasoning

Reasoning policies often face sparse, structured rewards: "did the chain end in a correct verdict?" PPO needs a value head and a non-trivial GAE budget for those. GRPO sidesteps both by sampling K rollouts per prompt and normalizing the advantage within the group. The trainer needs only a reference model (frozen base) plus a reward function. This matches DeepSeek-R1's recipe and lets us reuse our PhysicsReward unmodified.

## Inference path

`POST /v1/infer` proxies to a vLLM container with `--enable-prefix-caching`, which is the difference between feasibility and infeasibility for online RL: prompts hit the prefix cache and the per-token cost stays approximately constant in prompt length.
