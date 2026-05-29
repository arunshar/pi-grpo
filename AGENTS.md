# Agents

Pi-GRPO has four agents that orchestrate a training run. Each agent is a separate Python process with a typed RPC interface; we use this structure so a long PPO / GRPO run can survive the death of any one component.

## DataCuratorAgent (`app/agents/data_curator.py`)

| Field | Value |
|---|---|
| capability | `prefs.build`, `prefs.audit` |
| inputs | `HITLSource`, `pref_filter` |
| outputs | versioned `data/preferences/<algo>/v<n>.jsonl` |

Pulls HITL verdicts from the sibling `geotrace-agent` project, joins them against the original trace's regions and Pi-DPM scores, builds `(prompt, chosen, rejected)` triples, and audits them for label leakage and class imbalance.

## TrainerAgent (`app/agents/trainer_agent.py`)

| Field | Value |
|---|---|
| capability | `train.ppo`, `train.dpo`, `train.grpo` |
| inputs | `RunConfig` |
| outputs | run artifacts under `runs/<run_id>/` |

Submits a run on the local machine, on a Ray cluster, or on Kubernetes (via the configured launcher). Owns the adaptive KL controller, gradient clipping, and the checkpoint scheduler.

## EvaluatorAgent (`app/agents/evaluator.py`)

| Field | Value |
|---|---|
| capability | `eval.offline`, `eval.online` |
| inputs | `RunArtifacts`, `eval_config` |
| outputs | reports + W&B run; HITL labels for the `golden_dataset` slice |

Runs golden-dataset evaluation (validity, fluency, physics-pass rate) on every checkpoint and shadow-evaluates the production policy on 1 percent of online traffic.

## CoordinatorAgent (the user-visible orchestrator)

| Field | Value |
|---|---|
| capability | `run.submit`, `run.status`, `run.cancel` |
| inputs | `RunRequest` |
| outputs | `RunResult` |

Talks to the three agents above over JSON-RPC. The FastAPI endpoint `POST /v1/runs` lands here.

## Reward model

`app/reward_models/physics_reward_model.py` is not a separate agent; it is a Python module that:

- exposes `score(traj, prompt) -> RewardBreakdown`
- combines a hard physics floor (S-KBM violations), a soft physics floor (jerk / curvature percentiles), a Pi-DPM reconstruction-error term, and an optional preference-classifier term.

`RewardBreakdown` is structured so each term shows up in W&B as a separate panel and so the trainer can flag dominance of any single term (a known failure mode).
