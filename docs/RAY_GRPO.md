# Ray GRPO: Distributing the Physics-Aware GRPO Loop

This document is the runbook for the **opt-in** Ray backend for GRPO. It is
additive: the pure-torch `train_grpo` in `app/policy/driver.py` is untouched and
remains the default. The Ray path is a new entrypoint, `train_grpo_ray`, that
reuses the existing `GrpoTrainer`, `CausalPolicy`/`build_policy_pair`,
`MotionCodebook`, `PhysicsReward`, and `PiDpmScorer` without modification.

> Status: design + scaffolding. The reward-actor pool is the first piece to land
> because it targets the measured bottleneck. The parallel-rollout and DDP-learner
> pieces are staged behind it and carry explicit TODOs in the code. Do not read any
> throughput claim below as measured until the validation table in
> `docs/RAY_GRPO_RESULTS.md` (created by the experiment run) is filled in.


## 1. Where the time goes (why Ray at all)

Read `app/policy/driver.py::train_grpo`. One optimization step does three things:

1. **SAMPLE.** `policy.generate(prompts, k=group_size, ...)` produces `(B, K, T_r)`
   rollout tokens. One batched forward-sampling call.
2. **SCORE.** `_reward_matrix(rollouts, codebook, reward, scorer)` walks a **serial
   nested `B * K` Python loop**, calling `_trajectory_reward` per rollout:
   `MotionCodebook.tokens_to_states` -> `PhysicsReward.score(states,
   pi_dpm_log_prob=PiDpmScorer.log_prob(states))`. This is the bottleneck. It is
   pure-CPU, embarrassingly parallel across the `B * K` axis, and grows with both
   group size `K` and prompt batch `B`.
3. **LEARN.** `GrpoTrainer.step_update(GrpoBatch)` does one clipped-surrogate +
   KL backward/optimizer step on the GPU (or CPU for the tiny model).

The score step is the clean parallelization target: it is the only one that is both
serial today and CPU-bound. Sampling and learning are already batched tensor ops.


## 2. Architecture

Three layers, each independently switchable. Only the reward-actor pool is on the
critical path for the headline result; the other two are opt-in scaling knobs.

```
  prompts (B, T_p)
       |
       v
  [ optional: parallel rollout actors ]      <-- layer B (opt-in)
       |   each actor holds a policy replica, generates a shard of (B,K,T_r)
       v
  rollouts (B, K, T_r)  --- split on the B*K axis --->  RayRewardPool
       |                                                  |  layer A (headline)
       |                                  N reward actors, each owns its own
       |                                  MotionCodebook + PhysicsReward +
       |                                  PiDpmScorer; scores a shard serially
       |                                  with the SAME _trajectory_reward.
       v                                                  |
  rew (B,K), viol (B,K)  <----------- merge (reassemble in original index order)
       |
       v
  [ DDP learner ]                            <-- layer C (opt-in)
       GrpoTrainer.step_update(GrpoBatch); gradients all-reduced across ranks
```

### Layer A: reward-actor pool (`RayRewardPool`) -- the headline win

`RayRewardPool` is a pool of `N` long-lived Ray actors. Each actor constructs its
own reward path **once** (`MotionCodebook`, `PhysicsReward`, `PiDpmScorer`) so the
fixed setup cost is amortized across steps, never per-rollout. Per step the driver:

1. Flattens the `(B, K, T_r)` rollouts to a list of `B * K` token arrays, recording
   each item's `(i, j)` position.
2. Splits that flat list into `N` contiguous shards (round-robin or chunked).
3. Dispatches one `.remote` call per shard; each actor runs the **identical**
   `_trajectory_reward` over its shard, returning `(total, violation)` per item.
4. `ray.get`s all shards and **merges** them back into `rew (B, K)` and
   `viol (B, K)` at the original `(i, j)` indices.

The merge restores exact ordering, so the assembled matrices are independent of `N`
and of actor completion order. See the correctness gate below.

### Layer B: parallel rollout actors (opt-in)

Each rollout actor holds a replica of the current policy weights and generates a
shard of the `K` rollouts (or a shard of prompts). Throughput goes up; the cost is a
weight-broadcast each step and the risk of **staleness** if actors run ahead of the
learner (Section 5). For the tiny CPU model this layer is expected to be
**overhead-bound** and may show a flat or negative curve; that is an honest expected
result, not a bug.

### Layer C: DDP learner (opt-in)

`GrpoTrainer.step_update` runs under `torch.distributed` / Ray Train with gradients
all-reduced across data-parallel ranks. Only relevant once the model is large enough
that the backward pass dominates. For the tiny model it is pure overhead.


## 3. Three backends (same `train_grpo_ray`, same as PC-RF)

This mirrors the PC-RF scaling story exactly: one code path, three execution
substrates. The backend is selected by how `ray.init` / the Ray cluster is reached,
not by changing `train_grpo_ray`.

| Backend | How Ray is reached | Use | Notes |
|---|---|---|---|
| **Laptop (Ray-local)** | `ray.init(num_cpus=...)` in-process | dev, the CI mock/real-Ray tests | tiny model, `N` small; reward-pool parallelism is real even here on a multi-core box |
| **MSI (Ray-on-Slurm)** | start a head on the batch node, workers via `srun`, then `ray.init(address="auto")` | the real CPU scaling sweep for the reward pool, and any GPU learner run | login-node gotcha below; this is where the headline reward-eval curve is measured |
| **Anyscale** | `ray.init(address="anyscale://...")` / job submit | elastic scale-out, cost-per-unit projection | deferred behind MSI numbers (billing), same as PC-RF |

**MSI login-node gotcha (carried from PC-RF):** Ray sizes itself to the full visible
CPU count and spawns a worker swarm; on an MSI login node that swarm crawls under the
CPU-time cap. **Validate and sweep Ray on a COMPUTE node only.** Imports are fine on
the login node; actor execution is not. Pin `num_cpus` explicitly and never let a Ray
driver autoscale on a shared login node.

Environment: Ray is targeted at the **2.x** line. It is already declared in
`pyproject.toml` (`ray[default]>=2.34`); the Ray tests additionally exercise the
actor API, so the test environment must have Ray installed. **`local_mode` was
removed in Ray >= 2.40 and must not be used**; tests that need a live cluster use
`ray.init(num_cpus=2, include_dashboard=False)` and `ray.shutdown()` in a fixture,
and everything else mocks the actor `.remote` to a local call.


## 4. Convergence-parity plan (RL is noisier than supervised loss)

The first claim a skeptical reviewer makes is "your faster path changed the answer."
For a supervised loss you would overlay one loss curve. RL reward is **noisier**:
the reward is a function of sampled rollouts, so the per-step `reward_mean` has
sampling variance even at fixed weights. A single seed proves nothing. The plan:

1. **Bit-identical reward gate (per step, not just per run).** Because
   `PhysicsReward` is deterministic given states, the `RayRewardPool` output **must
   be bit-identical** to the serial `_reward_matrix` on the **same** rollouts, for
   any actor count `N`. This is the strongest possible parity check and it is an
   explicit unit test (the merge/ordering test + a same-rollouts equality test).
   It removes the reward from the list of things that could differ between backends.
2. **Multi-seed reward-trajectory overlay.** Run `train_grpo` (serial) and
   `train_grpo_ray` (pooled) across the **same set of >= 5 seeds** with identical
   `TrainConfig`. Because the reward gate is bit-identical and the optimizer is the
   same, a **fixed seed should reproduce the same trajectory** within
   floating-point tolerance (any divergence is float reduction order, not algorithm).
   Report mean +/- std of `reward_end` and of the per-step `reward_mean` band across
   seeds; the two bands must overlap.
3. **Distributional check, not point check.** Compare the seed-distributions of
   `reward_end`, final `kl`, and `violation` rate with a paired test across seeds
   (same seed = paired sample). Parity = no significant difference, with the bands
   plotted, not just a p-value.
4. **Determinism caveat, stated honestly.** If layer B (parallel rollouts) or layer
   C (DDP) is enabled, exact per-step reproduction is no longer guaranteed (rollout
   sharding and gradient all-reduce change reduction order and RNG draw order). In
   that mode parity drops from "bit-identical trajectory" to "distributional parity
   across seeds." The doc and the experiment must say which mode produced which
   curve.


## 5. Async / staleness experiment design

The interesting RL-systems question: let the rollout/reward actors **run ahead** of
the learner. The learner trains on rollouts generated by a slightly older policy
(off-policy lag of `s` steps). Hypothesis: **throughput goes up** (actors never idle
waiting for the optimizer step) but **off-policy KL / staleness goes up** (the
importance ratio in the clipped surrogate drifts, the group baseline ages).

- **Knob.** `staleness s` = how many learner steps the actors may lead by
  (`s = 0` is the synchronous on-policy baseline; `s = 1, 2, 4, ...` increasingly
  async). Implemented as a bounded queue of in-flight rollout batches between the
  actor pool and the learner.
- **Throughput axis (the win).** steps/sec and rollouts/sec vs `s`. Expect to rise
  then plateau as the learner becomes the bottleneck.
- **Off-policy cost axis (the price).** Track the realized ratio
  `exp(new_logp - action_logp_old)` drift, the reference KL, and the
  `reward_mean` / `violation` rate vs `s`. Expect KL and ratio variance to grow with
  `s`; at some `s*` the clipped surrogate clips so hard that effective learning
  stalls or reward regresses. Report the `s*` knee.
- **Reuse, do not fork.** `GrpoBatch.action_logp_old` already records the
  behavior-policy log-prob, which is exactly the quantity that makes the off-policy
  correction visible. The staleness experiment reads it; it does not need a new
  field.
- **The honest framing.** This is a throughput-vs-off-policy-KL **trade-off curve**,
  not a free win. The deliverable is the curve and the knee, not a single "async is
  faster" number.


## 6. Honesty about the tiny model (what will and will not move)

The default `TrainConfig` is a tiny model that trains on CPU in seconds. On that
model:

- **Rollout (layer B) and learner (layer C) curves may be FLAT or negative.** The
  per-step compute is tiny, so Ray actor dispatch, serialization, and weight
  broadcast dominate. This is **overhead-bound** and is the expected result. We will
  report it as such rather than hide it. To see those layers win you need a larger
  policy where generate/backward actually cost something.
- **The reward-eval curve (layer A) is the clean win.** The `B * K` serial Python
  loop over `_trajectory_reward` is real CPU work today and scales with `N` reward
  actors with near-linear speedup until the per-shard work no longer hides dispatch
  overhead. This is the headline scaling curve and the one that motivates the whole
  Ray path. We will show speedup vs `N` for several `(B, K)` sizes and mark where it
  saturates.
- **Therefore the paper/runbook story is reward-eval-led**, with rollout/learner
  curves shown honestly as overhead-bound at tiny scale and projected (not claimed)
  to larger models.


## 7. Entrypoints and module map

| Symbol | Where | Role |
|---|---|---|
| `train_grpo_ray` | new Ray driver module | opt-in Ray entrypoint; same `TrainConfig` / `TrainResult` contract as `train_grpo` |
| `RayRewardPool` | new Ray reward module | the `N`-actor reward pool; `score_matrix(rollouts) -> (rew, viol)` bit-identical to `_reward_matrix` |
| `_reward_matrix` / `_trajectory_reward` | `app/policy/driver.py` (unchanged) | the serial reference each actor reuses verbatim and the parity gate compares against |
| `GrpoTrainer` / `GrpoConfig` / `GrpoBatch` | `app/trainers/grpo_trainer.py` (unchanged) | the learner, reused as-is |
| `CausalPolicy` / `build_policy_pair` | `app/policy/model.py` (unchanged) | policy + frozen reference, reused as-is |

The Ray path never edits `train_grpo`, `_reward_matrix`, the trainer, the policy, or
the reward. It wraps them.


## 8. Validation and runbook commands

CPU-only, tiny, fast tests (mock the actor `.remote` to a local call where no cluster
is needed; use `ray.init(num_cpus=2, include_dashboard=False)` only where a live
actor is genuinely required):

```bash
# docs sanity (this file)
pytest tests/test_ray_grpo_docs.py -q

# reward-pool parity + ordering (added by the reward-pool agent)
pytest tests/test_ray_reward_pool.py -q

# full Ray-GRPO suite on a COMPUTE node (never the MSI login node)
pytest tests/ -k ray -q
```

MSI sweep (compute node, Ray-on-Slurm): start the Ray head on the allocated node,
join workers via `srun`, then run `train_grpo_ray` across `N = 1, 2, 4, 8` reward
actors at several `(B, K)` and collect the reward-eval speedup curve. Pin
`num_cpus`; do not autoscale on shared nodes.
