# Ray GRPO: measured scaling results

Validation table promised by `docs/RAY_GRPO.md`. Every number here is measured and
reproducible on CPU, not projected. Raw data: `paper/results/scaling_results.csv`
(the W curve), `paper/results/step_decomposition.csv` (per-step timing), and
`paper/results/crossover_bk.csv` (the B*K sweep). Reproduce with the commands at the
bottom.

## Setup

Reward-actor pool scaling on the GRPO score step, CPU only, on MSI Agate `agsmall`
(16 cores). The score step is a serial B*K reward loop over trajectories; it is the
only training stage that is both CPU-bound and embarrassingly parallel, so it is
the parallelization target. We distribute it over a Ray reward-actor pool
(`app/policy/ray_reward_pool.py`) and sweep the actor count W. The model is the tiny
policy used for the systems characterization, so per-item reward is cheap by design.

## Table R1: reward-actor pool scaling at a fixed shape (measured)

Fixed shape B=8, K=8, horizon=12, 10 steps (B*K = 64 trajectories per step).

| W (reward actors) | reward eval (s) | throughput (traj/s) | speedup vs W=1 | mean step (s) |
|---:|---:|---:|---:|---:|
| 1  | 0.278  | 2298.6 | 1.00x  | 0.097 |
| 2  | 2.786  | 229.8  | 0.100x | 0.343 |
| 4  | 6.350  | 100.8  | 0.044x | 0.701 |
| 8  | 11.680 | 54.8   | 0.024x | 1.236 |
| 16 | 21.911 | 29.2   | 0.013x | 2.256 |

At this small B*K the pool ANTI-scales: every added actor makes the run slower, and
W=16 is about 79x slower than serial. Read alone this looks like a failure. The
per-step decomposition shows it is a one-time cost, and the B*K sweep below shows it
reverses.

## Why: a one-time actor cold start, not per-step dispatch

`paper/results/step_decomposition.csv` splits each run into its steps. The reward
time of step 0 versus the warm steps tells the whole story:

| W | step 0 reward_s | warm-step reward_s (mean) |
|---:|---:|---:|
| 1  | 0.027  | 0.028 |
| 2  | 2.612  | 0.019 |
| 4  | 6.239  | 0.012 |
| 8  | 11.572 | 0.012 |
| 16 | 21.830 | 0.009 |

Two facts. First, the entire penalty is in step 0 and grows about linearly with W
(roughly 1.4 s per actor): it is the one-time cost of starting W workers and
importing torch in each forked worker. Second, the warm steps DO parallelize: the
warm reward time falls from 0.028 s (serial) to 0.009 s at W=16, a 3x speedup. The
pool works; it is the fixed cold start that is not amortized when B*K is tiny,
because a warm step at B*K = 64 is already so cheap (about 0.01 to 0.03 s) that no
realistic step count can pay back an 11 to 22 s startup.

## Table R2: the crossover vs B*K (measured)

The amortization condition is "warm per-step saving, summed over the run, exceeds
the per-actor cold start." That favors larger B*K. Holding horizon=24 and 8 steps,
sweeping B*K and reporting the best W>1 against the serial baseline:

| B*K | W=1 (traj/s) | best W>1 | best speedup | crossover? |
|---:|---:|:---:|---:|:---:|
| 16  | 1943.6 | W=2 | 0.026x | no  |
| 64  | 2078.5 | W=2 | 0.099x | no  |
| 256 | 2373.5 | W=2 | 0.49x  | no  |
| 512 | 2565.6 | W=2 | 1.12x  | YES |

The deficit shrinks monotonically with B*K and the pool CROSSES OVER at B*K = 512:
W=2 reaches 2860 traj/s versus the serial 2566 (reward eval 1.43 s vs 1.60 s),
beating the serial path despite paying the cold start. W=4 and W=8 do not yet win at
8 steps because their larger cold starts (about 4.8 s and 10.2 s) are not amortized
over so few steps; with more steps or larger B*K their crossover follows by the same
arithmetic.

## Table R3: the heavy-reward crossover (measured)

The other way to make the score step heavy is to raise the per-item reward cost
rather than B*K. The Pi-DPM scorer is the expensive term in production (a diffusion
forward pass); in the CPU proxy it is cheap, so we scale it with `reward_repeats`
(r evaluations of the deterministic score pass; the reward value is unchanged, only
the cost scales). Holding B*K = 64 fixed (horizon 12, 8 steps), the same shape where
the B*K sweep showed no crossover, and sweeping r:

| reward_repeats r | serial (traj/s) | best W>1 | best speedup |
|---:|---:|:---:|---:|
| 1   | 2087.0 | W=2 | 0.086x |
| 10  | 702.8  | W=2 | 0.256x |
| 50  | 224.4  | W=2 | 0.528x |
| 100 | 132.7  | W=2 | 0.925x |
| 200 | 69.2   | W=2 | 1.277x |

The best W>1 speedup climbs monotonically with per-item cost and the pool CROSSES
OVER at r = 200 (W=2: reward eval 5.79 s vs the serial 7.40 s). So heavy per-item
reward induces the crossover even at the small B*K where a cheap reward anti-scales.
Data: `paper/results/heavy_reward_bk64.csv`.

## Conclusion (the contribution)

The reward-actor pool is a net loss when the score step is light and a net win once
it is heavy enough that the warm per-step parallel saving, summed over the run,
exceeds the fixed per-actor cold start. "Heavy" has two independent axes and we
locate the crossover on both: by trajectory count, B*K = 512 (W=2); and by per-item
reward cost, r = 200 at B*K = 64 (W=2). The deliverable is not "Ray makes GRPO
faster" but the characterization: the pool helps exactly when the score step is
heavy, and the crossover point on either axis is set by the ratio of warm per-step
saving to per-actor startup cost. A production reward with a real (expensive) Pi-DPM
diffusion scorer sits in the heavy regime, so the pool is expected to pay off there
without the synthetic cost knob.

## Reproduce

```
# Table R1 (the W curve at fixed shape):
PIGRPO_ACTOR_COUNTS=1,2,4,8,16 PIGRPO_STEPS=10 PIGRPO_BATCH_PROMPTS=8 PIGRPO_GROUP_SIZE=8 PIGRPO_HORIZON=12 \
  PIGRPO_PYTHON=/users/1/arunshar/miniforge3/envs/pcrf/bin/python PIGRPO_REPO=$PWD \
  sbatch --account=shekhars --export=ALL scripts/grpo_ray_slurm.sbatch

# Table R2 (the B*K crossover sweep), one job per B*K point:
for BK in "4 4" "8 8" "16 16" "32 16"; do set -- $BK
  PIGRPO_ACTOR_COUNTS=1,2,4,8 PIGRPO_STEPS=8 PIGRPO_BATCH_PROMPTS=$1 PIGRPO_GROUP_SIZE=$2 PIGRPO_HORIZON=24 \
    PIGRPO_PYTHON=/users/1/arunshar/miniforge3/envs/pcrf/bin/python PIGRPO_REPO=$PWD \
    sbatch --account=shekhars --export=ALL scripts/grpo_ray_slurm.sbatch
done
# Each job writes scaling_results.csv + step_decomposition.csv under
# /scratch.global/$USER/pigrpo/<jobid>/. NOTE: pass env as a prefix (exported),
# NOT inside --export=..., because --export is itself comma-separated and would
# split a value like 1,2,4,8.
```
