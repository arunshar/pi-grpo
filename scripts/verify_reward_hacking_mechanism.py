"""Reward-level demonstration of the anti-reward-hacking mechanism.

HONEST SCOPE: this exercises the repo's real PhysicsReward to show the design
property that underlies the paper's reward-hacking claim, namely that the
unbounded hard S-KBM term overrides a high (hacked) preference signal. It does
NOT reproduce the paper's end-to-end 18%->0% trained-policy violation rate: that
requires a DPO-trained policy with a `log_prob_seq` method, which does not exist
in the repository, so that number is not reproducible from this code.

We build feasible and physically-infeasible (speeding) trajectories, attach a
large pref_logit to all of them (a preference model that has been hacked into
liking the infeasible ones), and compare two reward regimes:

  preference-only (w_hard = 0): the hacked preference wins -> infeasible passes
  physics-grounded (w_hard = 5, the repo default): the hard term dominates ->
      infeasible is caught regardless of the preference signal
"""
from __future__ import annotations

import numpy as np

from app.components.kinematic_bicycle import SkbmConfig
from app.components.physics_reward import EmpiricalEnvelope, PhysicsReward, RewardWeights

cfg = SkbmConfig()
v_max = cfg.v_max_mps


def traj(speed: float, n: int = 50) -> np.ndarray:
    s = np.zeros((n, 4))
    s[:, 0] = np.arange(n) * speed  # x advances at `speed`
    s[:, 3] = speed                 # constant velocity
    return s


feasible = [traj(0.5 * v_max) for _ in range(5)]      # well under the cap
infeasible = [traj(3.0 * v_max) for _ in range(5)]     # 3x over the speed cap
HACKED_PREF = 10.0                                     # preference model "likes" all of them

regimes = {
    "preference-only (w_hard=0)": RewardWeights(hard=0.0, soft=0.0, data=0.0, pref=1.0),
    "physics-grounded (w_hard=5, default)": RewardWeights(),  # hard=5,soft=1,data=1,pref=1
}

print(f"S-KBM speed cap v_max = {v_max:.1f} m/s | feasible speed = {0.5*v_max:.1f} | "
      f"infeasible speed = {3.0*v_max:.1f} | hacked pref_logit = +{HACKED_PREF}")
print()
for name, w in regimes.items():
    reward = PhysicsReward(cfg, weights=w, envelope=EmpiricalEnvelope())
    inf_totals = [reward.score(t, pref_logit=HACKED_PREF).total for t in infeasible]
    fea_totals = [reward.score(t, pref_logit=HACKED_PREF).total for t in feasible]
    inf_hard = [reward.score(t, pref_logit=HACKED_PREF).hard for t in infeasible]
    # "caught" = the reward rejects the infeasible trajectory (total < 0)
    caught = sum(1 for x in inf_totals if x < 0)
    catch_rate = caught / len(inf_totals)
    print(f"[{name}]")
    print(f"  infeasible mean total = {np.mean(inf_totals):+8.3f}  "
          f"(hard mean = {np.mean(inf_hard):+8.3f})")
    print(f"  feasible   mean total = {np.mean(fea_totals):+8.3f}")
    print(f"  hard-violation catch rate on infeasible set = {catch_rate:.0%}  "
          f"({caught}/{len(inf_totals)} rejected despite hacked +{HACKED_PREF} preference)")
    print()
