# Pi-DPM: Physics-informed Diffusion for Trajectory Anomaly Detection

Reference PyTorch implementation of

> Arun Sharma, Mingzhou Yang, Majid Farhadloo, Subhankar Ghosh, Bharat
> Jayaprakash, Shashi Shekhar. "Towards Physics-informed Diffusion for Anomaly
> Detection in Trajectories: A Summary of Results." 2nd ACM SIGSPATIAL
> International Workshop on Geospatial Anomaly Detection (GeoAnomalies '25),
> pp. 11-24. DOI: 10.1145/3764914.3770595

This is the real, GPU-ready model that the rest of the stack treats as the
"Pi-DPM service" (the `PiDpmScorer` reward bridge here, the gap detector in
GeoTrace-Agent, the anomaly head in DarkVesselNet). It grew out of the numpy
miniature in the `arun-papers` skill
(`code/physics-informed-diffusion-anomaly.py`), which used a polynomial fit and
analytic finite differences in place of the diffusion model and the neural
denoiser. Here both are real.

## What the paper claims, and where it lives in the code

| Paper component | Module | What it does |
|---|---|---|
| Encoder-decoder denoiser | `model.py` `TrajectoryDenoiser` | Transformer over the (T, 2) track; diffusion timestep + optional condition injected by adaLN-zero (DiT-style). Predicts the noise epsilon. |
| Diffusion probabilistic model | `diffusion.py` `GaussianDiffusion` | DDPM with cosine schedule, forward `q_sample`, eps-prediction, x0 recovery, deterministic DDIM reverse + `reconstruct`. |
| S-KBM physics constraint (Eq. 20) | `physics.py` `PhysicsResidual` | Differentiable Single-axle Kinematic Bicycle envelope (speed / accel / curvature / jerk), shared with `kinematic_bicycle.py`. Applied to the model's predicted clean sample so the physics gradient flows through the denoiser. |
| Physics-informed anomaly score (Def 2.4) | `scoring.py` `PiDPM.score` | `eps = w_rec * ||x0 - x0_hat||^2 + w_phy * R_phys(x0)`, flagged when `eps >= lambda`. |
| Reward bridge / log p(traj) | `scoring.py` `PiDPM.log_prob`, `../pidpm_scorer.py` | Negated anomaly score, consumed by the pi-grpo physics-aware reward model. |
| Training objective | `diffusion.py` `GaussianDiffusion.loss` | `L = ||eps - eps_theta||^2 + lambda_phys * R_phys(x0_hat) (+ lambda_rec * ||x0 - x0_hat||^2)`. |

## Design notes (the things that make it actually train)

- The diffusion runs in a model space `x = (x_phys - mean) / pos_scale` (~unit
  std), while the S-KBM envelope stays in physical units (m/s); `PhysicsResidual`
  de-scales by `pos_scale` before applying thresholds. This keeps the noise
  schedule sane without making the physics term meaningless.
- Envelope penalties are dimensionless (`relu(v / v_max - 1)^2`, etc.), so the
  loss is well-conditioned regardless of units.
- The physics term is SNR-weighted by `alpha_bar(t)` and applied to a clamped
  `x0_hat`, so high-noise steps (where `x0_hat` is unreliable) do not dominate.
  On the in-envelope normal manifold the term is near zero; it grows for
  envelope-violating samples, which is exactly what makes it discriminative.

## Run it

```bash
# train on the synthetic normal manifold (CPU smoke or GPU), save a checkpoint
python -m app.components.pidpm.train --epochs 15 --n 4096 --out /tmp/pidpm.pt

# evaluate anomaly detection; prints metrics MEASURED on this run
python -m app.components.pidpm.eval --ckpt /tmp/pidpm.pt --n 2048
```

The data is synthetic (`data.py`): smooth speed-bounded arcs for the normal
manifold, plus three anomaly families from the paper's case studies, teleport
(GPS spoofing), excess speed (the Cathay Phoenix circular-spoofing pattern), and
freeze (signal-denial gap). To train on real AIS, swap `TrajectoryDataset` for a
MarineCadastre / Danish-AIS loader that yields the same `(T, 2)` lon-lat windows
through `normalise()`; nothing downstream changes.

## Measured results (synthetic, honest)

A 15-epoch CPU run (`d_model=64, 3 layers, T=120 diffusion steps`, 4096 train /
2048 test, 22 s) measured on the labelled split:

| metric | value |
|---|---|
| AUROC | 0.988 |
| average precision | 0.992 |
| precision @ lambda(95%) | 0.943 |
| recall @ lambda(95%) | 0.977 |
| F1 @ lambda(95%) | 0.960 |

Per-anomaly score separation (normal `~0.23`): teleport `~40`, excess-speed
`~90`, freeze `~664`. These are synthetic-benchmark figures from `eval.py`, not
a claim about any real-world AIS leaderboard.

## Tests

`tests/test_pidpm.py` (8 cases): denoiser shapes + gradient flow, `q_sample` /
`predict_x0` roundtrip, physics flags excess speed and stays in-envelope for
plausible tracks, score separates a teleport, training reduces the loss,
checkpoint roundtrip, and the reward-bridge proxy preferring smooth tracks.
