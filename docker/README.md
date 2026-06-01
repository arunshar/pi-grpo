# Docker images

Two images, two jobs.

## Serving / training (GPU): `app/Dockerfile`

The production image. Built on `nvidia/cuda:12.4.1-cudnn-runtime`, installs the
full stack (torch CUDA, vLLM, DeepSpeed, Ray) and serves the FastAPI app. It
requires an NVIDIA GPU host and is intended for the cluster, not a laptop. The
full `docker-compose.yml` brings up this API alongside vLLM, Postgres, Redis,
and an OpenTelemetry collector.

```bash
docker compose up --build        # needs an NVIDIA GPU + the NVIDIA container runtime
```

Building and publishing this image belongs on a GPU CI runner or the training
cluster, and the built image goes to a container registry (for example GHCR),
not into git.

## Reproducibility / CI (CPU): `docker/Dockerfile.cpu`

A lightweight, GPU-free image whose only job is to prove the deterministic core
gives identical results inside a clean container as on bare metal: the S-KBM
physics, the hybrid reward, GRPO advantage normalization, the KL controller,
and the reward-level reward-hacking probe.

```bash
docker build -f docker/Dockerfile.cpu -t pi-grpo-repro:cpu .
docker run --rm pi-grpo-repro:cpu
```

Expected: the property/unit suite passes (13/13) and the probe prints a 0%
catch rate for a preference-only reward versus 100% for the physics-grounded
reward (infeasible total -490.5 vs feasible +10.0). These numbers are
deterministic and must match the bare-metal run.
