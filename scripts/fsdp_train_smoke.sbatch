#!/bin/bash
#SBATCH --job-name=fsdp-smoke
#SBATCH --account=shekhars
#SBATCH --partition=a100-4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# ============================================================================
#  FAST 2-GPU toy smoke for the FSDP trainer. Runs the toy model (no HF, no 7B
#  weights) on 2 GPUs through the FULL FSDP/NCCL path: per-block auto-wrap,
#  FULL_SHARD, bf16 mixed precision, forward + backward + an optimizer step.
#  This is the exact NCCL path that asserted "expected FORWARD_BACKWARD but
#  current state is IDLE" when the toy forward returned a bare object instead of
#  a NamedTuple. (CPU/gloo is NOT used: this container's torch build fails the
#  FSDP wrap on CPU, independent of the output-traversal bug under test.)
#  --redirects=3 --tee=3 tees EVERY rank's stdout/stderr so a non-rank-0 failure
#  is visible (the suppressed-traceback problem the first smoke hit).
#  PASS = no assert, both ranks print step losses + tok/s, fsdp_scaling_2.json
#  is written, and the run exits rc=0.
# ============================================================================
set -euo pipefail

REPO_ROOT="${PIGRPO_REPO:-${SLURM_SUBMIT_DIR:-$HOME/paper-work/pi-grpo}}"
SIF="${PIGRPO_LLM_SIF:-/scratch.global/${USER}/pigrpo_llm.sif}"

echo "[fsdp-smoke] node=$(hostname) repo=${REPO_ROOT} sif=${SIF}"
nvidia-smi -L || true

apptainer exec --nv \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --bind /scratch.global:/scratch.global \
  --bind "${REPO_ROOT}:${REPO_ROOT}" \
  "${SIF}" \
  bash -lc "
    set -euo pipefail
    cd '${REPO_ROOT}'
    export PYTHONPATH='${REPO_ROOT}:\${PYTHONPATH:-}'
    export NCCL_DEBUG=WARN OMP_NUM_THREADS=4
    python3 -c 'import torch; print(\"[fsdp-smoke] torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), \"gpus\", torch.cuda.device_count())'
    torchrun --standalone --nnodes=1 --nproc_per_node=2 --redirects=3 --tee=3 \
      scripts/fsdp_train.py \
      --model toy --toy-layers 2 --toy-dim 128 \
      --batch 1 --seq-len 64 --steps 4 --warmup 1 \
      --out-dir /scratch.global/${USER}/pigrpo/fsdp_smoke
  "

echo "[fsdp-smoke] done; result at /scratch.global/${USER}/pigrpo/fsdp_smoke/fsdp_scaling_2.json"
