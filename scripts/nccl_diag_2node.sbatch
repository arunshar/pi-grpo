#!/bin/bash
#SBATCH --job-name=nccl-diag-2node
#SBATCH --account=shekhars
#SBATCH --partition=a100-4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# ============================================================================
#  Inter-node NCCL diagnostic for the fsdp-2node hang (job 11374530 TIMEOUT).
#  Runs the minimal scripts/nccl_diag.py all-reduce across 2 nodes x 4 GPU = 8
#  ranks with NCCL_DEBUG=INFO so the transport selection (IB vs SOCKET) and the
#  stall point are printed, and TORCH_NCCL_BLOCKING_WAIT so a stalled collective
#  aborts in ~90s with a traceback instead of hanging to the wall-clock limit.
#  If this PASSES, the IB fabric is fine and the real hang is in the model path.
# ============================================================================
set -euo pipefail

REPO_ROOT="${PIGRPO_REPO:-${SLURM_SUBMIT_DIR:-$HOME/paper-work/pi-grpo}}"
SIF="${PIGRPO_LLM_SIF:-/scratch.global/${USER}/pigrpo_llm.sif}"
GPUS_PER_NODE="${FSDP_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$(( 29500 + (SLURM_JOB_ID % 2000) ))
WORLD=$(( SLURM_NNODES * GPUS_PER_NODE ))

echo "[nccl-diag] nodes=${SLURM_NNODES} gpus_per_node=${GPUS_PER_NODE} world=${WORLD}"
echo "[nccl-diag] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} nodelist=${SLURM_JOB_NODELIST}"

srun --kill-on-bad-exit=1 bash -lc "
  apptainer exec --nv \
    --bind /scratch.global:/scratch.global \
    --bind '${REPO_ROOT}:${REPO_ROOT}' \
    '${SIF}' \
    bash -lc '
      set -euo pipefail
      cd \"${REPO_ROOT}\"
      export PYTHONPATH=\"${REPO_ROOT}:\${PYTHONPATH:-}\"
      export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,ENV
      export NCCL_SOCKET_IFNAME=^lo,docker0
      export TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
      export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
      echo \"[nccl-diag] task host=\$(hostname) SLURM_NODEID=\${SLURM_NODEID}\"
      torchrun \
        --nnodes=${SLURM_NNODES} \
        --nproc-per-node=${GPUS_PER_NODE} \
        --node-rank=\${SLURM_NODEID} \
        --rdzv-backend=c10d \
        --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
        --rdzv-id=${SLURM_JOB_ID} \
        scripts/nccl_diag.py
    '
"

echo "[nccl-diag] srun returned rc=$?"
