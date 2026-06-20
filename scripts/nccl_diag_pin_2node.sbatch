#!/bin/bash
#SBATCH --job-name=nccl-diag-pin
#SBATCH --account=shekhars
#SBATCH --partition=a100-4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# ============================================================================
#  FIX confirmation for the inter-node NCCL hang. The IB-disable control
#  (11434381) still hung, and its NCCL INFO showed NCCL enumerating TWO
#  interfaces on two subnets (aes0 10.31.96.x + ib0 10.31.104.x): the nodes are
#  dual-homed, so ranks connect on mismatched interfaces and deadlock. Fix =
#  PIN NCCL to ONE interface. This run pins NCCL_SOCKET_IFNAME=ib0 (the IPoIB
#  subnet) and keeps native IB enabled. Expect "[diag] DONE all_reduce
#  correct=True".
# ============================================================================
set -euo pipefail

REPO_ROOT="${PIGRPO_REPO:-${SLURM_SUBMIT_DIR:-$HOME/paper-work/pi-grpo}}"
SIF="${PIGRPO_LLM_SIF:-/scratch.global/${USER}/pigrpo_llm.sif}"
GPUS_PER_NODE="${FSDP_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}"
LOGDIR="/scratch.global/${USER}/nccl_logs/${SLURM_JOB_ID}"
mkdir -p "$LOGDIR"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$(( 29500 + (SLURM_JOB_ID % 2000) ))
WORLD=$(( SLURM_NNODES * GPUS_PER_NODE ))

echo "[pin] nodes=${SLURM_NNODES} world=${WORLD} master=${MASTER_ADDR}:${MASTER_PORT} nodelist=${SLURM_JOB_NODELIST} logdir=${LOGDIR}"

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
      export NCCL_DEBUG_FILE=\"${LOGDIR}/nccl.%h.%p.log\"
      export NCCL_SOCKET_IFNAME=ib0          # PIN to a single interface (the fix)
      export TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
      export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
      echo \"[pin] task host=\$(hostname) SLURM_NODEID=\${SLURM_NODEID} NCCL_SOCKET_IFNAME=\${NCCL_SOCKET_IFNAME}\"
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

echo "[pin] srun returned rc=$?  (NCCL INFO logs in ${LOGDIR})"
