#!/bin/bash
#SBATCH --job-name=nccl-diag-noib
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
#  Root-cause confirmation for the inter-node NCCL hang (jobs 11374530 FSDP and
#  11427744 minimal all-reduce BOTH timed out on the FIRST cross-node collective
#  after a clean init_process_group). Hypothesis: the IB/GPUDirect transport,
#  established lazily on the first collective, cannot connect across nodes.
#
#  This run FORCES NCCL onto TCP (NCCL_IB_DISABLE=1) and writes NCCL INFO to a
#  per-rank file that survives a kill (NCCL_DEBUG_FILE). If the all-reduce now
#  COMPLETES ("[diag] DONE all_reduce correct=True"), IB is confirmed the culprit
#  and NCCL_IB_DISABLE=1 is a working (slower, TCP) workaround; the real fix is
#  selecting the correct IB HCA / GID for these nodes.
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

echo "[noib] nodes=${SLURM_NNODES} world=${WORLD} master=${MASTER_ADDR}:${MASTER_PORT} nodelist=${SLURM_JOB_NODELIST} logdir=${LOGDIR}"

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
      export NCCL_IB_DISABLE=1            # force TCP transport (the test)
      export NCCL_SOCKET_IFNAME=^lo,docker0
      export TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
      export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
      echo \"[noib] task host=\$(hostname) SLURM_NODEID=\${SLURM_NODEID} NCCL_IB_DISABLE=\${NCCL_IB_DISABLE}\"
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

echo "[noib] srun returned rc=$?  (NCCL INFO logs in ${LOGDIR})"
