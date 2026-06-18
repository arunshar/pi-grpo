#!/bin/bash
#SBATCH --job-name=pigrpo-scale
#SBATCH --account=shekhars
#SBATCH --partition=agsmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Reward-actor scaling sweep for the Ray GRPO path (CPU-only, agsmall).
#
# Self-contained: runs the W=1,2,4,8,16 sweep via scripts/grpo_scale.py, then
# writes a DONE marker and a human-readable summary under
#   /scratch.global/$USER/pigrpo/<jobid>/
# There is NO login-node poller; downstream steps should gate on the DONE
# marker (e.g. a Slurm dependency on this job id, or by polling the marker file
# from another batch job).
#
# Submit from the repo root:
#   sbatch scripts/grpo_ray_slurm.sbatch
# Optional overrides via environment:
#   PIGRPO_ACTOR_COUNTS=1,2,4,8 PIGRPO_STEPS=8 sbatch scripts/grpo_ray_slurm.sbatch

set -euo pipefail

# ----------------------------------------------------------------- knobs
ACTOR_COUNTS="${PIGRPO_ACTOR_COUNTS:-1,2,4,8,16}"
STEPS="${PIGRPO_STEPS:-5}"
BATCH_PROMPTS="${PIGRPO_BATCH_PROMPTS:-8}"
GROUP_SIZE="${PIGRPO_GROUP_SIZE:-8}"
HORIZON="${PIGRPO_HORIZON:-12}"
SEED="${PIGRPO_SEED:-42}"

# ----------------------------------------------------------------- paths
# Resolve the repo root. Under sbatch the script is COPIED to a spool dir
# (/var/spool/slurmd/...), so BASH_SOURCE does NOT point at the repo tree. Prefer
# an explicit override, then the submit dir (sbatch is launched from the repo
# root), then fall back to the script-location heuristic for a direct
# `bash scripts/grpo_ray_slurm.sbatch` invocation.
if [ -n "${PIGRPO_REPO:-}" ]; then
  REPO_ROOT="${PIGRPO_REPO}"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/scripts/grpo_scale.py" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

JOB_ID="${SLURM_JOB_ID:-local-$(date +%s)}"
SCRATCH_BASE="/scratch.global/${USER}/pigrpo"
OUT_DIR="${SCRATCH_BASE}/${JOB_ID}"
DONE_MARKER="${OUT_DIR}/DONE"
SUMMARY="${OUT_DIR}/summary.txt"

mkdir -p "${OUT_DIR}"

# ----------------------------------------------------------------- env
# Pin Ray + thread pools to the allocation so a 16-core node does not oversubscribe.
NCPU="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAY_DEDUP_LOGS=0
# Keep Ray's own scratch on node-local /tmp, not on the shared filesystem.
export RAY_TMPDIR="${TMPDIR:-/tmp}/ray-${JOB_ID}"
mkdir -p "${RAY_TMPDIR}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Module / venv activation. ADAPT to the environment that has Ray>=2.34 + torch.
# TODO(env): replace this block with the project's real activation. The sweep
# needs the `.[ray]` extra (see pyproject.toml optional-deps) plus the sibling
# Ray-driver task's app/policy/ray_driver.py. We do NOT hard-fail if module is
# missing so a venv-only setup also works.
if command -v module >/dev/null 2>&1; then
  module purge || true
  module load python3 || true
fi
if [ -n "${PIGRPO_VENV:-}" ] && [ -f "${PIGRPO_VENV}/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${PIGRPO_VENV}/bin/activate"
fi

PYBIN="${PIGRPO_PYTHON:-python3}"

echo "[grpo-scale] job=${JOB_ID} node=$(hostname) cpus=${NCPU}"
echo "[grpo-scale] repo=${REPO_ROOT}"
echo "[grpo-scale] out=${OUT_DIR}"
echo "[grpo-scale] W=${ACTOR_COUNTS} steps=${STEPS} B=${BATCH_PROMPTS} K=${GROUP_SIZE} T=${HORIZON}"

# ----------------------------------------------------------------- run
# On any failure we still write a FAILED marker so a downstream dependency can
# distinguish "not done yet" from "done but failed".
set +e
"${PYBIN}" "${REPO_ROOT}/scripts/grpo_scale.py" \
  --actor-counts "${ACTOR_COUNTS}" \
  --steps "${STEPS}" \
  --batch-prompts "${BATCH_PROMPTS}" \
  --group-size "${GROUP_SIZE}" \
  --horizon "${HORIZON}" \
  --seed "${SEED}" \
  --out-dir "${OUT_DIR}" \
  > "${SUMMARY}" 2>&1
RC=$?
set -e

{
  echo ""
  echo "exit_code=${RC}"
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "${SUMMARY}"

if [ "${RC}" -eq 0 ]; then
  echo "OK ${JOB_ID} $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${DONE_MARKER}"
  echo "[grpo-scale] DONE -> ${DONE_MARKER}"
else
  echo "FAILED ${JOB_ID} rc=${RC} $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${OUT_DIR}/FAILED"
  echo "[grpo-scale] FAILED rc=${RC} -> ${OUT_DIR}/FAILED" >&2
fi

cat "${SUMMARY}" || true
exit "${RC}"
