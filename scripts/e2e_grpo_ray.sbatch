#!/bin/bash
#SBATCH --job-name=pigrpo-e2e
#SBATCH --account=shekhars
#SBATCH --partition=agsmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# End-to-end Ray-GRPO smoke (CPU-only, agsmall) inside mirror_pnemo.sif.
#
# Runs scripts/e2e_grpo_ray.py, which performs three checks against the REAL
# code already in this repo:
#   (1) reward-actor sweep W=1,2,4 -> throughput is RECORDED per W (asserted);
#       whether it INCREASES is reported honestly (flat/regressing on a tiny CPU
#       task is expected, not a failure).
#   (2) a short train_grpo_ray run -> reward_end >= reward_start (asserted).
#   (3) the step-decomposition CSV (one row per (W, step)) + the per-W scaling
#       summary CSV are emitted (asserted non-empty).
#
# The single PASS/FAIL verdict, a DONE marker, and a human-readable summary are
# written to:
#   /scratch.global/$USER/pigrpo/e2e_summary.txt
# A downstream Slurm dependency can gate on it with:
#   grep -q '^PASS' /scratch.global/$USER/pigrpo/e2e_summary.txt
#
# Self-contained. Submit from anywhere:
#   sbatch scripts/e2e_grpo_ray.sbatch
# Optional overrides via environment:
#   PIGRPO_SIF=/path/to/mirror_pnemo.sif \
#   PIGRPO_ACTOR_COUNTS=1,2,4 PIGRPO_STEPS=8 \
#   sbatch scripts/e2e_grpo_ray.sbatch

set -euo pipefail

# ----------------------------------------------------------------- knobs
ACTOR_COUNTS="${PIGRPO_ACTOR_COUNTS:-1,2,4}"
STEPS="${PIGRPO_STEPS:-8}"
BATCH_PROMPTS="${PIGRPO_BATCH_PROMPTS:-4}"
GROUP_SIZE="${PIGRPO_GROUP_SIZE:-6}"
HORIZON="${PIGRPO_HORIZON:-10}"
SEED="${PIGRPO_SEED:-0}"

# ----------------------------------------------------------------- paths
# Resolve the repo root. Under sbatch the script is COPIED to a spool dir, so
# BASH_SOURCE does NOT point at the repo; prefer an explicit override, then the
# submit dir (sbatch is launched from the repo root), then the script-location
# heuristic for a direct `bash scripts/e2e_grpo_ray.sbatch` invocation.
if [ -n "${PIGRPO_REPO:-}" ]; then
  REPO_ROOT="${PIGRPO_REPO}"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/scripts/e2e_grpo_ray.py" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

SIF="${PIGRPO_SIF:-/scratch.global/${USER}/mirror_pnemo.sif}"

JOB_ID="${SLURM_JOB_ID:-local-$(date +%s)}"
SCRATCH_BASE="/scratch.global/${USER}/pigrpo"
OUT_DIR="${SCRATCH_BASE}/e2e-${JOB_ID}"
SUMMARY="${SCRATCH_BASE}/e2e_summary.txt"     # stable path the task names
DONE_MARKER="${OUT_DIR}/DONE"

mkdir -p "${OUT_DIR}"

# ----------------------------------------------------------------- env
# Keep thread pools and Ray scratch pinned to the allocation so an 8-core node
# does not oversubscribe and Ray's spill stays on node-local /tmp.
NCPU="${SLURM_CPUS_PER_TASK:-8}"

echo "[e2e] job=${JOB_ID} node=$(hostname) cpus=${NCPU}"
echo "[e2e] repo=${REPO_ROOT}"
echo "[e2e] sif=${SIF}"
echo "[e2e] out=${OUT_DIR}"
echo "[e2e] summary=${SUMMARY}"
echo "[e2e] W=${ACTOR_COUNTS} steps=${STEPS} B=${BATCH_PROMPTS} K=${GROUP_SIZE} T=${HORIZON}"

# Pick the container runner (Agate ships apptainer; older nodes singularity).
if command -v apptainer >/dev/null 2>&1; then
  RUNNER="apptainer"
elif command -v singularity >/dev/null 2>&1; then
  RUNNER="singularity"
else
  echo "[e2e] ERROR: neither apptainer nor singularity found on PATH" >&2
  RUNNER=""
fi

if [ -z "${RUNNER}" ] || [ ! -f "${SIF}" ]; then
  # Honest early failure: write the FAIL/DONE verdict so a dependency does not
  # hang waiting for a marker that will never arrive.
  {
    echo "FAIL throughput_recorded=False reward_nondecreasing=False csv_emitted=False"
    echo "reason=missing_container_or_runner runner='${RUNNER}' sif='${SIF}' sif_exists=$([ -f "${SIF}" ] && echo yes || echo no)"
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo ""
    echo "DONE"
  } > "${SUMMARY}"
  echo "FAILED ${JOB_ID} missing container/runner $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${OUT_DIR}/FAILED"
  echo "[e2e] FAILED: missing container or runner -> ${SUMMARY}" >&2
  cat "${SUMMARY}" || true
  exit 1
fi

# ----------------------------------------------------------------- run
# Bind the repo read-only into the container at the same path and set PYTHONPATH
# to the repo root so `import app...` resolves. The e2e driver writes its CSVs
# under OUT_DIR and the verdict to SUMMARY (also re-echoed below for the marker).
set +e
"${RUNNER}" exec \
  --bind "${REPO_ROOT}:${REPO_ROOT}:ro" \
  --bind "${SCRATCH_BASE}:${SCRATCH_BASE}" \
  --env "PYTHONPATH=${REPO_ROOT}" \
  --env "OMP_NUM_THREADS=1" \
  --env "MKL_NUM_THREADS=1" \
  --env "RAY_DEDUP_LOGS=0" \
  --env "RAY_TMPDIR=${TMPDIR:-/tmp}/ray-${JOB_ID}" \
  "${SIF}" \
  python3 "${REPO_ROOT}/scripts/e2e_grpo_ray.py" \
    --actor-counts "${ACTOR_COUNTS}" \
    --steps "${STEPS}" \
    --batch-prompts "${BATCH_PROMPTS}" \
    --group-size "${GROUP_SIZE}" \
    --horizon "${HORIZON}" \
    --seed "${SEED}" \
    --out-dir "${OUT_DIR}" \
    --summary "${SUMMARY}" \
  > "${OUT_DIR}/run.log" 2>&1
RC=$?
set -e

{
  echo ""
  echo "exit_code=${RC}"
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "${OUT_DIR}/run.log"

# If the driver did not produce a summary (e.g. it crashed before writing one),
# synthesize an honest FAIL verdict so the stable path always ends with DONE.
if [ ! -f "${SUMMARY}" ] || ! grep -q "DONE" "${SUMMARY}" 2>/dev/null; then
  {
    echo "FAIL throughput_recorded=False reward_nondecreasing=False csv_emitted=False"
    echo "reason=driver_crashed_before_writing_summary exit_code=${RC}"
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo ""
    echo "DONE"
  } > "${SUMMARY}"
fi

if [ "${RC}" -eq 0 ] && grep -q '^PASS' "${SUMMARY}" 2>/dev/null; then
  echo "OK ${JOB_ID} $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${DONE_MARKER}"
  echo "[e2e] DONE -> ${DONE_MARKER}"
else
  echo "FAILED ${JOB_ID} rc=${RC} $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${OUT_DIR}/FAILED"
  echo "[e2e] FAILED rc=${RC} -> ${OUT_DIR}/FAILED" >&2
fi

echo "----- e2e_summary.txt -----"
cat "${SUMMARY}" || true
echo "----- run.log (tail) -----"
tail -n 40 "${OUT_DIR}/run.log" || true

exit "${RC}"
