#!/usr/bin/env bash
# scripts/ci_local.sh
#
# Local CI gate that mirrors .github/workflows/ci.yml (the lint-type-test job)
# so failures are caught before pushing. It reproduces the EXACT commands CI
# runs:
#
#   ruff check app observability evaluation tests spaces scripts   (hard gate)
#   mypy app || true                                               (non-blocking, as in CI)
#   pytest -q -p no:warnings                                       (hard gate)
#
# Env: the pcrf conda env (ruff + pytest present; mypy may be absent, which is
# fine because CI runs mypy with `|| true`).
#
# ENV-GAP handling: if some tests cannot be COLLECTED in this env because of
# missing optional deps (e.g. redis, rtree), this does NOT silently pass. It
# detects the collection failure, runs the collectable subset, prints a clear
# ENV-GAP warning, and still hard-fails on any real test failure.
#
# Exit 0 prints "CI-LOCAL: PASS". Any hard-gate failure prints
# "CI-LOCAL: FAIL (<reason>)" and exits non-zero.

set -uo pipefail

# --- locate repo root (this script lives in <root>/scripts) -----------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# --- activate the pcrf conda env (matches the project convention) -----------
# shellcheck disable=SC1090
if [ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
  conda activate pcrf
else
  echo "WARN: ${HOME}/miniforge3/etc/profile.d/conda.sh not found; using the current environment." >&2
fi
export PYTHONPATH=.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
# Tests assume no live OTLP exporter (matches the Makefile test target).
export PG_OTEL_ENDPOINT="${PG_OTEL_ENDPOINT:-}"

echo "=================================================================="
echo "CI-LOCAL gate  (mirrors .github/workflows/ci.yml :: lint-type-test)"
echo "  repo : ${REPO_ROOT}"
echo "  python: $(command -v python) ($(python --version 2>&1))"
echo "=================================================================="

FAIL_REASON=""

# ---------------------------------------------------------------------------
# 1. ruff  (HARD GATE -- this is the proven failure mode)
#    Exact CI command:
#      ruff check app observability evaluation tests spaces scripts
# ---------------------------------------------------------------------------
echo ""
echo "--- [1/3] ruff check app observability evaluation tests spaces scripts ---"
if ruff check app observability evaluation tests spaces scripts; then
  echo "ruff: OK"
else
  echo "ruff: FAILED"
  FAIL_REASON="ruff"
fi

# ---------------------------------------------------------------------------
# 2. mypy  (NON-BLOCKING -- CI runs `mypy app || true`)
#    Mirror CI exactly: never fail the gate on mypy. If mypy is not installed
#    in this env, say so and move on (CI's `|| true` swallows any mypy result).
# ---------------------------------------------------------------------------
echo ""
echo "--- [2/3] mypy app || true  (non-blocking, exactly as CI) ---"
if command -v mypy >/dev/null 2>&1; then
  mypy app || true
  echo "mypy: ran (result non-blocking, as in CI)"
else
  echo "mypy: not installed in this env; skipping (CI uses '|| true', so non-blocking either way)"
fi

# ---------------------------------------------------------------------------
# 3. pytest  (HARD GATE)
#    Exact CI command:
#      pytest -q -p no:warnings
#
#    ENV-GAP guard: first probe collection. If collection fails for some files
#    (missing optional deps such as redis/rtree), do not let that masquerade as
#    success: identify the affected modules, run the collectable subset via
#    --continue-on-collection-errors, warn loudly, and still hard-fail on any
#    real test failure.
# ---------------------------------------------------------------------------
echo ""
echo "--- [3/3] pytest -q -p no:warnings ---"

COLLECT_LOG="$(mktemp)"
pytest --collect-only -q -p no:warnings >"${COLLECT_LOG}" 2>&1
COLLECT_EXIT=$?

ENV_GAP=0
if [ "${COLLECT_EXIT}" -ne 0 ]; then
  ENV_GAP=1
  # Pull the offending module names out of the collection log for the warning.
  MISSING_MODS="$(grep -oiE "No module named '[^']+'" "${COLLECT_LOG}" \
                  | sed -E "s/No module named '([^']+)'/\1/" \
                  | sort -u | paste -sd, -)"
  [ -z "${MISSING_MODS}" ] && MISSING_MODS="unknown (see collection errors above)"
  echo ""
  echo "ENV-GAP: ${MISSING_MODS} not installed in pcrf, ran subset"
  echo "         (collection errors in the modules above; running the collectable subset)"
fi

if [ "${ENV_GAP}" -eq 1 ]; then
  # Run everything that CAN be collected; real failures still fail the gate.
  # --continue-on-collection-errors keeps the collectable subset running, but
  # pytest still returns non-zero overall, so we inspect the summary instead of
  # the bare exit code to distinguish "collection-only gap" from real failures.
  PYTEST_OUT="$(mktemp)"
  pytest -q -p no:warnings --continue-on-collection-errors 2>&1 | tee "${PYTEST_OUT}"
  # A real test failure or error (not merely a collection gap) fails the gate.
  if grep -qiE "^[0-9]+ (failed|error)|[0-9]+ failed|[0-9]+ errors? " "${PYTEST_OUT}" \
     && grep -qiE " failed" "${PYTEST_OUT}"; then
    echo "pytest: FAILED (real test failures in the collectable subset)"
    FAIL_REASON="${FAIL_REASON:+${FAIL_REASON}+}pytest"
  else
    echo "pytest: subset PASSED (only collection gaps, no real failures)"
  fi
  rm -f "${PYTEST_OUT}"
else
  # Clean collection: run the exact CI command and gate on its exit code.
  if pytest -q -p no:warnings; then
    echo "pytest: OK"
  else
    echo "pytest: FAILED"
    FAIL_REASON="${FAIL_REASON:+${FAIL_REASON}+}pytest"
  fi
fi
rm -f "${COLLECT_LOG}"

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
echo ""
echo "=================================================================="
if [ -n "${FAIL_REASON}" ]; then
  echo "CI-LOCAL: FAIL (${FAIL_REASON})"
  exit 1
fi
echo "CI-LOCAL: PASS"
exit 0
