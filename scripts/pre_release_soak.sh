#!/bin/bash
# Pre-release soak gate — run locally before tagging.
#
# Mirrors the `soak-gate` job in .github/workflows/release.yml so you can catch
# a divergence before pushing. It runs the same four steps:
#
#   1. the randomized differential workload (model oracle)
#   2. the crash-injection workload sweep
#   3. the long soak (single-writer churn)
#   4. the multi-writer soak (concurrent writers on a shared store)
#
# Usage:
#   ./scripts/pre_release_soak.sh            # default 30-minute soak
#   ./scripts/pre_release_soak.sh 5          # 5-minute soak
#   ./scripts/pre_release_soak.sh --quick    # 1-minute smoke run
#   BSDB_SOAK_SECONDS=600 ./scripts/pre_release_soak.sh   # explicit seconds
#
# Exit code is non-zero if any step fails, so it can gate a release. A
# publishable markdown report is written to `soak-report.md` (override with
# SOAK_REPORT=/path/to/report.md).
set -euo pipefail

# First positional arg is the soak duration in minutes (default 30).
# `--quick` is a 1-minute smoke run. `BSDB_SOAK_SECONDS` (seconds) wins if set.
SOAK_MINUTES="${1:-30}"
if [[ "${SOAK_MINUTES}" == "--quick" ]]; then
    SOAK_MINUTES=1
fi
if ! [[ "${SOAK_MINUTES}" =~ ^[0-9]+$ ]]; then
    echo "error: expected a soak duration in minutes (or --quick), got '${SOAK_MINUTES}'" >&2
    exit 2
fi

if [[ -n "${BSDB_SOAK_SECONDS:-}" ]]; then
    SOAK_SECONDS="${BSDB_SOAK_SECONDS}"
else
    SOAK_SECONDS=$(( SOAK_MINUTES * 60 ))
fi

REPORT="${SOAK_REPORT:-soak-report.md}"

export SKIP_GPU_TESTS="${SKIP_GPU_TESTS:-1}"
export RUST_LOG="${RUST_LOG:-error}"
export BSDB_SOAK_SECONDS="${SOAK_SECONDS}"

STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=== Pre-release soak gate (soak=${SOAK_SECONDS}s / ${SOAK_MINUTES}m) ==="
echo "started_at=${STARTED_AT}"

echo
echo "--- 1/4 Randomized differential workload (model oracle) ---"
cargo test --test test_randomized_differential_workload -- --nocapture

echo
echo "--- 2/4 Crash-injection workload sweep ---"
cargo test --test test_crash_injection_workload -- --nocapture

echo
echo "--- 3/4 Long soak (${SOAK_SECONDS}s) ---"
SOAK_LOG="$(mktemp)"
MW_LOG="$(mktemp)"
trap 'rm -f "${SOAK_LOG}" "${MW_LOG}"' EXIT
cargo test --test soak -- --ignored --nocapture 2>&1 | tee "${SOAK_LOG}"

echo
echo "--- 4/4 Multi-writer soak (${SOAK_SECONDS}s) ---"
cargo test --test test_multi_writer_concurrency multi_writer_soak -- --ignored --nocapture 2>&1 | tee "${MW_LOG}"

FINISHED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo
echo "=== Soak stats (publishable) ==="
grep -h '\[soak-stats\]' "${SOAK_LOG}" "${MW_LOG}" || echo "(no [soak-stats] lines emitted)"

# Write a publishable markdown report.
{
    echo "# Pre-release soak report"
    echo
    echo "| Field | Value |"
    echo "|---|---|"
    echo "| Started | ${STARTED_AT} |"
    echo "| Finished | ${FINISHED_AT} |"
    echo "| Soak duration | ${SOAK_SECONDS}s |"
    echo "| Result | PASSED |"
    echo
    echo "## Soak stats"
    echo
    echo '```'
    grep -h '\[soak-stats\]' "${SOAK_LOG}" "${MW_LOG}" || echo "(no stats emitted)"
    echo '```'
} > "${REPORT}"
echo "wrote ${REPORT}"

echo
echo "=== Pre-release soak gate PASSED ==="
