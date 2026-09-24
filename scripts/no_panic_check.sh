#!/usr/bin/env bash
# scripts/no_panic_check.sh
#
# No-panic ratchet for production code paths.
#
# Counts `unwrap()` / `expect()` / `panic!` sites in *production* (non-test)
# code and fails if the count has grown relative to
# `scripts/no_panic_baseline.txt`.
#
# Why a ratchet instead of a hard `#![deny(clippy::unwrap_used)]`:
# both `src/lib.rs` and `benostreamdb-search/src/lib.rs` already carry
# `#![deny(warnings)]`, so enabling the restriction lints as errors today would
# fail the build on every not-yet-remediated site. The hard attribute is staged
# behind the `no-panic` cargo feature (see the crate roots) and is turned on in
# CI once Phase 1 remediation has driven the baseline down to the documented
# invariant allowlist.
#
# `cargo clippy --lib` compiles the crate with `cfg(test)` disabled, so
# `#[cfg(test)] mod tests` blocks are not compiled and their (idiomatic)
# unwraps are excluded automatically. `--no-deps` keeps the count to the
# selected package, so the search crate does not double-count the core crate.
# The separate `tests/` crates are never built by this script either.
#
# Usage:
#   scripts/no_panic_check.sh                 # compare against the baseline
#   scripts/no_panic_check.sh --write         # regenerate the baseline
#
# Environment:
#   NO_PANIC_FEATURES   feature set for the root package (default: python)
#   NO_PANIC_BASELINE   baseline file path (default: scripts/no_panic_baseline.txt)

set -uo pipefail

cd "$(dirname "$0")/.."

BASELINE_FILE="${NO_PANIC_BASELINE:-scripts/no_panic_baseline.txt}"
FEATURES="${NO_PANIC_FEATURES-python}"
WRITE=0
[[ "${1:-}" == "--write" ]] && WRITE=1

LINT_DENIES=(-D clippy::unwrap_used -D clippy::expect_used -D clippy::panic)

feature_args=()
[[ -n "${FEATURES}" ]] && feature_args=(--features "${FEATURES}")

# Populated by run_clippy.
COUNT=0

# run_clippy <label> <cargo clippy target args...>
#
# Sets COUNT to the number of no-panic lint diagnostics in the selected
# package. The denied lints are the expected, countable signal, so a non-zero
# cargo exit is fine *provided* diagnostics were emitted; a genuine rustc error
# (or a silent failure) aborts the script.
run_clippy() {
    local label="$1"
    shift
    local out rc count

    out="$(cargo clippy --no-deps "$@" --message-format=json -- "${LINT_DENIES[@]}" 2>&1)"
    rc=$?

    count="$(printf '%s\n' "${out}" \
        | grep -oE '"clippy::(unwrap_used|expect_used|panic)"' \
        | wc -l \
        | tr -d '[:space:]')"
    count="${count:-0}"

    if printf '%s\n' "${out}" | grep -qE 'error\[E[0-9]+\]'; then
        printf '%s\n' "${out}" | tail -n 40 >&2
        echo "no-panic: ${label}: build failed (rustc error)" >&2
        exit 2
    fi
    if ((rc != 0 && count == 0)); then
        printf '%s\n' "${out}" | tail -n 40 >&2
        echo "no-panic: ${label}: cargo failed with no lint diagnostics" >&2
        exit 2
    fi

    COUNT="${count}"
    printf '  %-38s %s\n' "${label}" "${count}"
}

total=0

run_clippy "benostreamdb (lib):" --lib ${feature_args[@]+"${feature_args[@]}"}
total=$((total + COUNT))

run_clippy "benostreamdb-search (lib):" -p benostreamdb-search --lib
total=$((total + COUNT))

# NOTE: binary targets are intentionally not counted here. `cargo clippy --bin
# <name>` re-lints the same-package lib (so the count duplicates the `--lib`
# line) and does not reliably emit the bin's own diagnostics. Bin sources are
# covered by the staged `no-panic` feature on their crate roots once CI turns it
# on, and are remediated directly.

if [[ "${WRITE}" == "1" ]]; then
    printf '%s\n' "${total}" >"${BASELINE_FILE}"
    echo "no-panic: wrote baseline ${total} to ${BASELINE_FILE}"
    exit 0
fi

if [[ ! -f "${BASELINE_FILE}" ]]; then
    echo "no-panic: baseline file ${BASELINE_FILE} is missing; run with --write" >&2
    exit 2
fi

baseline="$(tr -d '[:space:]' <"${BASELINE_FILE}")"
echo "no-panic: ${total} production panic site(s), baseline ${baseline}"

if ((total > baseline)); then
    echo "no-panic: FAIL - production panic sites increased (${baseline} -> ${total})." >&2
    echo "no-panic: replace unwrap()/expect()/panic! with ? / BenoStreamError," >&2
    echo "no-panic: or document a justified #[allow] if it is a true invariant." >&2
    exit 1
fi

if ((total < baseline)); then
    echo "no-panic: improved - lower the baseline to ${total} (scripts/no_panic_check.sh --write)."
fi

# Hard gate: with the `no-panic` feature enabled the restriction lints are hard
# errors for non-test code (see NO_PANIC_POLICY.md). Vendored and explicitly
# allowlisted modules are exempted in source. Now that the baseline is zero this
# is the real enforcement; the ratchet above stays as a fast, visible signal.
HARD_FEATURES="no-panic"
[[ -n "${FEATURES}" ]] && HARD_FEATURES="${FEATURES},no-panic"

echo "no-panic: enforcing the hard gate (--features ${HARD_FEATURES})"
if ! cargo clippy --no-deps --lib --features "${HARD_FEATURES}" >/dev/null 2>&1; then
    echo "no-panic: FAIL - hard gate violated (unwrap/expect/panic in production code)" >&2
    cargo clippy --no-deps --lib --features "${HARD_FEATURES}" 2>&1 \
        | grep -E '^(error|warning)' | head -n 20 >&2
    exit 1
fi

echo "no-panic: OK"
