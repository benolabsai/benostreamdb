#!/usr/bin/env bash
# scripts/pre_release_check.sh
# Pre-release steering gate — MUST pass 100% locally prior to git commit, tag, or push.
# Mirrors all critical checks in .github/workflows/ci.yml and .github/workflows/release.yml.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

echo "================================================================="
echo " BenoStreamDB Pre-Release Steering Gate"
echo "================================================================="
echo "Repo root: ${REPO_ROOT}"

# 1. Rust toolchain.
#
# `rust-toolchain.toml` is the single source of truth: cargo resolves the pinned
# channel for anything invoked inside this directory, and rustup auto-installs it
# on first use. This script used to hardcode the version here as well, which made
# a bump a three-place edit (pin file, 19 CI pins, this script) and — worse —
# silently kept verifying the *old* compiler whenever a machine had that
# toolchain lying around. It did exactly that during the 1.100.0 bump: a
# `rust-version = "1.100"` floor plus this 1.98.0 override produced
# "requires rustc 1.100" on every crate.
CARGO_CMD="${CARGO_CMD:-cargo}"
echo "Using: $(${CARGO_CMD} --version)"

# 2. Rust Formatting Check (mirrors ci.yml 'Check formatting')
echo ""
echo "--> [1/10] Checking Rust code formatting (cargo fmt --check)..."
if ! ${CARGO_CMD} fmt --check; then
    echo "ERROR: Rust code formatting check failed!"
    echo "Run 'cargo fmt' to automatically apply formatting fixes before committing."
    exit 1
fi
echo "PASS: cargo fmt --check passed."

# 3. Rust Clippy (mirrors ci.yml 'Run Clippy')
echo ""
echo "--> [2/10] Running Clippy on all targets (cargo clippy --all-targets)..."
if ! ${CARGO_CMD} clippy --all-targets; then
    echo "ERROR: Clippy checks failed!"
    exit 1
fi
echo "PASS: Clippy passed with 0 warnings."

# 4. Cargo Check (mirrors ci.yml 'Run Cargo Check' & 'Check binary targets')
echo ""
echo "--> [3/10] Checking compilation on all targets & bins..."
${CARGO_CMD} check --all-targets
${CARGO_CMD} check --bins
echo "PASS: Cargo check passed on all targets and binaries."

# 5. Rustdoc Lint (mirrors ci.yml 'Build API docs')
echo ""
echo "--> [4/10] Checking documentation links and build (cargo doc)..."
${CARGO_CMD} doc --no-deps --features java
echo "PASS: cargo doc passed without broken doc links."

# 6. Cargo Deny / Security Advisory Check (mirrors ci.yml 'Cargo Audit')
echo ""
echo "--> [5/10] Running Cargo Deny security audit (cargo deny check)..."
if command -v cargo-deny &> /dev/null; then
    cargo deny --manifest-path "${REPO_ROOT}/Cargo.toml" check
    echo "PASS: cargo deny audit passed (0 unreviewed vulnerabilities/bans)."
else
    echo "WARNING: cargo-deny is not installed locally. Skipping local audit step."
fi

# 7. Version Single-Source Verification (mirrors ci.yml 'version-check')
echo ""
echo "--> [6/10] Asserting version single-source alignment across all connectors..."
python3 - <<'PY'
import pathlib, re, sys

root = pathlib.Path(".")
cargo_text = (root / "Cargo.toml").read_text()
m_ver = re.search(r'(?ms)^\[workspace\.package\].*?^version\s*=\s*"([^"]+)"', cargo_text)
if not m_ver:
    print("ERROR: Could not find [workspace.package].version in Cargo.toml")
    sys.exit(1)
v = m_ver.group(1)
print(f"Authoritative workspace version: {v}")

errs = []

# Pyproject dynamic versions
for p in [
    "pyproject.toml",
    "contrib/benostreamdb-search/pyproject.toml",
    "server/flight_sql/pyproject.toml",
    "contrib/benostreamdb-mcp/pyproject.toml",
]:
    path = root / p
    if not path.exists():
        errs.append(f"{p}: file not found")
        continue
    if 'dynamic = ["version"]' not in path.read_text():
        errs.append(f'{p}: [project] must declare dynamic = ["version"]')

# Inter-crate dependencies
for p in [
    "contrib/benostreamdb-search/Cargo.toml",
    "server/flight_sql/Cargo.toml",
]:
    path = root / p
    if not path.exists():
        errs.append(f"{p}: file not found")
        continue
    m = re.search(r'benostreamdb = \{ version = "([^"]+)"', path.read_text())
    if not m or m.group(1) != v:
        found = m.group(1) if m else "missing"
        errs.append(f"{p}: benostreamdb dependency pin is '{found}', expected '{v}'")

# dbt adapter checks
dbt_pyproject = root / "dbt-benostreamdb/pyproject.toml"
if dbt_pyproject.exists():
    m = re.search(r'benostreamdb>=([^",\s]+)', dbt_pyproject.read_text())
    if not m or m.group(1) != v:
        found = m.group(1) if m else "missing"
        errs.append(f"dbt-benostreamdb/pyproject.toml: benostreamdb>= floor is '{found}', expected '{v}'")

dbt_project_yml = root / "dbt-benostreamdb/dbt/include/benostreamdb/dbt_project.yml"
if dbt_project_yml.exists():
    m = re.search(r'^version:\s*([0-9\.\w]+)', dbt_project_yml.read_text(), re.MULTILINE)
    if not m or m.group(1) != v:
        found = m.group(1) if m else "missing"
        errs.append(f"dbt-benostreamdb/dbt/include/benostreamdb/dbt_project.yml: version is '{found}', expected '{v}'")

# Spark POM revision
spark_pom = root / "spark-benostreamdb/pom.xml"
if spark_pom.exists():
    m = re.search(r'<revision>([^<]+)</revision>', spark_pom.read_text())
    if not m or m.group(1) != v:
        found = m.group(1) if m else "missing"
        errs.append(f"spark-benostreamdb/pom.xml: <revision> is '{found}', expected '{v}'")

# Trino POM revision
trino_pom = root / "trino-benostreamdb/pom.xml"
if trino_pom.exists():
    m = re.search(r'<revision>([^<]+)</revision>', trino_pom.read_text())
    if not m or m.group(1) != v:
        found = m.group(1) if m else "missing"
        errs.append(f"trino-benostreamdb/pom.xml: <revision> is '{found}', expected '{v}'")

if errs:
    print("\nVersion Alignment Errors:")
    for e in errs:
        print(f"  - {e}")
    sys.exit(1)

print("PASS: All connectors, crates, and metadata match workspace version.")
PY
echo "PASS: Version alignment verified."

# Packaging and Symlink Integrity Check (prevents crates.io upload failures)
echo ""
echo "--> Checking packaging integrity and dangling symlinks..."
BROKEN_SYMLINKS=$(find "${REPO_ROOT}" -not -path '*/.*' -not -path '*/target/*' -not -path '*/benchmarks/*' -xtype l)
if [[ -n "${BROKEN_SYMLINKS}" ]]; then
    echo "ERROR: Broken symlinks detected in repository:"
    echo "${BROKEN_SYMLINKS}"
    exit 1
fi
if ! ${CARGO_CMD} package --manifest-path "${REPO_ROOT}/Cargo.toml" --list --allow-dirty > /dev/null 2>&1; then
    echo "ERROR: 'cargo package --list' failed! Package contains unarchivable files."
    ${CARGO_CMD} package --manifest-path "${REPO_ROOT}/Cargo.toml" --list --allow-dirty
    exit 1
fi
echo "PASS: Packaging integrity and symlink targets verified."

# 8. Rust Unit & Regression Tests (including fuzzer regression test cases)
echo ""
echo "--> [7/10] Running core Rust unit & regression tests..."
${CARGO_CMD} test --lib core::planner::between_dump_tests
echo "PASS: Core parser and fuzzer regression tests passed."

# 9. Fuzz Smoke Check (untrusted connector input parsers)
echo ""
echo "--> [8/10] Running fuzz smoke test on untrusted parser inputs..."
if command -v cargo-fuzz &> /dev/null && command -v rustup &> /dev/null && rustup toolchain list | grep -q "^nightly"; then
    echo "Running 500-run smoke test on connector_inputs with cargo-fuzz..."
    (cd "${REPO_ROOT}/fuzz" && cargo +nightly fuzz run connector_inputs -- -runs=500 -max_total_time=10)
    echo "PASS: Fuzz smoke test passed with 0 crashes."
else
    echo "NOTE: cargo-fuzz or nightly toolchain not found; relying on Rust regression unit tests."
fi

# 10. Python Regressions
echo ""
echo "--> [9/10] Running Python regression tests..."
PY_BIN="python3"
if [[ -f "${REPO_ROOT}/../benolabs/.venv/bin/python" ]]; then
    PY_BIN="${REPO_ROOT}/../benolabs/.venv/bin/python"
elif [[ -f "${REPO_ROOT}/.venv/bin/python" ]]; then
    PY_BIN="${REPO_ROOT}/.venv/bin/python"
fi

if [[ -f "${REPO_ROOT}/tests/python/test_vector_string_conformance.py" ]]; then
    echo "Running test_vector_string_conformance.py with ${PY_BIN}..."
    ${PY_BIN} "${REPO_ROOT}/tests/python/test_vector_string_conformance.py"
fi
echo "PASS: Python regression tests passed."

# 12. Published-README guard
#
# A crate's README is frozen into the .crate tarball at publish time and cannot be
# edited afterwards, and PyPI behaves the same way for the long description. A
# pinned "Developer Preview (vX.Y.Z)" banner therefore goes stale on crates.io the
# moment the next release ships, and the only way to correct it is another release.
echo ""
echo "--> Checking that the README banner carries no pinned version..."
if grep -nE 'Developer Preview \(v[0-9]+\.[0-9]+\.[0-9]+\)' "${REPO_ROOT}/README.md"; then
    echo "FAIL: README.md pins a version in the Developer Preview banner."
    echo "      Remove the number: the published README cannot be edited after a"
    echo "      release, and the version already lives in CHANGELOG.md."
    exit 1
fi
echo "PASS: README banner is version-free."

# 13. Manifest TOML validity
#
# Every tool that walks the tree parses these manifests itself: pytest during
# rootdir discovery, maturin for the build backend, pip for resolution. A stray
# byte on line 1 of a pyproject therefore fails inside a test run or a wheel
# build, far from the edit that caused it. 0.16.1 shipped exactly that defect,
# with three manifests prefixed by a fragment of shell text, and the failure
# surfaced as "Expected '=' after a key" from pytest. Parse them here instead.
echo ""
echo "--> Checking that connector manifests parse as TOML..."
if ! "${PY_BIN}" - <<'PY'
import pathlib, sys, tomllib

bad = []
for p in ["pyproject.toml", "dbt-benostreamdb/pyproject.toml",
          "contrib/benostreamdb-search/pyproject.toml",
          "server/flight_sql/pyproject.toml",
          "contrib/benostreamdb-mcp/pyproject.toml"]:
    path = pathlib.Path(p)
    if not path.exists():
        bad.append(f"{p}: missing")
        continue
    try:
        tomllib.loads(path.read_text())
    except Exception as exc:
        bad.append(f"{p}: {exc}")
    else:
        first = path.read_text().splitlines()[0] if path.read_text() else ""
        if first != "[build-system]":
            bad.append(f'{p}: line 1 is {first!r}, expected "[build-system]"')
if bad:
    print("\n".join("  - " + b for b in bad))
    sys.exit(1)
print("PASS: connector manifests parse and start at [build-system].")
PY
then
    echo "FAIL: malformed connector manifest (see above)."
    exit 1
fi

# 11. Git Status & Uncommitted Check
echo ""
echo "--> [10/10] Checking for unformatted or uncommitted modifications..."
if ! git diff --quiet; then
    echo "WARNING: Modified uncommitted files found in working tree:"
    git status -s
fi

echo ""
echo "================================================================="
echo " ALL PRE-RELEASE CHECKS PASSED (100% GREEN)                      "
echo " Safe to commit, tag, and push release.                          "
echo "================================================================="
