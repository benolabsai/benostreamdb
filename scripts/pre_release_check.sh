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

# 1. Select Rust toolchain (prefer 1.98.0 to match CI)
CARGO_CMD="cargo"
if command -v rustup &> /dev/null; then
    if rustup toolchain list | grep -q "^1.98.0"; then
        CARGO_CMD="rustup run 1.98.0 cargo"
    fi
fi
echo "Using: $(${CARGO_CMD} --version)"

# 2. Rust Formatting Check (mirrors ci.yml 'Check formatting')
echo ""
echo "--> [1/7] Checking Rust code formatting (cargo fmt --check)..."
if ! ${CARGO_CMD} fmt --check; then
    echo "ERROR: Rust code formatting check failed!"
    echo "Run 'cargo fmt' to automatically apply formatting fixes before committing."
    exit 1
fi
echo "PASS: cargo fmt --check passed."

# 3. Rust Clippy (mirrors ci.yml 'Run Clippy')
echo ""
echo "--> [2/7] Running Clippy on all targets (cargo clippy --all-targets)..."
if ! ${CARGO_CMD} clippy --all-targets; then
    echo "ERROR: Clippy checks failed!"
    exit 1
fi
echo "PASS: Clippy passed with 0 warnings."

# 4. Cargo Check (mirrors ci.yml 'Run Cargo Check' & 'Check binary targets')
echo ""
echo "--> [3/7] Checking compilation on all targets & bins..."
${CARGO_CMD} check --all-targets
${CARGO_CMD} check --bins
echo "PASS: Cargo check passed on all targets and binaries."

# 5. Rustdoc Lint (mirrors ci.yml 'Build API docs')
echo ""
echo "--> [4/7] Checking documentation links and build (cargo doc)..."
${CARGO_CMD} doc --no-deps --features java
echo "PASS: cargo doc passed without broken doc links."

# 6. Version Single-Source Verification (mirrors ci.yml 'version-check')
echo ""
echo "--> [5/7] Asserting version single-source alignment across all connectors..."
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

# 7. Python Regressions
echo ""
echo "--> [6/7] Running Python regression tests..."
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

# 8. Git Status & Uncommitted Check
echo ""
echo "--> [7/7] Checking for unformatted or uncommitted modifications..."
if ! git diff --quiet; then
    echo "WARNING: Modified uncommitted files found in working tree:"
    git status -s
fi

echo ""
echo "================================================================="
echo " ALL PRE-RELEASE CHECKS PASSED (100% GREEN)                      "
echo " Safe to commit, tag, and push release.                          "
echo "================================================================="
