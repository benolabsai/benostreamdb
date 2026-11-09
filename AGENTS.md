# BenoStreamDB Agent Steering & Release Guidelines

All AI agents, pairing assistants, and contributors operating in this repository must strictly adhere to the following release and verification engineering rules.

---

## 1. Mandatory Pre-Release Steering Gate

**NEVER tag, release, or push commits to GitHub without executing the local pre-release suite:**

```bash
./scripts/pre_release_check.sh
```

This script mirrors all required gates from `.github/workflows/ci.yml` and `.github/workflows/release.yml`:
1. `cargo fmt --check` (Rust code style & formatting)
2. `cargo clippy --all-targets` (Clippy zero-warning check under Rust 1.98.0)
3. `cargo check --all-targets` & `cargo check --bins` (Compilation of library, tests, and binaries)
4. `cargo doc --no-deps --features java` (Documentation link integrity)
5. `version-check` (Single-source version assertion across all connectors, POMs, and pyproject files)
6. Python regression tests (`tests/python/test_vector_string_conformance.py`)

If any check fails, **stop immediately and fix the underlying issue locally**. Never push code expecting CI to catch it.

---

## 2. Git Pre-Push Hook

The repository provides a tracked git pre-push hook in `scripts/git-hooks/pre-push`.
Install it in any local checkout:

```bash
./scripts/install_hooks.sh
```

The hook automatically intercepts `git push` (for both branches and tags) and blocks the push if `./scripts/pre_release_check.sh` fails. **Never bypass this hook using `--no-verify`.**

---

## 3. Rust Formatting & Lints

- Always run `cargo fmt` after editing any Rust file in `src/`, `server/`, or `contrib/`.
- Ensure Rust code compiles cleanly with Clippy on toolchain `1.98.0` (`rustup run 1.98.0 cargo clippy --all-targets`).

---

## 4. Single Source of Truth for Versions

The single authoritative version is defined in:
```toml
# Cargo.toml
[workspace.package]
version = "X.Y.Z"
```

When bumping versions for any release:
1. Update `[workspace.package].version` in `Cargo.toml`.
2. Update inter-crate dependency pins in:
   - `server/flight_sql/Cargo.toml` (`benostreamdb = { version = "X.Y.Z", ... }`)
   - `contrib/benostreamdb-search/Cargo.toml` (`benostreamdb = { version = "X.Y.Z", ... }`)
3. Update connector metadata:
   - `spark-benostreamdb/pom.xml` (`<revision>X.Y.Z</revision>`)
   - `trino-benostreamdb/pom.xml` (`<revision>X.Y.Z</revision>`)
   - `dbt-benostreamdb/pyproject.toml` (`benostreamdb>=X.Y.Z`)
   - `dbt-benostreamdb/dbt/include/benostreamdb/dbt_project.yml` (`version: X.Y.Z`)
4. Verify dynamic versioning in all Python packages (`pyproject.toml`, `server/flight_sql/pyproject.toml`, `contrib/benostreamdb-search/pyproject.toml`, `contrib/benostreamdb-mcp/pyproject.toml`).
5. Run `./scripts/pre_release_check.sh` to assert full synchronization.

---

## 5. In-Process SQL & Registration

- `Table.sql(query)` binds the table directly to the default alias `'t'` (`SELECT ... FROM t`).
- To query tables by named identifiers (`SELECT ... FROM events`), register the table in an in-process `Session`:
  ```python
  session = bsdb.Session()
  session.register("events", table)
  df = session.sql("SELECT * FROM events ...")
  ```
