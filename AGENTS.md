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

---

## 6. Authorial Voice for Documentation, Papers, and Release Collateral

Every word of prose committed to this repository follows the authorial voice of Richard Albright: the README, `docs/`, `CHANGELOG.md`, blog posts, benchmark reports, academic papers, and release notes. The authoritative guide is `../benolabs/.agents/skills/authorial-voice/SKILL.md` and the always-on steering rule is `../benolabs/.agents/rules/authorial_voice.md`. When those files and this summary disagree, read the guide, then ask.

### 6.1 Core Pillars

1. **Dual audience clarity.** Assume two readers at once: an engineer who will run the code, and a technical decision maker who signs the infrastructure bill. Open each major section with a plain statement of what happens and why it matters in operational terms (cost, memory ceilings, cluster count, vendor freedom), then follow immediately with the schema, the query, the benchmark controls, or the math.
2. **Empirical, never promotional.** Every claim carries a number, a mechanism, or a physical constraint. If a sentence could survive with the numbers deleted, it is an advertisement and does not belong.
3. **Systems grounded mechanics.** Name the actual components: Apache Iceberg, Apache Parquet, Puffin sidecars, Apache DataFusion, Arrow `RowSelection`, AVX2 kernels, cgroup limits. "The storage layer" is a failure to look.
4. **Honest about trade offs, negative results, and boundaries.** State what the system does not do, where performance inverts, and which scale is out of scope. A reader trusts a paper that concedes a loss more than one that reports only wins.
5. **Technical marketing woven throughout.** Never confine persuasion to the top of a document. Put it in section transitions, architecture comparisons, and benchmark callouts, and make it argue through architectural contrast rather than adjectives.

### 6.2 Punctuation and Mechanical Rules

- **No em dashes.** Do not join words with `---` in LaTeX or an em dash in prose. Split the sentence, use a colon, use parentheses, or use a contrastive connective. The construction reads as machine written.
- **No hyphenated compound modifiers between words.** Write `two stage`, `high dimensional`, `in process`, `single node`, `read only`, `build time`, `float point` compounds as separate words or restructure the sentence. Numeric and named compounds keep their hyphens (`32-KB`, `128-D`, `512-MiB`, `Walsh-Hadamard`, `Johnson-Lindenstrauss`, `deletion-vector-v1`, `--strict-hardware`). The tell the reader notices is not the hyphen itself, it is the pile of coined compounds standing in for a verb.
- **No tilde approximations.** Do not write `~GB`, `~1.3 MB`, or `$\sim 40$`. Print the figure the harness produced, state the bound, or describe the order of magnitude in words. The LaTeX tie in `512~MiB` is spacing and stays.
- **Define every acronym at first use.** Expand it in the abstract and again in the body before the abbreviation appears: Fast Walsh-Hadamard Transform (FWHT), maximum inner product search (MIPS), 8 bit scalar quantization (SQ8), Hierarchical Navigable Small World (HNSW), Product Quantization (PQ). A reader who meets `FWHT` cold stops reading.
- **One name per concept.** Do not alternate between `TQ8` and `BQ8` for the same quantizer, or between `OOM` and `out of memory` inside one section.
- **Active voice, finding first.** `We ingest the corpus and partition chunks by section boundaries`, not `The corpus is ingested and chunks are partitioned`. State the result, then the mechanism or measurement that supports it.

### 6.3 Forbidden Vocabulary and Replacements

| Avoid | Use |
|---|---|
| "seamless", "effortless", "magical" | Zero copy Arrow record batching; `pip install benostreamdb` with no external service |
| "blazing fast", "revolutionary", "game changing" | Sub millisecond retrieval; AVX2 SIMD distance kernels at 1,700 QPS on eight cores |
| "massive I/O reduction" | 32 KB Parquet pages with Arrow `RowSelection` cutting rerank I/O from 512 MB to 1.3 MB |
| "next generation database" | Vector and graph engine over Apache Iceberg Puffin sidecars |
| "best in class vector search" | HNSW with BenoQuant TQ8 isotropic scalar quantization |
| "massive cost savings" | Eliminates the three system tax and the multi hop network round trips |

### 6.4 Scope Discipline for Release Text

State what a release does and does not contain before describing what it improves. Name the boundary explicitly, for example that the engine is single node with no distributed consensus, no cross-table multi statement transactions, and no automated failover, and that 100M vectors is the empirical single node boundary at a 4.8 times working set to RAM ratio. A preview note that hides its own scope is worse than no note.
