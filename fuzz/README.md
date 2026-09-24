# Fuzzing

Coverage-guided fuzzing for BenoStreamDB's untrusted-input parsers, using
[`cargo-fuzz`](https://github.com/rust-fuzz/cargo-fuzz) (libFuzzer). The goal is
the GA requirement that **malformed input returns an error, never crashes the
process** — the parser and request-body surfaces are the places where bytes from
outside the system first meet the engine.

## Targets

| Target | Surface |
| --- | --- |
| `parse_dense` | pgvector dense literal `'[1.0, 2.0, 3.0]'::vector` |
| `parse_sparse` | sparse literal `{1:0.5, 10:0.3}` with dimension bounds/duplicate checks |
| `parse_binary` | bit/hex literals `B'10110101'`, `'\xB5'` with/without expected bit count |
| `sql_rewriters` | `strip_partitioned_by` + `rewrite_sql_string` (pgvector operators/casts) |
| `qdrant_request_body` | Qdrant-compatible request bodies (`#[serde(untagged)]` enums) |

The Elasticsearch `_search`/`_bulk` bodies are not covered yet: the request types
live in private handler modules. Exposing a `#[doc(hidden)] pub` parse hook for
those bodies is the natural next target.

## Running

Requires a nightly toolchain (libFuzzer) and `cargo-fuzz`:

```bash
rustup toolchain install nightly
cargo install cargo-fuzz

cd fuzz
cargo fuzz run parse_dense                       # indefinite
cargo fuzz run parse_dense -- -max_total_time=300  # 5-minute budget
cargo fuzz run sql_rewriters -- -rss_limit_mb=4096
```

`fuzz/Cargo.toml` is its own workspace root, so fuzzing never pulls
`libfuzzer-sys` into the main workspace build.

`fuzz/Cargo.lock` is a copy of the root lockfile. It is kept so the fuzz
workspace resolves the same dependency versions as the main build — a fresh
resolution pulls `aws-*` versions that require a newer `rustc` than the pinned
toolchain. Refresh it with `cp Cargo.lock fuzz/Cargo.lock` after dependency
changes.

## Corpus and regression promotion

- `corpus/<target>/` holds the seed inputs; libFuzzer grows them as it runs.
- A crash is written to `artifacts/<target>/`. When the `Fuzz` workflow (see
  `.github/workflows/fuzz.yml`) finds one:
  1. Reproduce locally with `cargo fuzz run <target> artifacts/<target>/<crash>`.
  2. Minimise with `cargo fuzz tmin <target> artifacts/<target>/<crash>`.
  3. Move the minimised input into `corpus/<target>/` so it is replayed on every
     future run, and add a `#[test]` in the owning module asserting the input is
     rejected rather than panicking.
