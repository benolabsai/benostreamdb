# No-Panic Policy

BenoStreamDB is a long-running database process. A single `unwrap()` on an
unexpected `None`/`Err` in a request path, an ingest loop, or a decode path
crashes the whole daemon: in-flight queries are dropped, the process restarts,
and the WAL has to recover. Panics are therefore treated as defects in
production code, not as an acceptable error strategy.

## Scope

| Code | Rule |
| --- | --- |
| Production library code (`src/**` compiled without `cfg(test)`) | No `unwrap()` / `expect()` / `panic!` except documented invariants |
| Production binaries (`src/bin/{bsdb,gateway,iceberg_rest,benostreamdb-admin}.rs`, `benostreamdb-search/src/main.rs`) | Same rule |
| `#[cfg(test)] mod tests` blocks | Exempt — tests may unwrap freely |
| `tests/**` integration crates | Exempt |
| `build.rs` | Build scripts are exempt in spirit, but `build.rs` was converted to return `Result` anyway |

## The rule

Prefer, in order:

1. **Propagate**: `?` with the native error type (`HyperstreamError`,
   `DataFusionError`, `anyhow::Error`) via `ok_or_else` / `context`.
2. **Make the function total**: where Arrow's `data_type()` (or an equivalent
   guard) already proves the downcast cannot fail, use
   `.map(|a| ...).unwrap_or(<neutral>)` or `if let ... else { <neutral> }`
   instead of `unwrap()`. A violated invariant then degrades to `null` / `false`
   / `skip` rather than panicking.
3. **Documented invariant**: only where the API genuinely has no infallible
   constructor and failure is a programming error caught at startup, add a
   narrowly scoped `#[allow(clippy::unwrap_used, clippy::expect_used)]` with a
   justification comment. Every such site must also be listed below.

## Enforcement

### Ratchet (`scripts/no_panic_check.sh`)

Because `src/lib.rs` and `benostreamdb-search/src/lib.rs` already carry
`#![deny(warnings)]`, flipping the restriction lints to hard errors immediately
would fail the build on every not-yet-remediated site. Until remediation
completes, enforcement is a **ratchet**:

```bash
bash scripts/no_panic_check.sh           # fail if production panic sites grew
bash scripts/no_panic_check.sh --write   # lower the baseline after a fix
```

The script runs `cargo clippy --no-deps --lib` with
`-D clippy::unwrap_used -D clippy::expect_used -D clippy::panic`. Using `--lib`
(not `--all-targets`) compiles the crate with `cfg(test)` **off**, so test-only
unwraps are excluded automatically. It compares the diagnostic count against
`scripts/no_panic_baseline.txt` and fails on an increase.

Run it in CI as the `no-panic` job in `.github/workflows/ci.yml`.

### Hard gate (staged)

Once the baseline reaches the residual-invariant set below, the ratchet is
replaced by a hard lint gate. The attributes are already in place, staged behind
the `no-panic` cargo feature:

```rust
#![cfg_attr(
    all(not(test), feature = "no-panic"),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]
```

At that point CI adds `--features no-panic` and the feature becomes the
permanent enforcement mechanism; the ratchet script can be retired.

## Baseline and inventory

Initial production baseline (measured before remediation): **289** sites
(247 core library + 42 search library). **Current baseline: 0** — the library
surfaces of both crates are clean and the hard gate is enforced in CI.

Two module-scoped exemptions remain, both justified in source:

| Exemption | Sites | Why |
| --- | ---: | --- |
| `src/core/index/hnsw_rs/**` | 69 | Vendored `hnsw_rs` (dual MIT/Apache-2.0). `unwrap`/`expect`/`panic` here assert graph invariants that hold by construction in the builder and the inner search loop, where threading `Result` would add a branch to the innermost distance comparison. `src/core/index/hnsw_rs/mod.rs` carries `#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]` with the rationale. |
| `src/telemetry/metrics.rs` | 13 | `prometheus` metric constructors for static definitions; the crate has no infallible API. See the allowlist below. |

Everything else — the search crate, graph UDFs, the storage/read path, the
ingest and index-build path, merge/compaction/catalog, the algorithms, the GPU
backends, the Python/FFI bindings, and `src/lib.rs` — was converted to `?`, made
total, or narrowly allowlisted at function scope (see the allowlist section).

### Deferred

- **`hnswio` file I/O.** A truncated or corrupt index file can fail a
  `read_exact`/`write_all` inside the vendored dump/load path, which the
  module-level exemption therefore still covers. Converting those entry points to
  `Result` changes upstream signatures; it is tracked here as the only
  user-reachable panic path left in the library.
- **Binary targets.** `src/bin/*` and `benostreamdb-search/src/main.rs` are not
  covered by this ratchet: `cargo clippy --bin <name>` re-lints the same-package
  lib and does not reliably emit the bin's own diagnostics. They are remediated
  directly and are covered by the staged feature once bin-root attributes land.

## Residual-invariant allowlist

These sites are permitted to keep a documented `#[allow]` because the API has no
infallible form and failure is a startup-time programming error, not a runtime
condition:

- **`src/telemetry/metrics.rs`** — `prometheus` metric constructors
  (`IntGauge::new`, `IntCounterVec::new`, …) return `Result` and are only
  fallible on an invalid name or duplicate registration. All names/help strings
  are static constants. Failure happens once at startup before the server binds
  and is unreachable from a request. Guarded by
  `#[allow(clippy::unwrap_used, clippy::expect_used)]` on `Metrics::new`.

- **Compile-time regex literals** (e.g. `Regex::new(r"...")`) — the pattern is a
  literal, so construction cannot fail. These may use `expect` with a message.

- **`chrono::NaiveDate::from_ymd_opt(1970, 1, 1)`** — replaced by
  `NaiveDate::default()` where possible (it *is* 1970-01-01), which is
  infallible.

- **`src/lib.rs` jemalloc ctl** — `tikv_jemalloc_ctl::epoch::mib()` is a
  process-lifetime singleton; failure indicates a broken allocator build.

- **`src/core/table/state.rs` `Table::runtime()`** — `rt` is set by every public
  constructor; the `expect` asserts a structural invariant. Returning
  `Option`/`Result` would ripple through every synchronous caller just to
  express "this cannot happen". Guarded by a function-level
  `#[allow(clippy::expect_used)]`.

- **`src/python/helpers.rs` `TOKIO_RUNTIME`** — `Runtime::new()` has no
  infallible form and fails only if the OS cannot give the process a reactor and
  worker threads. If that happens the binding layer is unusable, so failing once
  at first use with a clear message is correct. Guarded by
  `#[allow(clippy::expect_used)]`.

Anything not in this list should be converted to `?`, made total, or the
allowlist entry justified in review.
