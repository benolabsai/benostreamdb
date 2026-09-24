# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.10.0] - 2026-09-24

### Changed
- **Resource limits are now cgroup-aware and always active (serverless-safe).**
  The ingest RAM high-water mark previously defaulted to 80% of the *host's*
  physical memory (`/proc/meminfo`), so a 4 GB container on a 32 GB host got a
  ~25 GB ceiling and OOM-killed instead of throttling. `core::resources` now
  resolves the memory actually available to the process — cgroup v2
  `memory.max`, then cgroup v1 `memory.limit_in_bytes`, then host RAM, then a
  conservative 4 GiB fallback — and every memory limit is derived from it:
  `BSDB_MAX_INGEST_RAM_GB` and `BSDB_INGEST_MEMORY_BUDGET_GB` default to 80% of it,
  and `BSDB_INDEX_BUILD_CONCURRENCY` scales as one build per ~8 GiB (capped at the
  CPU count, at least 1). The "off unless configured" semantics are gone: an
  unset or non-positive variable now means "use the derived default", never
  "disable the guard". `BSDB_MIN_FREE_DISK_GB` defaults to 1 GiB, and
  `BSDB_INDEX_BUILD_CONCURRENCY=0` no longer re-enables unbounded fan-out.
  `HeapTrimPolicy::from_budget_or_env` always returns a policy. See
  `RESOURCE_LIMITS.md` for how the knobs interact, with worked 4 GB → 128 GB
  examples.
- **`bincode` 1.3 → 2.x.** bincode 1.3.3 is unmaintained (RUSTSEC-2025-0141).
  The vendored HNSW dump/load path now uses `bincode::serde` with
  `bincode::config::legacy()`, which is byte-compatible with 1.3, so existing
  index dumps still load. The yanked `chacha20` (0.10.1 → 0.10.2) and `spin`
  (0.9.8 → 0.9.9) transitive crates were updated. The unmaintained `paste`
  (RUSTSEC-2024-0436) remains, pulled in transitively by `datafusion-common`
  52.x; it is a compile-time proc-macro with no runtime surface and is
  documented in `deny.toml`.
- **`cargo fmt --all` applied** across the workspace.
- **Memory guards now size from *available* memory, not total.** The value that
  derives `BSDB_MAX_INGEST_RAM_GB`, `BSDB_INGEST_MEMORY_BUDGET_GB`, and
  `BSDB_INDEX_BUILD_CONCURRENCY` previously used the host's `MemTotal`, so on a
  shared machine the guards were sized for RAM the process could not actually
  get — the demo load was OOM-killed at ~74 GB RSS on a 121 GiB host because the
  derived high-water mark was ~102 GB. `effective_memory_bytes()` now prefers
  `MemAvailable` (Linux) for the host-RAM fallback; a cgroup limit or
  `RLIMIT_AS` ceiling is still used as-is. New `usable_memory_bytes()` exposes
  the value.

### Added
- **Disk-headroom admission limit + a single resource-limits reference.** New
  `core::resources` module: `BSDB_MIN_FREE_DISK_GB` refuses a flush when a local
  table's filesystem is below the threshold, so the failure is a clear error
  *before* any bytes are written rather than a partial segment, and
  `benostreamdb_free_disk_bytes` exposes the sampled headroom. Fail-open when
  the threshold is unset or the free space cannot be queried. Every
  resource/back-pressure knob (`BSDB_MAX_INGEST_RAM_GB`,
  `BSDB_INGEST_MEMORY_BUDGET_GB`, `BSDB_INDEX_BUILD_CONCURRENCY`,
  `BSDB_MIN_FREE_DISK_GB`, and the CPU/concurrency bounds) is now documented in
  one place: `RESOURCE_LIMITS.md`.
- **No-panic gate enforced for the library.** The production `unwrap()`/
  `expect()`/`panic!` count across both crates went **289 → 0** (247 core library
  + 42 search library remediated). With the baseline at zero the staged
  `no-panic` cargo feature is now live: `scripts/no_panic_check.sh` runs the
  ratchet *and* `cargo clippy --features python,no-panic`, which turns the three
  restriction lints into hard errors for non-test code. The only exemptions are
  the vendored `hnsw_rs` subtree (graph invariants in the inner search loop) and
  `src/telemetry/metrics.rs` (static `prometheus` definitions with no infallible
  constructor), both module-scoped and documented in `NO_PANIC_POLICY.md`.
- **Fuzzing workspace (`fuzz/`).** Coverage-guided fuzzing (cargo-fuzz /
  libFuzzer) for the untrusted-input parsers: the dense, sparse, and binary
  vector literal parsers, the SQL string rewriters (`strip_partitioned_by`,
  `rewrite_sql_string`), and the Qdrant-compatible request bodies (whose
  `#[serde(untagged)]` enums are a classic pathological-input surface). Seed
  corpora are committed; the `Fuzz` workflow runs each target for a 5-minute
  budget on push and uploads crash artifacts. `fuzz/` is its own workspace
  root so `libfuzzer-sys` never enters the main build, and it shares the root
  lockfile to stay on the pinned toolchain. See `fuzz/README.md`.
- **Soak / stress suite (`tests/stress/`).** A maintained replacement for the
  one-off `test_oom.py` script, asserting graceful degradation under saturation:
  a low ingest RAM high-water mark must pause and resume rather than OOM
  (`test_memory_backpressure.py`), gated index builds must not starve queries
  (`test_cpu_saturation.py`), many commit cycles must stay durable
  (`test_disk_io.py`), and a mixed write/search/read churn must keep memory
  bounded over a wall-clock budget (`test_mixed_soak.py`). Skipped unless
  `BSDB_STRESS=1`; run on push by the `Soak` workflow. A Rust long-running soak
  harness (`tests/soak.rs`, `#[ignore]`) provides the same coverage in-process.
- **Event-driven ingest back-pressure.** The ingest RAM high-water mark
  (`BSDB_MAX_INGEST_RAM_GB`) previously blocked writers with a flat 500 ms sleep
  loop that re-read `/proc/self/status` on every iteration — wasting time when
  memory had already been freed and delaying resumption when it had not. Writers
  now wait on a `tokio::sync::Notify` signalled by the tasks that release memory
  (a segment index build finishing, or the ingest loop trimming the heap) with a
  250 ms bounded fallback poll, so they resume the instant memory is reclaimed.
  New Prometheus metrics expose the sampled RSS gauge, the back-pressure pause
  count and duration distribution, and index-build gate wait time
  (`benostreamdb_ingest_rss_bytes`,
  `benostreamdb_ingest_backpressure_pauses_total`,
  `benostreamdb_ingest_backpressure_pause_seconds`,
  `benostreamdb_index_build_gate_wait_seconds`). Covered by
  `memory_reclaimed_notification_wakes_blocked_writer`.
- **No-panic ratchet for production code paths.** `scripts/no_panic_check.sh`
  counts `unwrap()`/`expect()`/`panic!` sites in production (non-test) code via
  `cargo clippy --lib` with the restriction lints forced on, and fails if the
  count grows beyond `scripts/no_panic_baseline.txt`. Test code (`#[cfg(test)]`
  blocks and the `tests/` crates) is excluded automatically. Wired into CI as
  the `no-panic` job. A staged `no-panic` cargo feature already carries the
  `#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used,
  clippy::panic))]` attributes for when remediation completes. See
  `NO_PANIC_POLICY.md`.

### Fixed
- **The `bsdb-search` server binary could not be built.** `benostreamdb-search`'s
  `main.rs` installed `dhat::Alloc` as the global allocator while the
  `benostreamdb` library installs jemalloc on Linux, so linking failed with
  "the `#[global_allocator]` in this crate conflicts with global allocator in:
  benostreamdb". Heap profiling is now opt-in behind a `dhat-heap` feature and
  the default build works. CI now runs `cargo check --bins` (and the search
  crate's bin), which would have caught this: clippy does not lint bin targets
  when a same-package lib exists, so nothing was compiling them.
- **Panics removed from the network-server binaries.** `src/bin/gateway.rs`
  unwrapped a user-supplied filter (`QueryFilter::parse(...).unwrap()` in the
  `POST /query` handler) — a malformed filter now returns 400. Also fixed: the
  gateway's store creation, mock batch construction and listener bind; the
  iceberg_rest server's bind, graceful-shutdown and metrics response;
  `bsdb.rs`'s `print_batches`; and the search server's bind/parse/signal-handler
  `expect`s. The four production bins and the search bin now carry the
  `cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used,
  clippy::panic))` gate.
- **Panics removed from the search request path.** `benostreamdb-search` had 42
  production `unwrap()` sites: `handlers/search.rs` unwrapped JSON map lookups
  and every Arrow→JSON downcast, and `handlers/{bulk,qdrant,docs,metrics}.rs`,
  `infer.rs` and `state.rs` unwrapped request data. JSON lookups now use
  `ok_or_else`/`if let`, and the Arrow downcasts (guaranteed by the
  `data_type()` match) are total — a violated invariant yields `null` instead of
  crashing the handler. The crate is now clean.
- **Panics removed from the graph UDF compute path.** All 14
  `src/core/sql/graph_udf/*.rs` accumulators unwrapped `states[i]` downcasts in
  `merge_batch`/`update_batch`; they now return
  `DataFusionError::Execution`. These run inside SQL `GROUP BY` for graph
  analytics, so a malformed state previously aborted the query with a panic.
- **Panics removed from the manifest decode, reader filter, and segment paths.**
  `ManifestValue::from_array` (Arrow type-invariant downcasts), the inverted-index
  filter's key downcasts, and the segment writer's path handling no longer
  `unwrap()`; the filter's `NaiveDate::from_ymd_opt(1970, 1, 1).unwrap()` sites
  became `NaiveDate::default()`.
- **Panics removed from the write/index-build path.** `table/write.rs` schema
  `index_of` lookups, `index/csr_graph.rs` fixed-size reads and id
  binary-searches, and `index/build_graph.rs` edge downcasts are now total.
- **`build.rs` returns `Result`** instead of unwrapping `OUT_DIR`/file writes.
- **Unbounded concurrent index builds OOM-killed large loads.** The write path
  spawns one segment index build per flush onto the tokio runtime, whose worker
  count is the CPU count. On a 32-core workstation that meant up to 32 builds in
  flight, each holding its segment's vectors plus the HNSW/IVF/quantizer
  structures (several GB at the demo's 1 GB flush size) — the Wikipedia load hit
  **105 GB RSS** and was OOM-killed, and before that thrashed so badly that
  per-chunk time exploded 43 min → 8.65 h. Builds now go through a shared
  semaphore (`Table::index_build_gate`, default 2, `BSDB_INDEX_BUILD_CONCURRENCY`)
  that back-pressures the writer; the bounded working set also removes the
  page-cache eviction that made the unbounded case slow, not just fatal.
- **`add_index` re-indexed every prior segment on each call (O(n²) demo-load
  OOM).** `add_index` on a non-empty table unconditionally triggered
  `backfill_indexes_async`, which rebuilt the indexes of *every* manifest entry
  via an unbounded `futures::future::join_all` — with no "already indexed" skip
  and no per-segment build gate (the single permit gated the whole task, not the
  fan-out). The demo's chunked load calls `add_index` in every fresh process, so
  each chunk re-indexed all previously loaded data: per-chunk time grew
  293s → 885s → 2071s and peak RSS climbed until the process was killed. Backfill
  now skips segments that already carry the required `(column, index_type)` pair
  and bounds in-flight builds with `buffer_unordered(index_build_concurrency)`,
  taking the shared build gate per segment. `prepare_demo.py` also sets the
  default device once and only applies `add_index` for columns not already
  configured (the config is restored from the manifest on open). A full
  51.8M-row node load now completes in **~45 min** (est.) at a flat **~46–56 s
  per M rows (~20k rows/s)**, versus the escalating 293s → 885s → 2071s that
  preceded the kill — see `examples/web_ui/README.md`.
- **Heap not returned to the OS at flush/build boundaries.** glibc keeps freed
  memory in per-thread arenas, so the HNSW/TQ builders' millions of small
  allocations ratcheted RSS toward the sum of every arena's high-water mark. Both
  the flush path and the end of each background index build now call
  `memory::trim_if_over_budget` (budget via `BSDB_INGEST_MEMORY_BUDGET_GB`,
  default 8 GiB) so freed arena pages are released before the next build starts.
- **`simple_kmeans` allocated a `Vec` per vector** in the capacity-capped final
  assignment. Over millions of vectors that churned and fragmented the heap. It
  now reuses a thread-local scratch buffer, keeping the strict capacity cap.

### Security
- **PR security checks + a local AI diff review.** A new `PR Security Checks`
  workflow runs `cargo-deny` (advisories, bans, licenses, sources) and gitleaks
  on every pull request — free and secret-free. For deeper review,
  `scripts/ai_pr_review.py` is a local tool that reads the standard `OPENAI_*`
  environment variables (OpenRouter by default) and asks an LLM to look for
  *injected* vulnerabilities in a diff (backdoors, weakened checks, credential
  exfiltration, unsafe deserialization, injection, disabled security controls,
  supply-chain changes), printing findings and optionally posting a PR comment.
  It is intentionally not run in CI, so no secrets are stored.
- **`SECURITY.md` right-sized for a one-person open-source project.** The bug
  bounty program was removed, the response SLA is now explicitly best-effort,
  and only the latest release is supported (fast-moving project).

## [0.9.1] - 2026-09-23

### Fixed
- **Metal distance kernels read past the end of the vector buffer.** The
  dispatch covers `ceil(n/256)*256` threads, but the kernels indexed rows by
  `thread_position_in_grid` with no bounds guard, so up to 255 threads read out
  of bounds. Added an `n_vectors` argument + guard to all six kernels, and
  aligned the Hamming (`!=`) and Jaccard (`> 0.0`) comparisons with the CPU/CUDA
  definition. Covered by the new macOS CI job (Apple Silicon).
- **rustdoc: `VEC_TMP_MAGIC` linked to a private item**, breaking
  `cargo doc --features python,wgpu,enterprise,java`. Now plain text.
- **CUDA distance kernels launched with the wrong grid and no shared memory.**
  `CudaBackend::compute_distance` used `LaunchConfig::for_num_elems`, which packs
  rows into 1024-thread blocks and sets `shared_mem_bytes: 0`, but the kernels
  use one block per row with an `extern __shared__` reduction — an illegal
  memory access, not a wrong answer. Now launches `n_vectors` blocks of 256
  threads with the shared memory sized for the block. Caught by the new
  cross-backend harness.
- **CUDA JIT now works with pip's CUDA 13 wheels.** cudarc 0.13's nvrtc loader
  probes a fixed candidate list (`libnvrtc.so`, `libnvrtc64*.so`,
  `libnvrtc.so.{12,11,10,1}`) that predates CUDA 13, so `nvidia-*-cu13` wheels
  (`libnvrtc.so.13`) were never found: the probe panicked and the GPU silently
  fell back to CPU. `core::index::nvrtc` now resolves *whatever* `libnvrtc.so*`
  is installed — any version, any layout (`BSDB_NVRTC_PATH`, the interpreter's
  `site-packages`, `CUDA_HOME`/`CUDA_PATH`, `PYTHONPATH`, `LD_LIBRARY_PATH`,
  system paths) — preloads the `libnvrtc-builtins` companion with `RTLD_GLOBAL`,
  and compiles the embedded `.cu` sources itself, handing the PTX to cudarc. No
  re-exec, no env-var prefix, no shim script (`scripts/create_cuda_shims.sh`
  removed).
- **Demo load resume no longer duplicates rows.** `scripts/prepare_demo.py`
  resumed from the rounded-down chunk boundary on the assumption that chunks
  commit atomically. They don't: the write path spills to a real commit whenever
  the buffer exceeds `BENOSTREAM_CACHE_GB` (default 1 GB), so a killed chunk
  leaves partial rows committed and the round-down re-wrote them. Resume now
  continues from the exact committed row count (`_resume_offset`), guarded by a
  regression test.

### Added
- **GPU acceleration for sparse vectors (dense-conversion path)**: batched
  `bsdb.sparse_l2_batch` / `bsdb.sparse_cosine_batch` /
  `bsdb.sparse_inner_product_batch` convert a sparse query plus N sparse vectors
  to dense and dispatch to the existing dense GPU kernels. Backend-agnostic, so
  it works on every backend; it pays off once the batch clears
  `GPU_DISPATCH_THRESHOLD`. Equivalence with the sparse CPU reference is covered
  by `sparse_dense_equivalence_tests`.
- **GPU-accelerated packed-binary distance (CUDA + WGPU)**: `GpuBackend::compute_binary_distance`
  plus packed-u8 Hamming/Jaccard kernels for CUDA (`hamming_packed.cu`,
  `jaccard_packed.cu`) and WGPU (`wgpu_binary_kernel.wgsl`, covering AMD/Intel
  via Vulkan). New batched Python API `bsdb.hamming_distance_batch` /
  `bsdb.jaccard_distance_batch` (one packed query vs N packed vectors), plus
  Metal (`mps/hamming_packed.metal`, `mps/jaccard_packed.metal`) — verified by
  the new `metal` CI job on Apple Silicon. Cross-backend harness passes on an
  RTX 3090 (`["cpu", "cuda", "wgpu"]`) and on macOS CI (`["cpu", "mps"]`).
- **Cross-backend GPU correctness harness**: `cross_backend_matches_cpu_all_metrics`
  runs the same vectors through every *available* backend (CUDA, Metal, WGPU)
  and asserts agreement with the **CPU reference (the gold source)** within
  tolerance. Backends absent from the machine are skipped, so the same test runs
  everywhere. Verified on an RTX 3090 with `["cpu", "cuda", "wgpu"]` across
  L2/Cosine/IP/L1/Hamming/Jaccard.
- **Native ingest orchestrator (A4)**: `Table::ingest_async` /
  `table.ingest(paths, chunk_rows, parallelism, index_all, resume)` plans parquet
  inputs into row-range work units, runs a bounded `buffer_unordered(parallelism)`
  pool where each worker builds a *private* segment (data + indexes) concurrently,
  and commits completed segments through the OCC manifest CAS. Completed units are
  recorded in a `_ingest_state.json` sidecar so an interrupted load resumes at the
  unit boundary. Returns a report (`units_total/skipped/committed`,
  `rows_ingested`, `segments`). `IngestOptions::compact_after` runs
  `rewrite_data_files` at the end so segment/manifest counts stay bounded.
  CLI: `bsdb table ingest --uri … --input … [--plan] [--chunk-rows N]
  [--parallelism N] [--index-all] [--compact]`, with `--row-start/--row-end` as
  the serverless thin-runner mode (`Table::ingest_range_async`).
  **Multi-format inputs**: `.parquet` (row-range units) plus `.csv`, `.json` /
  `.ndjson`, and `.arrow` / `.ipc` / `.feather` (one unit per file, streamed
  whole-file with schema inference) — all via Arrow readers already in the tree.
- **Ingest memory discipline**: `core::memory::HeapTrimPolicy` returns freed
  heap pages to the OS (`malloc_trim`) at work-unit boundaries once RSS exceeds
  a budget, so long-lived in-process loads no longer ratchet toward the sum of
  every glibc arena's high-water mark. Wired through
  `IngestOptions::memory_budget_bytes`, the `BSDB_INGEST_MEMORY_BUDGET_GB` env
  var, `bsdb table ingest --memory-budget-gb`, and
  `table.ingest(..., memory_budget_gb=…)`. Allocator evaluation recorded in
  `core::memory`: glibc + `malloc_trim` chosen; jemalloc deferred; mimalloc
  rejected (static-TLS under pyo3).
- **Multi-machine ingest (A4)**: `WorkCoordinator` trait + `ObjectStoreCoordinator`
  backend (`core::table::coordinator`) — lease-based work stealing over the
  object store, reusing `FileBasedLock` (CAS claim + heartbeat + expiry-steal).
  `Table::ingest_coordinated_async` claims units dynamically (build in parallel,
  commit serially via the OCC CAS, release-on-failure so another node retries);
  a dead node's lease expires and its unit is stolen. Surfaces:
  `bsdb table ingest --coordinate [--lease-ttl-secs N]` and
  `table.ingest(..., coordinate=True, lease_ttl_secs=300)`. No broker, no etcd,
  no Raft cluster.

### Removed
- **Dead OpenCL kernels** (`src/core/index/opencl/*.cl`) — no `OpenClBackend`
  was ever wired into `ComputeBackend`.

### Performance
- **IVF clustering is now balanced (k-means++ seeding + capacity cap).** The
  index build slowed down over successive ingest chunks because `simple_kmeans`
  seeded centroids at random, leaving dense regions uncovered: one 750k-row
  segment produced 173 clusters where the largest held ~25x the mean (604 KB vs
  443 B of row-id mappings). Since the per-bucket HNSW build is superlinear in
  cluster size, those few oversized clusters dominated Pass 3. Centroids are now
  seeded with **k-means++ (D² sampling)** and the final assignment is
  **capacity-capped** at 1.5x the mean, spilling overflow points to their
  next-nearest cluster. Regression test
  `test_kmeans_clusters_are_balanced_on_skewed_data` asserts no cluster exceeds
  the cap on a deliberately skewed dataset.
- **Out-of-core HNSW-IVF build is now parallel and skips a full file re-scan.**
  Two changes to `HnswIvfIndex::build_from_file`:
  1. The per-bucket HNSW graphs (Pass 3) were built in a **sequential** loop —
     the dominant cost of a large index build. Buckets are independent and
     bounded in size, so they now build with `rayon` (`into_par_iter`), using
     all cores.
  2. The temp vector file written by `build_vector_index` now carries a
     self-describing header (`magic` + `dim`); the builder derives the vector
     count from the file length instead of re-reading the whole file just to
     count vectors (a multi-GB read per segment). Legacy header-less files still
     work via the old full-scan path.
  Tuning knobs (no rebuild): `BSDB_HNSW_N_LISTS`, `BSDB_HNSW_M`,
  `BSDB_HNSW_EF_CONSTRUCTION`.

## [0.9.0] - 2026-09-23

### Fixed
- **Statistics pruning was inert — four independent faults, all fixed.** Per-column
  min/max never reached the planner, so only partition pruning did any work:
  1. Parquet statistics were **disabled** (`EnabledStatistics::None`) → now `Chunk`.
  2. The Avro manifest writer hardcoded `lower_bounds` / `upper_bounds` /
     `null_value_counts` to `Null`, and the reader rebuilds `column_stats` from
     exactly those three → added `encode_iceberg_value` (mirror of the existing
     `decode_iceberg_value`) and `bounds_avro_values`.
  3. The reader never unwrapped Avro's nullable-union wrapper: `parse_map_int_long`
     / `parse_map_int_bytes` matched a bare `Array`, but a `["null", T]` field
     decodes as `Union(1, Box(Array(..)))`, so every stats map silently parsed as
     `None` → `unwrap_nullable_union`.
  4. `BETWEEN` produced no `QueryFilter` at all (DataFusion does not lower it to
     `>= AND <=`) → added an `Expr::Between` arm.
  Result: `id >= 100` prunes 5/5 segments ("column max < filter min"),
  `id = 15` prunes 4/5, `id >= 40` keeps exactly the matching segment; query
  results unchanged.
- **Binary column statistics pruned matching rows** — the manifest writer
  encodes bounds as raw bytes (`encode_iceberg_value`), but the reader decoded
  `binary`/`fixed` bounds unconditionally to **base64**, so a `blob` column's
  stats came back as `"YmFy"`/`"cXV4"` while the filter literal was `"foo"`.
  `"cXV4" < "foo"` therefore held and the segment was pruned with
  `StatsBelowMin`, dropping the matching row (surfaced by
  `tests/all_types_index_test.rs`). `decode_iceberg_value` now treats
  `binary`/`fixed` symmetrically with `string` — UTF-8 when valid, base64 only
  as a fallback for non-UTF-8 bytes.
- **`Device.auto_detect()` raised instead of falling back** when
  `torch.cuda.is_available()` reported `True` but no usable device existed (no
  GPU, or a missing `libnvrtc`): constructing the CUDA/ROCm/Intel device now
  falls through to native probing and finally CPU rather than propagating the
  error. Previously this failed every GPU-context test on a GPU-less host with
  a CUDA-enabled torch build.
- **AWS Glue `metadata_location` is now authoritative**: Glue is the only
  catalog whose commit API cannot return a metadata location (REST/Nessie
  return it; Hive/JDBC set it directly), so the client must supply one. The
  code reconstructed it from the snapshot's `sequence-number`, which only
  matches the metadata version because `write.rs` happens to set those two
  counters equal. The writer now captures the path `TableMetadata::save_to_store`
  returns and sends an explicit `set-metadata-location` update; Glue prefers it,
  falls back to the old derivation with a warning, and warns rather than
  silently no-op'ing when neither is available.

### Added
- **`vector_search_scored` — ids and scores with no Parquet I/O**: new Python
  method returning `(segment_id, row_id, score)` straight from the HNSW/BM25
  index, for seed discovery, RRF fusion, and candidate reranking. Fetch rows
  only for the winners. (`Table::execute_vector_search_as_scored` and
  `HybridReader::vector_search_index_raw` already existed underneath.)
- **`EXPLAIN` now says *why* segments were pruned**: `QueryPlanner::might_match_condition`
  gained `classify_condition(entry, filter, emit_metrics)`, returning a
  `PruneReason` (partition below-min / above-max / not-in-IN-list, stats
  all-null / below-min / above-max / not-in-IN-list). `explain()` prints a
  ranked breakdown instead of a bare count, and suppresses metric emission so
  diagnostic EXPLAINs don't inflate the operational pruning counters.
- **Sparse query vectors as SQL maps**: the vector-search sort-expr parser now
  accepts a sparse query as a `Map<key, f32>` literal
  (`{'1': 0.5, '10': 0.3}`), in addition to the existing `Struct` form
  (`indices`/`values`/`dim`). Keys may be integer or numeric-string typed;
  entries are sorted by index and de-duplicated. A map cannot carry the vector
  dimension, so it is inferred as `max(index) + 1` — use the Struct form to
  state the true dimension.
- **OR-over-ranges pushdown (A1.6)**: `TableProvider::scan` recognises a
  same-column disjunction of ranges — `(id BETWEEN 1 AND 5) OR (id BETWEEN 50
  AND 55)`, including the lowered `a >= x AND a <= y` form — and unions the
  per-range index bitmaps to skip segments that cannot match before reading
  them. Mixed-column disjunctions are left to DataFusion, and pruning only
  happens when the column is indexed and every range yields a bitmap.
- **Row-value IN-list pushdown on the read path (A1.7)**: new
  `Table::read_pk_filter_async`, plus a `TableProvider::scan` hook, push
  `(c1, c2) IN ((..), (..))` down as a single expression and use the per-column
  inverted indexes to prune non-matching segments. The guard requires *every*
  PK column to be indexed, since a partial index would under-count matches.
- **Sparse vector Arrow IPC serialization (A1.9)**: the `ArrowType` trait was a
  zero-copy `as_bytes(&[Self]) -> &[u8]`, which a variable-length `SparseVector`
  cannot satisfy (its elements live in separate allocations). It is now an owned
  `to_bytes`/`from_bytes` pair with a self-describing little-endian encoding for
  sparse vectors; fixed-width `f32`/`u8` still `bytemuck`-cast.
  `ArrowHnsw::get_vector` returns an owned `Vec<T>`.

### Fixed
- **Score-only vector search failed / read the whole row**: projecting only the
  synthesised `distance` column built an *empty* Parquet projection and failed
  with "must either specify a row count or at least one column". An empty
  projection is now the signal to skip Parquet entirely — the reader emits the
  score straight from the index search using an explicit row count, and
  `fetch_results_by_id` short-circuits the same way. Removed a stray `println!`
  from the HNSW chunk-search hot path.
- **Vector-search column projection was ignored**: `HybridReader::read_rows_by_id`
  took a `columns` parameter but never used it (`_columns`), so a vector search
  projecting two columns still read *every* column of the row from Parquet. It
  now builds a projected schema and reads only what was asked for, falling back
  to the full schema when a requested name is unknown.

### Removed
- Dead code: `src/core/planner/filter.rs` and `src/core/planner/vector_search.rs`.
  Neither was declared as a module (`planner.rs` has no `mod filter;` /
  `mod vector_search;`) and nothing referenced them, so they were orphaned
  duplicates of the live `QueryFilter` / `VectorSearchParams` in `planner.rs`.
  Removing them is a no-op for behaviour.
- **MVCC manifest commits (lock-free)**: the global `commit.lock` is gone from
  `update_schema` — schema/index-spec evolution is now pure optimistic
  concurrency (`PutMode::Create` + rebase). `CommitMetadata::skip_missing_remove_paths`
  lets a writer rebase onto a newer snapshot when a candidate file was
  concurrently removed (compaction uses it, so racing compactions no longer
  abort). New `Table::snapshot_version()` exposes the monotonic snapshot id
  (also on the Python `Table`). Metrics: `benostreamdb_manifest_commit_rebases_total`,
  `benostreamdb_manifest_commit_skipped_removals_total`.
- **Cross-partition compaction**: `PartitionSpec::partition_batch` now applies
  the declared Iceberg transform (`identity`/`void`/`bucket`/`truncate`/`year`/
  `month`/`day`/`hour`) via the canonical `IcebergTransform`, so a merged bin
  re-partitions deterministically. `CompactionOptions::allow_cross_partition`
  (default `true`) controls whether bins may span partitions.

### Fixed
- **`add_primary_key` / `drop_primary_key` panicked from Python** with
  "Cannot start a runtime from within a runtime": the async
  `_validate_pk_uniqueness` called the sync `read_with_columns`, which
  re-enters the Tokio runtime. It now uses the async read path.
- **Index joins returned no rows for non-int/string keys**:
  `extract_distinct_values` only handled `Int32`/`Int64`/`Utf8`, so a join on
  a date, timestamp, float, boolean, or `large_string` key produced an empty
  filter. It now derives keys via `ManifestValue::from_array` (all scalar
  types).
- **`SparseVector(indices, values, dim)` rejected plain Python lists** despite
  documenting them; it now accepts lists and NumPy arrays
  (`PyArrayLike1` + `AllowTypeChange`).
- **Partition values were the raw source value, not the transform result**:
  `partition_batch` ignored `PartitionField::transform`, so `bucket`/`truncate`/
  time partitions stored the untransformed value. Now routed through
  `IcebergTransform` (which also gained `bucket(N)`/`truncate(W)` parenthesis
  syntax and `large_string`/`Utf8View` support).
- **Manifest Avro partition type was derived from the source column**: a
  `bucket` transform yields an `int`, but the writer declared the source type
  (e.g. `string`), so the commit failed with an Avro type mismatch. The type is
  now derived from the transform.
- **Compaction dropped the Hive partition directory**: compacted files were
  written to the table root instead of `col=value/`, silently losing the
  physical partitioning. Compaction now preserves the partition path.
- **`ManifestValue::from_array` ignored `LargeUtf8`**: `large_string` partition
  columns (PyArrow's default) silently became `null`. Also added `Utf8View`,
  `Date32`, `Date64`, and `Timestamp` handling.

### Performance
- **Chunked-ingest memory is now a knob, not a wall**: demo load defaults to
  fresh-process 2M-row chunks (measured 9.5 GB peak per chunk, 2M rows/118 s);
  `--load-chunk-rows 500000` fits serverless tasks (~3-4 GB). ROADMAP gained
  the scale-out items this implies (serverless chunk orchestrator on the OCC
  manifest CAS; long-lived-process allocator discipline).
- **Out-of-core vector serving**: `DiskCache::get_mmap` now applies
  `MADV_RANDOM` to serving mmaps (HNSW graph/vectors, CSR offsets/edges/dict).
  Combined with the existing `use_mmap` default and TQ4 quantization, whole-site
  dense search runs on memory-constrained hosts with a bounded resident set.
- **Demo prep is now laptop-feasible on RAM**: `prepare_demo.py` embed streams
  per row-group (was materializing the whole slice, ~9 GB/worker), resolve joins
  per edge-chunk against one broadcast title map (was ~47 GB), and load deletes
  each embedding shard after ingesting it (bounds peak disk). Embed runs on GPU;
  default model pivoted to `all-MiniLM-L6-v2` (384-d seed-index centroids,
  ~6,000 sent/s on an RTX 3090 → ~2.5 h for 51.8M pages; bge-large-1024 measured
  279 sent/s = 50 h, impractical for a whole-site seed index).

### Fixed
- **Allocator-stranded memory during index-heavy loads**: the whole-site node
  load (32 in-process HNSW-IVF/TQ8 builders) ratcheted RSS to 82 GB — builders
  free everything, but glibc strands freed small allocations in per-thread
  arenas (430 observed) *and* in the unreturnable interior of the main heap
  (~2.6 GB per million rows even with `MALLOC_ARENA_MAX=2`). Two-part fix:
  the Python extension now calls `mallopt(M_ARENA_MAX, 2)` at import (respects
  an explicit user value; no-op on musl/macOS), and the demo load writes the
  nodes table in **fresh-process 10M-row chunks** so the allocator high-water
  mark resets per chunk — peak RSS ~25 GB regardless of dataset size.
- **Embedding column type**: the demo pipeline wrote `LargeList` embeddings, which
  the vector index silently ignores (it matches `List`/`FixedSizeList` only) —
  switched to `FixedSizeList`, so the HNSW index actually builds.
- **Demo load OOM + data-loss hazards** (`prepare_demo.py`): stage `load` read
  each embedding shard with a whole-shard `pq.read_table` — a 37 GB single-shard
  run was OOM-killed by the kernel; it now streams shards in 250k-row batches
  (verified position-aligned across multi-shard/coprime-batch boundaries).
  Shard deletion moved to strictly AFTER table commit (an earlier delete-on-
  advance variant destroyed the only embedding copy). Stage `embed` now rotates
  5M-row shards (`BSDB_EMBED_SHARD_ROWS` override) so load can reclaim disk
  incrementally. Covered by `tests/python/test_prepare_demo.py`.

### Added
- **Two-level Graph RAG in the Wikipedia demo** (`examples/web_ui/app.py`): the
  local-search tab now reranks with a bitmap-filtered vector search — the PPR
  neighborhood becomes an `id IN (...)` RoaringBitmap predicate on the HNSW
  index (topology prunes, semantics orders), mirroring the seed-index → CSR
  expansion → filtered-rerank architecture validated at whole-site scale.
- **Roadmap**: LangChain + LlamaIndex connector detail (Active Roadmap §9,
  Phase 12): vector stores, Graph-RAG retrievers wrapping
  `graph_rag_search`/`drift_search`, and an edge-table property-graph store.
- **Wikipedia demo dataset pipeline** (`scripts/build_demo_dataset.py`): consumes
  the full dump parquets, resolves mixed curid/title edge endpoints to int64
  curids (parallel, memory-bounded), prunes to the largest connected component and
  a dense hub-centered subgraph **inside BenoStreamDB** (`connected_components`,
  `degree_centrality`, `subgraph`), and emits `data/demo_nodes.parquet` /
  `data/demo_edges.parquet` with integer node ids the CSR index requires.
  Optional `sentence-transformers` embeddings (default `BAAI/bge-large-en-v1.5`,
  1024-d) for the UI's semantic entity resolution.

### Fixed
- **Merge/upsert nested-runtime panic**: `Table::merge` no longer drives the synchronous
  `MergePlanner` entry points inside a running tokio runtime (fixes
  `Cannot start a runtime from within a runtime` in `test_compound_pk.py`);
  `runtime_block_on` additionally offloads to a dedicated thread when invoked from
  within a runtime context.
- **Merge-path performance**: one reused current-thread runtime per `MergePlanner`
  (was: a new runtime per async call), object-store client hoisted out of per-segment
  loops, streams/bloom checks drained in a single runtime turn, and key matching
  changed from O(source × segment) linear scan to a hash index.
- **Graph UDAFs were stubs**: implemented real BFS in `shortest_path` and
  `graph_neighbors`, union-find in the `connected_components` UDAF, and neighborhood
  Jaccard similarity in `jaccard_coefficient` (previously returned hardcoded values).
- **Missing `connecting_paths` UDF**: implemented and registered
  `ConnectingPathsUDF` (pairwise BFS paths between seed nodes).
- **Scalar-arg overwrite in multi-partition merges**: `hops`/`directed` in the
  `subgraph`/`graph_neighbors`/`connecting_paths` accumulators are now optional so
  empty partitions no longer clobber captured values during `merge_batch`.
- **Python graph API drift**: `subgraph`, `connecting_paths`, `shortest_path`,
  `graph_neighbors` wrappers now match the Rust bindings (CSR-index route when
  `graph_column` is given, SQL route otherwise; `hops`/`directed`/`max_depth`
  accepted); `drift_search` now passes keyword arguments to the binding.
- `subgraph()` now returns full edge rows (payload columns such as `weight`
  preserved) by joining the extracted edge set back against the table.
- Added `jinja2` to the `dev` extra so `test_dbt_macro_file_syntax` runs in CI.

### Performance
- **Frontier-based out-of-core graph traversal**: new `core::algorithms::frontier`
  executes BFS (level-synchronous, one hash join per hop over a deduplicated
  symmetric adjacency, parquet-materialized visited/frontier state with early
  exit) inside DataFusion. The `subgraph()` and `graph_neighbors()` bindings now
  route through it instead of the UDAF accumulator that buffered the entire edge
  set plus a `HashMap` adjacency in RAM (tens of GB at 100M+ edge scale). The
  UDAFs remain available for raw SQL use.
- **Connected components at scale**: rewrote `compute_connected_components` from
  textbook per-hop label propagation (O(diameter) rounds, two full edge joins per
  round) to **pointer jumping + edge contraction** with a single symmetric edge
  join per round and a cheap checksum convergence probe. Validated on the
  English-Wikipedia link graph (160.5M raw / 91.7M clean edges):
  `connected_components()` completes in **~137s** (was effectively intractable
  at this scale).
- **Label propagation**: pre-materializes a symmetric edge set so each round does
  one join instead of two; per-run unique temp directory; state-sized convergence.

### Fixed
- **Intermittent vector-index panic**: `ArrowHnsw::get_vector` casts an Arrow
  binary value slice to `&[f32]`; when the IPC buffer landed on a misaligned byte
  offset, `bytemuck::cast_slice` panicked with
  `TargetAlignmentGreaterAndInputNotAligned` (debug-only, order-dependent). The
  binary buffer is now repacked into an aligned buffer once at load when needed,
  keeping the search hot path zero-copy.

### Notes
- Evaluated `mimalloc` as a `#[global_allocator]` for the Python extension; it
  fails to import under glibc (`cannot allocate memory in static TLS block`) and
  was reverted. A future option is `ld_preload`/`GLIBC_TUNABLES` or a non-TLS
  allocator.

---

## [0.8.1] - 2026-09-19

### Added
- **Graph RAG (CSR Graph + Shared Algorithms)**:
  - Extracted shared graph algorithms into `src/core/algorithms/` (connected components, label propagation, topological sort).
  - Added CSR graph index (`src/core/index/csr_graph.rs`) and builder (`build_graph.rs`); registered `IndexAlgorithm::CsrGraph` across manifest, index_config, segment, and Python helpers.
  - Refactored graph UDFs (pagerank, personalized_pagerank, shortest_path, strongly_connected_components, connected_components, label_propagation, jaccard_coefficient) to use shared algorithms + CSR graph.
  - Added `drift_search` UDF and Python GraphAPI + drift_search bindings.
- **Vector Index Joins**:
  - Reworked the index-join optimizer and physical plan; added a vector search sort-expression parser; updated HNSW-IVF, GPU, and HNSW paths.
- **Raw Vector Search API**:
  - Added `execute_vector_search_raw_with_config` for raw vector search returning `ScoredResult` with segment and row IDs.
- **PrimaryKeyFilter Row-Value IN-List Pushdown**:
  - Added `PrimaryKeyFilter` type representing a set of candidate PK rows (row-value IN list).
  - `from_expr` supports `Expr::InList`, equality, AND of disjoint columns, and OR (row-value IN lists).
  - `from_batch` builds from a RecordBatch; `to_expr` converts back to a DataFusion Expr; `to_query_filter` yields a single-column IN-list QueryFilter for inverted-index pushdown.
  - Reworked `check_primary_key_uniqueness_async` to push the whole batch as a single expression.
- **Time32/Time64 Datatype Support**:
  - Safely ignore Time datatypes for in-memory vector indexing.

### Changed
- Reworked compaction, reader (filter/scan), segment, manifest, query, and cache; renamed `MergeMode::MergeOnWrite` to `CopyOnWrite`.
- Extended PyTable bindings and the Python package with graph and drift_search APIs.
- DRY up release workflow, add deps, and document tech debt.

### Fixed
- Fixed FFI `load_async_with_cache_key` call missing `use_mmap` argument (compile error with `java` feature).
- Fixed clippy lints: `manual_map`, `needless_borrow`, `new_without_default`, `len_zero`, `needless_borrows_for_generic_args`.

---

## [0.8.0] - 2026-09-16

### Added
- **Graph RAG & Graph Analytics**:
  - Added a comprehensive suite of graph UDFs: pagerank, personalized_pagerank, shortest_path, strongly_connected_components, connected_components, label_propagation, jaccard_coefficient, degree_centrality, preferential_attachment, subgraph, louvain_communities, modularity, clustering_coefficient, adamic_adar, connecting_paths, neighbors, resource_allocation, to_graphviz, and topological_sort.
  - Added a graph search handler to the `benostreamdb-search` gateway.
  - Added the Python Graph API and graph RAG pipeline bindings.
  - Added dbt graph macros and graph RAG edge-table documentation.
- **Arrow IPC Vector Index & Micro-Batch Streaming Ingest Buffer**:
  - Implemented the Arrow IPC vector index and a micro-batch streaming ingest buffer.
  - Reworked HNSW-IVF index construction and search.
- **Zero-Copy Arrow IPC vectorSearch FFI Bridge**:
  - Added a zero-copy Arrow IPC `vectorSearch` FFI bridge for Spark and Trino.
  - Added `vectorSearch` JNI bindings to the Java connectors.
- **Python CLI for Background Services**:
  - Added Python CLI commands for installing and uninstalling background services.
  - Added a universal installer/uninstaller and a centralized configuration file for background services.
  - Added native background service configurations for `benostream-search`.

### Changed
- Reworked vector search for correctness and SQL aggregate consistency.
- Updated the Trino and Spark connectors.

### Fixed
- Updated rustls to 0.23.45 to fix RUSTSEC-2026-0285.
- Various connector and CI/CD hotfixes.

### Build / CI
- DRY up the CI pipeline to dynamically install `[dev]` extras directly from the built wheel.
- Added networkx, cffi, and scikit-learn to CI test environments.
- Upgraded `actions/checkout` from v4 to v5 for Node.js 24 support.
- Added the Renovate workflow; removed Dependabot.

---

## [0.7.0] - 2026-09-10

### Added
- **Multi-Vector Search & Reciprocal Rank Fusion (RRF)**:
  - Coordinate multi-vector search queries across multiple vector columns (e.g., `title_vec` and `body_vec`) using reciprocal rank fusion ($1 / (k + \text{rank} + 1)$).
  - DataFusion SQL physical plan optimization and optimizer pushdown for multi-vector expressions (`dist_l2(col1, ...) as dist1, dist_l2(col2, ...) as dist2`).
  - Added programmatic and SQL end-to-end integration tests in `tests/test_multi_vector_search.rs`.
- **Composite Scalar Roaring Bitmap Indexes (`IndexAlgorithm::CompositeBitmap`)**:
  - Multi-column point and range query acceleration using combined composite inverted bitmap indexes.
  - Virtual composite column naming and tokenization using exact `"identity"` tokenization to preserve multi-column key terms (`val1\0val2`).
  - Fluent table API `table.add_composite_index(name, columns)` and filter rewriting.
  - Added end-to-end integration tests in `tests/test_composite_index.rs`.
- **Apache Polaris & Lakekeeper Iceberg REST Catalog OAuth2 Client Credentials**:
  - Implemented standard `/v1/oauth/tokens` client credentials grant flow for Iceberg REST catalogs.
  - Added token caching with automatic expiry tracking and refresh within 60-second window.
  - Injected `Authorization: Bearer <token>` across all REST catalog requests (`load_table`, `create_table`, `commit_table`).
- **Core Community TurboQuant™ (TQ4 / TQ8) Scalar Quantization**:
  - Polar Quantization (PQ) and TurboQuant 4-bit / 8-bit quantized HNSW vector index algorithms integrated into open-source core engine.
- **Production Hardening & Correctness Enhancements**:
  - **Vector Metric Propagation**: Dynamic metric parsing (`VectorMetric::from_str`) and propagation from `IndexAlgorithm` (`L2`, `Cosine`, `InnerProduct`, `L1`, `Hamming`, `Jaccard`) through index construction and Puffin/Parquet serialization.
  - **Global KNN Ordering**: Refactored `merge_and_rerank_vector_results` to guarantee monotonic ascending distance order across all returned batches without unordered `HashMap` bucketing.
  - **Metric Parity Test Suite**: Added `tests/test_vector_metrics_parity.rs` establishing 100% nearest-neighbor accuracy against exact brute force ground truth across all 6 metrics.
  - **Strict Query Execution Semantics**: Segment vector search failures now fail the query immediately with actionable diagnostics rather than silently omitting rows.
  - **Zero-Warning Standard**: Eliminated diagnostic `println!` statements in favor of structured `tracing::debug!`, resolved all clippy compiler warnings with `#![deny(warnings)]`.

---

## [0.6.0] - 2026-09-09

### Added
- **Multi-Protocol Gateway Ecosystem**:
  - **`benostreamdb-search` Service (`bsdb-search` binary)**: Dual Elasticsearch 7.10 (Port 9200) and Qdrant (Port 6333) compatible REST APIs over BenoStreamDB tables.
    - Elasticsearch 7.10 API: Full document CRUD (`POST /{index}/_doc`), hybrid search (`POST /{index}/_search` with BM25 + HNSW kNN + Reciprocal Rank Fusion), index management (`PUT /{index}`, `POST /{index}/_refresh`), and cluster health (`GET /_cluster/health`).
    - Qdrant REST API: Collection management (`/collections/{name}`), point upsert/retrieval (`/collections/{name}/points`), and vector search (`/collections/{name}/points/search`).
    - Prometheus metrics exporter on `/metrics` (Port 9090).
  - **`benostreamdb-flight` Gateway Service**: Native Arrow Flight SQL gRPC gateway (Port 50051) enabling zero-copy analytics for DuckDB, Polars, Apache Spark, and JDBC/ODBC BI tools via standard Flight SQL/ADBC.
- **Multi-Flavor GPU Acceleration & Hardware Auto-Detection**:
  - Optional GPU acceleration exposed across `benostreamdb-search` and `benostreamdb-flight` via `cuda`, `wgpu`, `rocm`, `intel`, and `all-gpu` feature flags.
  - Runtime device selection via `BENOSEARCH_DEVICE=auto|cuda[:N]|rocm[:N]|intel[:N]|mps|cpu`.
  - Active compute backend (`compute` block) exposed in `GET /` cluster info and `GET /_cluster/stats`.
  - `docker/Dockerfile.gpu`: Multi-flavor GPU container image with CUDA 12 runtime, NVRTC JIT compilation, and Vulkan/Mesa drivers for AMD Radeon and Intel Arc.
  - `docker/docker-compose.gpu.yml`: Compose GPU override with hardware reservations and device pass-through.
- **Iceberg Compaction Resilience & Index Recovery**:
  - `Table::recover_indexes_async(&self)` / `recover_indexes(&self)`: Re-indexes data files that are missing overlay index sidecars, recovering fast vector (HNSW) and keyword (BM25) search after external Iceberg tools (Spark `rewriteDataFiles`, Trino `OPTIMIZE`, PyIceberg) compact table data files.
- **Docker Container Infrastructure**:
  - `benostreamdb/quickstart:latest`: Single all-in-one developer container running ES 7.10 (9200), Qdrant (6333), and Flight SQL (50051).
  - `benostreamdb/search:latest`: Standalone production search microservice.
  - `benostreamdb/flight:latest`: Standalone production Arrow Flight SQL microservice.
  - `docker/docker-compose.quickstart.yml`: Single-command full stack with MinIO (S3), Project Nessie catalog, and BenoStreamDB.
  - `docker-compose.production.yml`: Production multi-container configuration with health checks and resource limits.
- Okapi BM25 keyword scoring (tunable `k1`/`b`) with an English analyzer in the core engine; public `keyword_search_index` API.
- Smart hybrid trigger fusing keyword (BM25) and vector (HNSW) results via reciprocal rank fusion (RRF, k=60).
- Background segment index builds with `wait_for_background_tasks_async` for deterministic refresh semantics.

## [0.5.3] - 2026-06-21

### Added
- Unified catalog table loading logic (`load_from_catalog`) across Glue, Hive, Nessie, and REST catalog wrappers in python bindings.
- Explicit commit-time flushing and background synchronization in integration tests (`test_turboquant_integration.py` and `test_wikipedia.py`).

### Changed
- Fixed remote path routing in `flush_async` to correctly construct relative remote paths for staged files instead of using local staging paths.
- Upload data files synchronously relative to commit operation to avoid read-after-write race conditions in remote catalogs.
- Fixed a missing file upload call in the background indexing task for vector indexes.

### Fixed
- Clippy compiler warnings and errors in HNSW/PQ modules.

---

## [0.5.2] - 2026-06-15

### Changed
- Bumped version to 0.5.2.
- Resolved write buffer schema alignment & list array indexing bugs.

---

## [0.5.1] - 2026-06-07

### Changed
- Optimized vector search.
- Fixed WAL tests.

---

## [0.5.0] - 2026-06-03

### Added
- Comprehensive unit test suites for Spark and Trino connectors
- Thread-local GPU context for concurrent query safety
- Structured error types and fallback chain for production readiness
- Query configuration API
- Metrics instrumentation for observability
- Release profiles (`release` and `release-lto`) in Cargo.toml
- `WriteAheadLog::append_fire_and_forget` — non-blocking WAL append that hands off batches to the WAL worker without blocking on `fdatasync`, eliminating ~800 ms write latency per call

### Changed
- CUDA is now an optional feature (no longer required for CI builds)
- FFI bounds hardened with panic safety guarantees
- Public APIs documented with rustdoc
- **`Table(index_all=False)` is now the default** (previously `True`). Automatic HNSW/BM25 index building on commit is now opt-in. This eliminates a silent 15–18 s background build that previously fired on every `commit()` for any table containing a vector column. To restore the old behaviour: `Table(uri, index_all=True)` or `table.index_all = True`.
- **`Table(autocommit=False)` is now the default** (previously `True`). Writes accumulate in an in-memory buffer and must be explicitly committed with `table.commit()`. This eliminates unexpected auto-flush overhead during ingestion loops.
- `write_async` no longer auto-detects columns named `"embedding"` as implicit HNSW index targets. Only columns explicitly registered via `add_index()` or `set_index_columns()` are indexed.
- `manifest_manager.load_latest_full` replaced with `load_latest` in the `flush_async` hot path, eliminating two unnecessary full manifest scans per commit.
- Primary key uniqueness check upgraded from O(N²) to O(N) using `HashSet`.

### Fixed
- Rustdoc warnings across the codebase
- PyO3 API compatibility issues
- Compilation errors in core modules
- Critical broken functionality items

---

## [0.4.0] - 2026-05-10

### Added
- Tiered manifests with 8MB dynamic chunking for large-scale deployments
- File-based distributed locking for concurrent write safety
- OTLP (OpenTelemetry) tracing integration
- FFI panic safety boundaries
- Chaos and concurrent writers test suites
- HNSW SIMD indexing (stabilized, replaced legacy simdeez dependency)

### Changed
- Modularized query execution engine
- Refactored monolithic `reader.rs` into modular files
- Refactored monolithic `manifest.rs` into `manager.rs` and `types.rs`
- Extracted segment indexing logic into separate builder modules
- Replaced `std::sync` primitives with `parking_lot` for better concurrency
- Feature-gated `opencl3` and `wgpu` for binary size reduction
- Upgraded `hashbrown` dependency and trimmed `sqlx` features
- Eliminated panicking `unwrap()` calls in core library code

### Fixed
- Filter fallback to Iceberg manifests when no index exists on the column
- Hybrid search stability and correctness
- GPU test fallback when CPU-only environment detected
- Vector search schema compatibility

---

## [0.3.3] - 2026-04-25

### Fixed
- Vector search schema alignment

---

## [0.3.2] - 2026-04-23

### Changed
- Stabilized streaming architecture
- Switched documentation deployment to GitHub Actions

### Fixed
- Streaming read and Python bindings

---

## [0.3.1] - 2026-04-15

### Fixed
- TQ8/blob_type index load dispatch

---

## [0.3.0] - 2026-04-14

### Added
- **High-Density Storage Milestone**: TQ4 (4-bit) and TQ8 (8-bit) TurboQuant quantization
- Global runtime support
- Schema evolution infrastructure
- Finalized indexing infrastructure

### Fixed
- Multiple stability fixes across storage and query layers

---

## [0.2.6] - 2026-04-12

### Changed
- Stabilized parallel execution
- Modernized logging infrastructure
- Updated GPU installation guidance

### Fixed
- Categorical column handling
- Metal backend type mismatch on macOS

---

## [0.2.3] - 2026-04-11

### Changed
- Modernized GPU acceleration
- Aligned with PyTorch standards for device detection

---

## [0.2.1] - 2026-04-10

### Changed
- Auto-sync pyproject version from Cargo.toml
- Made cudarc optional for non-CUDA CI builds

### Fixed
- CUDA and MPS device recognition
- PK validation memory and schema merge logic

---

## [0.2.0] - 2026-04-09

### Added
- Core engine refactor
- WAL deduplication
- Thread-safe GPU context
- Hardware-agnostic compute dispatch with thread-local context management

### Changed
- Extracted read, write, builder, schema, and fluent APIs from monolithic table module
- Introduced `TableBuilder` to streamline initialization
- Eliminated implicit tokio runtime creation

---

## [0.1.12] - 2026-04-05

### Added
- Dynamic GPU backend detection (PyTorch-style `build.rs`)
- Single-source version management

### Changed
- Unified version across all metadata files
- Removed non-standard extra-features mapping
- Removed native Windows target (WSL2 recommended)
- Updated PyPI classifiers for Python 3.13 and 3.14 support
- Optimized CI matrix for universal Python 3.10 abi3 wheels

### Fixed
- Missing library `ImportError` on Linux wheels
- PyPI trailing data wheel rejection on macOS
- Manifest manager Rust compiler error

---

## [0.1.9] - 2026-04-04

### Added
- Partitioned tables support
- SQL aggregation fixes

### Changed
- Standardized Table API across Python integration tests
- Increased floating-point tolerance for GPU distance kernels

### Fixed
- Inverted index row ID encoding
- Integration test stability

---

## [0.1.8] - 2026-04-04

### Changed
- Transitioned to explicit Device API
- Broadened Intel GPU detection
- Standardized Intel GPU backend naming

### Fixed
- PyArrow schema interoperability
- Mutability errors in `python_gpu_context.rs`

---

## [0.1.7] - 2026-04-03

### Added
- **10x ingestion speedup** via Async HNSW and Iceberg v3 ZSTD optimizations

### Changed
- Release stabilization and documentation improvements

---

## [0.1.6] - 2026-04-02

### Added
- Strict primary key uniqueness enforcement with inverted index lookups

### Fixed
- PK uniqueness for upsert operations
- macOS build compatibility

---

## [0.1.5] - 2026-04-02

### Added
- HNSW-IVF stabilization
- Python extras hardware mapping (`all_gpu`, `intel_cpu`)

### Changed
- Fully internalized `hnsw_rs` to `src/core/index/hnsw_rs` for crates.io compatibility
- Resolved `unexpected_cfgs` warnings

---

## [0.1.3] - 2026-04-01

### Fixed
- HNSW crate patch compatibility

---

## [0.1.2] - 2026-03-31

### Added
- Vector search parameter tuning
- Enhanced diagnostics and query optimization
- Python API improvements
- Explain plan enhancements

### Changed
- Improved API design and explain output
- Stabilized hybrid search pipeline and RAG demo
- CI: gated FFI and connector tests behind `java` feature
- CI: removed Linux aarch64 from release matrix
- CI: switched to Zig-based cross-compilation
- CI: added Rust caching for faster builds

### Fixed
- Restored `add_index_columns` method for Python/Rust bindings
- OpenCL linker errors and cross-compilation issues
- Manylinux compatibility for both architectures

---

## [0.1.1] - 2026-03-30

### Added
- Vector UDFs and aggregates registered in SQL context
- Hybrid search stabilization
- Initial RAG demo support

### Changed
- Aligned development guidelines with pragmatic programmer principles
- Registered vector operators with DataFusion for hybrid queries

---

## [0.1.0] - 2026-03-30

### Added
- Initial release of BenoStreamDB
- Serverless index-streaming database with overlay indexing
- Apache Iceberg V2/V3 compliance
- Persistent scalar (RoaringBitmap) and vector (HNSW) indexes
- Native SQL support via DataFusion
- Python bindings via PyO3
- Multi-backend GPU acceleration (CUDA, ROCm, Metal, Intel XPU)
- Fluent query API with method chaining
- pgvector-compatible SQL operators

---

[Unreleased]: https://github.com/benolabsai/benostreamdb/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/benolabsai/benostreamdb/compare/v0.8.1...v0.9.0
[0.5.3]: https://github.com/benolabsai/benostreamdb/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/benolabsai/benostreamdb/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/benolabsai/benostreamdb/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/benolabsai/benostreamdb/compare/v0.4.1...v0.5.0
[0.4.0]: https://github.com/benolabsai/benostreamdb/compare/v0.3.3...v0.4.0
[0.3.3]: https://github.com/benolabsai/benostreamdb/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/benolabsai/benostreamdb/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/benolabsai/benostreamdb/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/benolabsai/benostreamdb/compare/v0.2.6...v0.3.0
[0.2.6]: https://github.com/benolabsai/benostreamdb/compare/v0.2.3...v0.2.6
[0.2.3]: https://github.com/benolabsai/benostreamdb/compare/v0.2.1...v0.2.3
[0.2.1]: https://github.com/benolabsai/benostreamdb/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/benolabsai/benostreamdb/compare/v0.1.12...v0.2.0
[0.1.12]: https://github.com/benolabsai/benostreamdb/compare/v0.1.9...v0.1.12
[0.1.9]: https://github.com/benolabsai/benostreamdb/compare/v0.1.8...v0.1.9
[0.1.8]: https://github.com/benolabsai/benostreamdb/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/benolabsai/benostreamdb/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/benolabsai/benostreamdb/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/benolabsai/benostreamdb/compare/v0.1.3...v0.1.5
[0.1.3]: https://github.com/benolabsai/benostreamdb/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/benolabsai/benostreamdb/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/benolabsai/benostreamdb/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/benolabsai/benostreamdb/releases/tag/v0.1.0
