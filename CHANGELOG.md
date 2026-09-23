# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **CUDA JIT now works with pip's CUDA 13 wheels.** cudarc 0.13's nvrtc loader
  probes a fixed candidate list (`libnvrtc.so`, `libnvrtc64*.so`,
  `libnvrtc.so.{12,11,10,1}`) that predates CUDA 13, so `nvidia-*-cu13` wheels
  (`libnvrtc.so.13`) were never found: the probe panicked and the GPU silently
  fell back to CPU. `core::index::nvrtc` now resolves *whatever* `libnvrtc.so*`
  is installed — any version, any layout (`HDB_NVRTC_PATH`, the interpreter's
  `site-packages`, `CUDA_HOME`/`CUDA_PATH`, `PYTHONPATH`, `LD_LIBRARY_PATH`,
  system paths) — preloads the `libnvrtc-builtins` companion with `RTLD_GLOBAL`,
  and compiles the embedded `.cu` sources itself, handing the PTX to cudarc. No
  re-exec, no env-var prefix, no shim script (`scripts/create_cuda_shims.sh`
  removed).
- **Demo load resume no longer duplicates rows.** `scripts/prepare_demo.py`
  resumed from the rounded-down chunk boundary on the assumption that chunks
  commit atomically. They don't: the write path spills to a real commit whenever
  the buffer exceeds `HYPERSTREAM_CACHE_GB` (default 1 GB), so a killed chunk
  leaves partial rows committed and the round-down re-wrote them. Resume now
  continues from the exact committed row count (`_resume_offset`), guarded by a
  regression test.

### Added
- **Native ingest orchestrator (A4)**: `Table::ingest_async` /
  `table.ingest(paths, chunk_rows, parallelism, index_all, resume)` plans parquet
  inputs into row-range work units, runs a bounded `buffer_unordered(parallelism)`
  pool where each worker builds a *private* segment (data + indexes) concurrently,
  and commits completed segments through the OCC manifest CAS. Completed units are
  recorded in a `_ingest_state.json` sidecar so an interrupted load resumes at the
  unit boundary. Returns a report (`units_total/skipped/committed`,
  `rows_ingested`, `segments`). `IngestOptions::compact_after` runs
  `rewrite_data_files` at the end so segment/manifest counts stay bounded.
  CLI: `hdb table ingest --uri … --input … [--plan] [--chunk-rows N]
  [--parallelism N] [--index-all] [--compact]`, with `--row-start/--row-end` as
  the serverless thin-runner mode (`Table::ingest_range_async`).
  **Multi-format inputs**: `.parquet` (row-range units) plus `.csv`, `.json` /
  `.ndjson`, and `.arrow` / `.ipc` / `.feather` (one unit per file, streamed
  whole-file with schema inference) — all via Arrow readers already in the tree.
- **Ingest memory discipline**: `core::memory::HeapTrimPolicy` returns freed
  heap pages to the OS (`malloc_trim`) at work-unit boundaries once RSS exceeds
  a budget, so long-lived in-process loads no longer ratchet toward the sum of
  every glibc arena's high-water mark. Wired through
  `IngestOptions::memory_budget_bytes`, the `HDB_INGEST_MEMORY_BUDGET_GB` env
  var, `hdb table ingest --memory-budget-gb`, and
  `table.ingest(..., memory_budget_gb=…)`. Allocator evaluation recorded in
  `core::memory`: glibc + `malloc_trim` chosen; jemalloc deferred; mimalloc
  rejected (static-TLS under pyo3).
- **Multi-machine ingest (A4)**: `WorkCoordinator` trait + `ObjectStoreCoordinator`
  backend (`core::table::coordinator`) — lease-based work stealing over the
  object store, reusing `FileBasedLock` (CAS claim + heartbeat + expiry-steal).
  `Table::ingest_coordinated_async` claims units dynamically (build in parallel,
  commit serially via the OCC CAS, release-on-failure so another node retries);
  a dead node's lease expires and its unit is stolen. Surfaces:
  `hdb table ingest --coordinate [--lease-ttl-secs N]` and
  `table.ingest(..., coordinate=True, lease_ttl_secs=300)`. No broker, no etcd,
  no Raft cluster.

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
  Tuning knobs (no rebuild): `HDB_HNSW_N_LISTS`, `HDB_HNSW_M`,
  `HDB_HNSW_EF_CONSTRUCTION`.

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
  (also on the Python `Table`). Metrics: `hyperstreamdb_manifest_commit_rebases_total`,
  `hyperstreamdb_manifest_commit_skipped_removals_total`.
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
  5M-row shards (`HDB_EMBED_SHARD_ROWS` override) so load can reclaim disk
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
  a dense hub-centered subgraph **inside HyperStreamDB** (`connected_components`,
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
  - Added a graph search handler to the `hyperstreamdb-search` gateway.
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
  - Added native background service configurations for `hyperstream-search`.

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
  - **`hyperstreamdb-search` Service (`hypersearch` binary)**: Dual Elasticsearch 7.10 (Port 9200) and Qdrant (Port 6333) compatible REST APIs over HyperStreamDB tables.
    - Elasticsearch 7.10 API: Full document CRUD (`POST /{index}/_doc`), hybrid search (`POST /{index}/_search` with BM25 + HNSW kNN + Reciprocal Rank Fusion), index management (`PUT /{index}`, `POST /{index}/_refresh`), and cluster health (`GET /_cluster/health`).
    - Qdrant REST API: Collection management (`/collections/{name}`), point upsert/retrieval (`/collections/{name}/points`), and vector search (`/collections/{name}/points/search`).
    - Prometheus metrics exporter on `/metrics` (Port 9090).
  - **`hyperstreamdb-flight` Gateway Service**: Native Arrow Flight SQL gRPC gateway (Port 50051) enabling zero-copy analytics for DuckDB, Polars, Apache Spark, and JDBC/ODBC BI tools via standard Flight SQL/ADBC.
- **Multi-Flavor GPU Acceleration & Hardware Auto-Detection**:
  - Optional GPU acceleration exposed across `hyperstreamdb-search` and `hyperstreamdb-flight` via `cuda`, `wgpu`, `rocm`, `intel`, and `all-gpu` feature flags.
  - Runtime device selection via `HYPERSEARCH_DEVICE=auto|cuda[:N]|rocm[:N]|intel[:N]|mps|cpu`.
  - Active compute backend (`compute` block) exposed in `GET /` cluster info and `GET /_cluster/stats`.
  - `docker/Dockerfile.gpu`: Multi-flavor GPU container image with CUDA 12 runtime, NVRTC JIT compilation, and Vulkan/Mesa drivers for AMD Radeon and Intel Arc.
  - `docker/docker-compose.gpu.yml`: Compose GPU override with hardware reservations and device pass-through.
- **Iceberg Compaction Resilience & Index Recovery**:
  - `Table::recover_indexes_async(&self)` / `recover_indexes(&self)`: Re-indexes data files that are missing overlay index sidecars, recovering fast vector (HNSW) and keyword (BM25) search after external Iceberg tools (Spark `rewriteDataFiles`, Trino `OPTIMIZE`, PyIceberg) compact table data files.
- **Docker Container Infrastructure**:
  - `hyperstreamdb/quickstart:latest`: Single all-in-one developer container running ES 7.10 (9200), Qdrant (6333), and Flight SQL (50051).
  - `hyperstreamdb/search:latest`: Standalone production search microservice.
  - `hyperstreamdb/flight:latest`: Standalone production Arrow Flight SQL microservice.
  - `docker/docker-compose.quickstart.yml`: Single-command full stack with MinIO (S3), Project Nessie catalog, and HyperStreamDB.
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
- Initial release of HyperStreamDB
- Serverless index-streaming database with overlay indexing
- Apache Iceberg V2/V3 compliance
- Persistent scalar (RoaringBitmap) and vector (HNSW) indexes
- Native SQL support via DataFusion
- Python bindings via PyO3
- Multi-backend GPU acceleration (CUDA, ROCm, Metal, Intel XPU)
- Fluent query API with method chaining
- pgvector-compatible SQL operators

---

[Unreleased]: https://github.com/rla3rd/hyperstreamdb/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/rla3rd/hyperstreamdb/compare/v0.8.1...v0.9.0
[0.5.3]: https://github.com/rla3rd/hyperstreamdb/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/rla3rd/hyperstreamdb/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/rla3rd/hyperstreamdb/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/rla3rd/hyperstreamdb/compare/v0.4.1...v0.5.0
[0.4.0]: https://github.com/rla3rd/hyperstreamdb/compare/v0.3.3...v0.4.0
[0.3.3]: https://github.com/rla3rd/hyperstreamdb/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/rla3rd/hyperstreamdb/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/rla3rd/hyperstreamdb/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/rla3rd/hyperstreamdb/compare/v0.2.6...v0.3.0
[0.2.6]: https://github.com/rla3rd/hyperstreamdb/compare/v0.2.3...v0.2.6
[0.2.3]: https://github.com/rla3rd/hyperstreamdb/compare/v0.2.1...v0.2.3
[0.2.1]: https://github.com/rla3rd/hyperstreamdb/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/rla3rd/hyperstreamdb/compare/v0.1.12...v0.2.0
[0.1.12]: https://github.com/rla3rd/hyperstreamdb/compare/v0.1.9...v0.1.12
[0.1.9]: https://github.com/rla3rd/hyperstreamdb/compare/v0.1.8...v0.1.9
[0.1.8]: https://github.com/rla3rd/hyperstreamdb/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/rla3rd/hyperstreamdb/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/rla3rd/hyperstreamdb/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/rla3rd/hyperstreamdb/compare/v0.1.3...v0.1.5
[0.1.3]: https://github.com/rla3rd/hyperstreamdb/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/rla3rd/hyperstreamdb/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/rla3rd/hyperstreamdb/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/rla3rd/hyperstreamdb/releases/tag/v0.1.0
