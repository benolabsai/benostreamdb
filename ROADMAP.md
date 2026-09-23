# Real-World Testing & Production Readiness Plan

## Overview

This document outlines the step-by-step plan to take HyperStreamDB from PoC to production-ready.

**Timeline:** ~8 weeks  
**Current Phase:** Phases 1–10 COMPLETE ✅ | Active: Core Product (Native Ingest Orchestrator) + Client Ecosystem

---

## Phase 1: Real-World Testing (Weeks 1-2) ✅ COMPLETE

### Objectives
- Validate performance with real datasets
- Identify bottlenecks
- Establish baseline metrics

### Test Datasets

#### 1. NYC Taxi Dataset ✅
- **Size:** 3M rows (January 2023 subset)
- **Purpose:** Test scalar filtering, compaction, manifest scaling
- **Download:** `./tests/data/download_nyc_taxi.sh`
- **Test:** `python tests/integration/test_nyc_taxi.py`

**Results (2026-01-18):**
| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Ingest throughput | >100K rows/sec | **753,782 rows/sec** | ✅ |
| Query latency (indexed, p99) | <100ms | **85ms** | ✅ |
| Compaction (3M rows) | <5min | **4.91s** | ✅ |

#### 2. Synthetic Vector Embeddings ✅
- **Size:** 100K vectors, 768-dim (BERT-like)
- **Purpose:** Test HNSW performance, vector search
- **Generate:** `python tests/data/generate_embeddings.py`
- **Test:** `python tests/integration/test_vector_search.py`

**Results (2026-01-18):**
| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Vector search (100K, parallel) | <10s | **5.0s** | ✅ |
| Vector search (10K segment) | <50ms | ~500ms* | ⚠️ |
| Recall@10 | >95% | **100%** | ✅ |
| Index build time (100K) | <10min | **62s** | ✅ |

*Note: <50ms target achievable with scalar filter pre-pruning to 1-2 segments. 
Parallel loading (16 workers auto-detected) achieves 5s for 100K vectors across 10 segments.

#### 3. Wikipedia + Embeddings ✅
- **Size:** 100K documents with 768-dim embeddings
- **Purpose:** Test hybrid queries (scalar + vector)
- **Generate:** `python tests/data/generate_wikipedia.py`
- **Test:** `python tests/integration/test_wikipedia.py`

**Results (2026-01-18):**
| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Ingest (with embeddings) | >50K rows/sec | **4,563 rows/sec*** | ⚠️ |
| Scalar filter (all columns) | <500ms | 1,553ms | ⚠️ |
| Scalar filter (w/projection) | <500ms | **112ms** | ✅ |
| Vector search (100K) | <10s | **3.9s** | ✅ |
| Hybrid query | <10s | **3.9s** | ✅ |

*Notes:
- Ingest I/O bound by 768D embedding writes (~315MB total)
- Scalar query uses STRING INVERTED INDEX + COLUMN PROJECTION
- With `columns=[]` parameter: skip embedding reads → **142x speedup**

### Tasks
- [x] Create data download scripts
- [x] Create benchmark framework (Criterion)
- [x] Create integration tests
- [x] Run NYC Taxi tests
- [x] Run vector search tests
- [x] Run Wikipedia hybrid query tests
- [x] Profile and optimize bottlenecks
- [x] Document performance results

### Key Optimizations Implemented
1. **Parallel HNSW Loading** - Auto-detects system RAM and loads segment indexes concurrently
2. **Roaring Bitmap Indexes** - Sub-100ms indexed queries on 3M+ rows
3. **String Inverted Indexes** - Fast equality filtering on string/category columns
4. **Date/Timestamp Indexes** - Inverted indexes with day-granularity for time filtering
5. **Query Planner Pruning** - Skips segments based on column statistics
6. **Configurable Parallelism** - `table.set_max_parallel_readers(n)` for memory-constrained environments
7. **Column Projection** - Skip reading unused columns (e.g., embeddings) → 142x faster scalar queries

### Performance Baseline (2026-01-18)

| Operation | Dataset | Performance | Notes |
|-----------|---------|-------------|-------|
| Query (selective) | NYC Taxi 3M | **85ms** | High-selectivity ID filter |
| Vector Search k=10 | 100K vectors | **4,598ms** | 10 segments, 16 parallel readers |
| Scalar (all cols) | Wikipedia 100K | 1,187ms | Full column scan |
| Scalar (projected) | Wikipedia 100K | **14ms** | 142x speedup via projection |

**Analysis:**
- **Vector search**: Native HNSW-IVF indexing avoids full scans.
- **Indexed queries**: Fast sub-100ms lookups for selective filters.
- **Column projection**: Significant performance gains by skipping large embedding columns.
- **Scale**: Designed to maintain O(1) lookup performance at petabyte scale.

Run benchmark: `python tests/benchmarks/benchmark_vs_iceberg.py`

---

## Phase 2: Nessie Integration (Week 3) ✅ COMPLETE

### Objectives
- Implement Iceberg REST Catalog v2 client
- Support table branching/merging
- Enable multi-table transactions

### Implementation

#### 1. Nessie REST Client
```rust
// src/catalog/nessie.rs
pub struct NessieClient {
    base_url: String,
    http_client: reqwest::Client,
}

impl NessieClient {
    pub async fn create_table(...) -> Result<()>;
    pub async fn load_table(...) -> Result<TableMetadata>;
    pub async fn commit(...) -> Result<()>;
    pub async fn create_branch(...) -> Result<()>;
    pub async fn merge_branch(...) -> Result<()>;
}
```

#### 2. Python API
```python
catalog = hdb.NessieCatalog("http://localhost:19120")
table = catalog.create_table("db.table1", schema=schema)
catalog.create_branch("dev", from_ref="main")
```

### Tasks
- [x] Implement Nessie REST client
- [x] Add catalog integration tests
- [x] Update Python bindings
- [x] Test with local Nessie instance
- [x] Document catalog usage

---

## Phase 3: Performance Optimization (Weeks 4-5) ✅ COMPLETE

### Objectives
- Implement Iceberg-Compatible API
    - **Goal:** Drop-in replacement via API compatibility (Client/Catalog)
- Optimize query planning
- Parallelize compaction
- Add caching layers
- Implement v2 Row-Level Mutation (Merge-on-Read & Copy-on-Write)
- Adopt Iceberg v3 Semantics (Views, Materialized Views)

### Optimizations

#### 1. Query Planner
- Partition pruning
- File pruning (manifest stats)
- Index selection

#### 2. Parallel Compaction
- Worker pool for concurrent bin processing
- Target: 4x speedup on multi-core

#### 3. Caching
- Manifest cache (avoid S3 reads)
- [x] Index cache (HNSW/bitmap)
- [x] LRU eviction policy (via `moka` crate)

#### 4. SIMD Acceleration
- AVX2 (x86_64) and NEON (ARM64) intrinsics for L2 distance calculations
- Significant speedup for vector comparisons

### Tasks
- [x] Implement Index-Accelerated Merge (Query Planner)
- [x] Add manifest/index caching
- [x] Implement Merge-on-Read (Deletion Vectors)
- [x] Add parallel compaction
- [x] Adopt Iceberg v3 Semantics (Views)
- [x] Ingest Performance Fix (Replaced JSON Index with Parquet)
- [x] Optimize Reader Performance (Metadata Caching)
- [x] Implement Native Hybrid Search (Scalar + Vector)
- [x] Implement Iceberg-Compatible API
    - [x] Define `Catalog` trait in `src/catalog/mod.rs`
    - [x] Refactor `NessieClient` to implement `Catalog`
    - [x] Ensure `TableMetadata` struct matches Iceberg spec
    - [x] Update Python bindings to use generic Catalog
- [x] Compare before/after metrics (benchmarked against baseline in Phase 1)

---

## Phase 3.5: Native SQL Support (DataFusion Integration) ✅ COMPLETE

### Objectives
- Enable full SQL queries (`SELECT`, `GROUP BY`, `ORDER BY`, `LIMIT`, `JOIN`)
- Leverage DataFusion's query optimizer
- Push down scalar filters to HyperStream indexes
- Optimize joins with Index Nested Loop Join

### Implementation
- **Dependency**: `datafusion`
- **Wrappers**:
    - `HyperStreamTableProvider` (implements `TableProvider`)
    - `HyperStreamExecutionPlan` (implements `ExecutionPlan`)
    - `IndexNestedLoopJoinExec` (custom physical plan for index-accelerated joins)
- **Python API**: `table.sql("SELECT ...")` and `session.sql("SELECT ...")`

### Tasks
- [x] Add `datafusion` dependency
- [x] Implement `HyperStreamTableProvider`
- [x] Implement `HyperStreamExecutionPlan` (Filter Pushdown)
- [x] Bind `SessionContext` to Python (`PySession`)
- [x] Verify SQL queries in integration tests (Select, Limits, Joins)
- [x] Implement **Index Nested Loop Join** (O(1) index lookups for join inner table)
- [x] Implement **Boolean Indexing** (native boolean support in inverted indexes)
- [x] Multi-table JOIN support with index optimization

---

## Phase 4.5: Multi-Catalog Support (Weeks 6-7) ✅ COMPLETE

### Objectives
- Support multiple catalog implementations beyond Nessie
- Enable enterprise adoption with Hive/Glue/Unity catalogs
- Maintain pluggable catalog abstraction

### Catalog Implementations

**Priority 1: REST Catalog** (1 week)
- Iceberg-standard REST API
- Vendor-neutral, multi-cloud
- Simplest implementation

**Priority 2: AWS Glue** (1 week)
- Native AWS integration
- Cloud-native catalog
- High demand from AWS users

**Priority 3: Hive Metastore** (2 weeks)
- Enterprise standard
- Thrift RPC integration
- Highest enterprise demand

**Priority 4: Unity Catalog** (2 weeks)
- Databricks ecosystem
- Growing adoption
- Modern catalog features

### Tasks
- [x] Implement REST Catalog (`src/catalog/rest.rs`)
- [x] Implement AWS Glue Catalog (`src/catalog/glue.rs`)
- [x] Implement Unity Catalog (`src/catalog/unity.rs`)
- [x] Implement Hive Metastore Catalog (`src/catalog/hive.rs`)
- [x] Add catalog selection API (`create_catalog()`)
  - Supported types: Hive, Nessie, REST, Glue, Unity
  - Added TOML configuration support via `create_catalog_from_config`
- [x] Update Python bindings for all catalogs
- [x] Integration tests for each catalog (Verified creation/config via factory tests)
- [x] Documentation for catalog configuration (Python docs updated)
- [x] Support OAuth2 client credentials flow in REST Catalog for Apache Polaris integration

---

## Phase 5: Spark/Trino Connector APIs (Weeks 8-9) ✅ COMPLETE

### Objectives
- Add file-level and split-level read APIs
- Enable Spark/Trino parallelism
- Support dbt integration via connectors

### API Additions

**1. File-Level APIs** (Week 8)
```rust
// Enable Spark task parallelism
pub fn list_data_files() -> Result<Vec<DataFileInfo>>;
pub fn read_file(file_path: &str, filter: Option<&str>) -> Result<Vec<RecordBatch>>;
pub fn get_table_statistics() -> Result<TableStatistics>;
```

**2. Split-Level APIs** (Week 9)
```rust
// Enable Trino fine-grained parallelism
pub fn get_splits(max_split_size: usize) -> Result<Vec<Split>>;
pub fn read_split(split: &Split, columns: Vec<String>) -> Result<Vec<RecordBatch>>;
```

**3. Statistics APIs**
```rust
// Query planner optimization
pub struct TableStatistics {
    row_count: u64,
    file_count: usize,
    total_size_bytes: u64,
    column_stats: HashMap<String, ColumnStatistics>,
}
```

### Tasks
- [x] Implement `list_data_files()` API
- [x] Implement `read_file()` with filter pushdown
- [x] Implement `get_table_statistics()` API
- [x] Implement `get_splits()` for byte-range parallelism
- [x] Implement `read_split()` with column projection
- [x] Add partition support (identity, bucket, truncate, temporal transforms, partition pruning)
- [x] Integration tests for file/split APIs
- [x] Benchmark parallelism improvements (parallel segment reads verified in Phase 1 & Criterion benchmarks)

### Connector Development (Post-API)
- [x] Spark DataSource V2 connector (Java/Scala - `spark-hyperstream`)
- [x] Trino Connector SPI implementation (Java - `trino-hyperstream`)
- [x] dbt adapter (`dbt-hyperstreamdb` - native Arrow Flight SQL adapter with vector search macros & partition-looping incremental materialization)

---

## Phase 6: Operational Tooling (Week 10) ✅ COMPLETE

### Objectives
- CLI for operations
- Metrics/monitoring
- Observability

### Tools

#### 1. CLI
```bash
hdb compact s3://bucket/table
hdb vacuum s3://bucket/table --older-than-days 7
hdb stats s3://bucket/table
hdb repair s3://bucket/table
```

#### 2. Metrics (Prometheus)
- Compaction duration
- Query latency
- Index hit/miss rate
- Storage usage

#### 3. Tracing (Jaeger)
- Distributed tracing
- Query execution breakdown

### Tasks
- [x] Implement CLI tool (hdb binary with REPL & SQL support)
- [x] Add Prometheus metrics (`/metrics` endpoint and Prometheus exporter)
- [x] Add tracing spans (`tracing-opentelemetry` & subscriber infrastructure)
- [x] Create Grafana dashboards & metrics documentation
- [x] Document monitoring setup

---

## Phase 6.5: Search & Query Gateways (Weeks 10-11) ✅ COMPLETE

### Objectives
- Expose engine over standard search and database protocols
- OpenSearch / Elasticsearch 7.10 REST compatibility for document search
- Qdrant REST compatibility for unstructured vector collections
- Arrow Flight SQL Gateway for zero-copy SQL analytics and dbt integration

### Implementations
1. **`hyperstreamdb-search` (Search REST Gateway)**:
   - Dual-protocol server: Port 9200 (OpenSearch/ES 7.10) & Port 6333 (Qdrant)
   - Okapi BM25 text search with doc-length sidecars
   - HNSW vector search with metadata filtering
   - Reciprocal Rank Fusion (RRF) hybrid search
   - Full Prometheus metrics (`/metrics`)
2. **`hyperstreamdb-flight` (Arrow Flight SQL Gateway)**:
   - Arrow Flight SQL gRPC service on Port 50051
   - Zero-copy Arrow record batch streaming with DataFusion execution engine
   - Supports ADBC, JDBC, and ODBC clients
3. **`dbt-hyperstreamdb` (Official dbt Adapter)**:
   - Vector search macros (`vector_distance`, `knn_search`, `vector_avg`, `type_vector`, `type_sparsevec`)
   - Custom materializations (`table`, `incremental` with partition-looping `insert_overwrite`)
   - DDL with Iceberg `PARTITIONED BY` syntax

---

## Phase 7: Production Hardening & Concurrency Control ✅ COMPLETE

### Objectives
- Vendor-neutral distributed locking & concurrency control
- Structured error handling & telemetry
- Data integrity & chaos resilience

### Implementations in Codebase
1. **Cloud-Agnostic Distributed Locking (`src/core/lock.rs`)**:
   - Implemented `FileBasedLock` over `object_store::ObjectStore` using atomic conditional creates / CAS (`PutMode::Create`).
   - Lease heartbeats with configurable TTL and clock skew drift protection.
   - Zero vendor lock-in (runs seamlessly over S3, GCS, Azure Blob, and local filesystems—no proprietary services like DynamoDB).
2. **Optimistic Concurrency Control (OCC) (`src/core/manifest/manager/commit.rs`)**:
   - Manifest commits use atomic snapshot swaps with exponential backoff retry loops.
   - Tested under massive multi-threaded contention (verified in `tests/test_concurrent_writers.rs` and `tests/test_concurrency_robust.rs`).
3. **Structured Observability (`src/telemetry/`)**:
   - Replaced ad-hoc logging with `tracing` and `tracing-opentelemetry` spans across query planning, index scanning, and compaction.
   - Integrated Prometheus metrics via `/metrics` endpoint.
4. **Resilience & Chaos Testing (`tests/test_chaos.rs`)**:
   - Verified graceful degradation: missing or corrupted index sidecars automatically fall back to full Parquet scans without panics or query failures.
   - ACID durability verified under abrupt termination (`tests/test_durability_robust.rs`).

---

## Phase 8: Documentation & Developer Guides ✅ COMPLETE

### Documentation Suite in `docs/`
- **Sphinx / ReadTheDocs Configuration**: Set up in `docs/source/conf.py` and `docs/requirements.txt`.
- **API & SQL Guides**:
  - `PGVECTOR_SQL_GUIDE.md` — Complete guide for pgvector operators (`<->`, `<=>`, `<#>`, `<+>`, `<~>`, `<%>`).
  - `PYTHON_VECTOR_API.md` — Fluent Python query API, index chaining, and hardware management.
  - `ICEBERG_V2_V3_API.md` — Iceberg V2/V3 metadata specifications, row lineage, and position deletes.
  - `GPU_SETUP_GUIDE.md` — Multi-backend GPU configuration (CUDA, ROCm/Vulkan, Apple Metal, Intel XPU).
  - `CONCURRENCY.md` & `COMPREHENSIVE_GUIDE.md` — Concurrency model and architecture breakdown.
- **Service Quickstarts**:
  - `GETTING_STARTED.md` — Search API quickstart for OpenSearch 7.10 and Qdrant endpoints.
  - `OPENSEARCH_COMPATIBILITY.md` — API support matrix and error envelope documentation.

---

## 🎯 Active Roadmap — Core Product First, Dependency-Ordered

> **Reorganized 2026-09-22.** Sections are ordered so that **core-product work comes first**, then by dependency (each section unblocks the next). Within each section, **completed items are listed first, followed by the next steps**. Completed phases (1–10) are retained below as history.

### Part A — Core Product

#### A1. Core Engine Correctness & Concurrency
*Foundational: unblocks the ingest orchestrator and TB-scale operation.*

**Completed**
- [x] Cloud-agnostic distributed locking (`FileBasedLock`, CAS `PutMode::Create`) — Phase 7.
- [x] OCC manifest commits with retry/backoff — Phase 7.
- [x] **Index Join Enhancements**: multi-column joins via `RowConverter` (hash-probe on encoded rows). `extract_distinct_values` now derives keys through `ManifestValue::from_array`, so joins work on every scalar key type (ints, floats, booleans, `Utf8`/`LargeUtf8`/`Utf8View`, dates, timestamps) — previously only `Int32`/`Int64`/`Utf8`, which silently returned no rows for other key types. ✅
- [x] **Time Datatype Support**: `Time32`/`Time64` write, read, range-filter, and inverted-index paths verified end-to-end (round-trips as `time32[s]`/`time64[us]`). ✅
- [x] **Primary-key operations from Python**: `add_primary_key`/`drop_primary_key` no longer panic with "Cannot start a runtime from within a runtime" — the async `_validate_pk_uniqueness` was calling the sync `read_with_columns` (which re-enters the Tokio runtime); it now uses the async read path. ✅
- [x] **Sparse vector construction**: `SparseVector(indices, values, dim)` accepts plain Python lists as well as NumPy arrays (`PyArrayLike1` + `AllowTypeChange`), matching its documented API. ✅
- [x] **Complex Range Pushdown (A1.6)**: `TableProvider::scan` recognises a same-column disjunction of ranges (`(id BETWEEN 1 AND 5) OR (id BETWEEN 50 AND 55)`, including the lowered `a >= x AND a <= y` form) and unions the per-range index bitmaps to skip segments that cannot match. Mixed-column disjunctions are left to DataFusion; pruning only happens when the column is indexed and every range yields a bitmap, so correctness never depends on the index. ✅
- [x] **Row-Value In-List Pushdown (A1.7)**: `Table::read_pk_filter_async` plus a `TableProvider::scan` hook push `(c1, c2) IN ((..), (..))` down as one expression and use the per-column inverted indexes to prune non-matching segments (guarded on *every* PK column being indexed, since a partial index would under-count matches). ✅
- [x] **Sparse Vectors — Arrow IPC serialization (A1.9)**: the `ArrowType` trait was a zero-copy `as_bytes(&[Self]) -> &[u8]`, which a variable-length `SparseVector` cannot satisfy. It is now an owned `to_bytes`/`from_bytes` pair, with a self-describing little-endian encoding for sparse vectors (`[count]` then per vector `[dim][nnz]` + `nnz`×`[index][value]`); fixed-width `f32`/`u8` still `bytemuck`-cast. `ArrowHnsw::get_vector` returns an owned `Vec<T>`. ✅
- [x] **Removed dead code**: deleted `src/core/planner/filter.rs` and `src/core/planner/vector_search.rs` — neither was declared as a module, so they were orphaned duplicates of the live `QueryFilter` / `VectorSearchParams` in `planner.rs`. ✅
- [x] **Vector Search I/O — projection pushdown**: `HybridReader::read_rows_by_id` took a `columns` parameter and ignored it (`_columns`), so a vector search projecting two columns still read **every** column of the row from Parquet. It now builds a projected schema and reads only what was requested (falling back to the full schema for unknown names). ✅
- [x] **Scored-result reader path**: `Table::execute_vector_search_as_scored` / `execute_keyword_search_as_scored` already return `ScoredResult {segment_id, row_id, score}` straight from the HNSW/BM25 indexes with no Parquet I/O, and are what `HybridSearchCoordinator::execute_hybrid` uses. ✅
- [x] **Vector Search I/O — score-only short-circuit**: a projection naming only synthesised columns (e.g. `columns=["distance"]`) used to build an *empty* Parquet projection and fail with "must either specify a row count or at least one column". The reader now treats an empty projection as the signal to skip Parquet entirely and emit the score straight from the index search (`RecordBatch::try_new_with_options` with an explicit row count). `fetch_results_by_id` short-circuits the same way for the read path. Also removed a stray `println!` from the HNSW chunk-search hot path. ✅
- [x] **Sparse Vectors → DataFusion representation**: the vector-search sort-expr parser now accepts a sparse query as a `Map<key, f32>` literal (`{'1': 0.5, '10': 0.3}`) in addition to the existing `Struct` form (`indices`/`values`/`dim`). Keys may be integer or numeric-string typed; entries are sorted by index and de-duplicated. A map cannot carry the dimension, so it is inferred as `max(index) + 1` — the Struct form is still the way to state the true dimension. ✅
- [x] **Concurrent Manifest Writes (MVCC)**: lock-free commits. The global `commit.lock` is gone from `update_schema` (pure OCC); `CommitMetadata::skip_missing_remove_paths` lets a writer rebase onto a newer snapshot when a candidate was concurrently removed (compaction uses it); `Table::snapshot_version()` exposes the monotonic snapshot id. ✅
- [x] **Cross-Partition Compaction**: re-enabled. `PartitionSpec::partition_batch` now applies the declared Iceberg transform (`identity`/`void`/`bucket`/`truncate`/`year`/`month`/`day`/`hour`) via the canonical `IcebergTransform`, so merged bins re-partition deterministically; compaction preserves the Hive partition directory; `CompactionOptions::allow_cross_partition` controls the behaviour. ✅
- [x] **Vector Search I/O**: scored results come straight from the reader. `execute_vector_search_as_scored` / `execute_keyword_search_as_scored` return `ScoredResult {segment_id, row_id, score}` with no Parquet I/O; `Table::read_rows_by_id` honours its column projection (it used to ignore it and read every column); a score-only projection (`columns=["distance"]`) skips Parquet entirely; and `vector_search_scored()` exposes it in Python. ✅
- [x] **Explain Plan Metrics for Pruning**: `QueryPlanner::classify_condition(entry, filter, emit_metrics)` returns a `PruneReason`, and `explain()` prints a ranked breakdown ("2 segment(s): partition value > max") instead of a bare count. `emit_metrics=false` on the diagnostic path so EXPLAIN doesn't inflate the operational counters. ✅
- [x] **`Table.explain` was shadowed on the Python wrapper**: `Table.__init__(..., explain: bool = False)` assigned `self.explain`, so the engine method was unreachable as `t.explain(...)` (`TypeError: 'bool' object is not callable`). The flag is now `self._explain` and `explain()` is a real method. ✅
- [x] **EXPLAIN / EXPLAIN ANALYZE as SQL**: verified working through `execute_sql` — `EXPLAIN SELECT ...` returns the logical + physical plan, `EXPLAIN ANALYZE` returns "Plan with Metrics" incl. `output_rows` / `elapsed_compute`. ✅
- [x] **Statistics pruning was completely inert** — **FIXED**. Four independent faults, any one of which was sufficient:
  1. **Parquet statistics were disabled** (`EnabledStatistics::None`), so files carried no min/max. → `Chunk`.
  2. **The manifest writer hardcoded `lower_bounds` / `upper_bounds` / `null_value_counts` to `Null`**, and the reader rebuilds `column_stats` from exactly those three. → added `encode_iceberg_value` (mirror of `decode_iceberg_value`) + `bounds_avro_values`.
  3. **The reader never unwrapped Avro's nullable-union wrapper.** `parse_map_int_long` / `parse_map_int_bytes` matched a bare `AvroValue::Array`, but a `["null", T]` field decodes as `Union(1, Box(Array(..)))` — so every stats map silently parsed as `None`. → `unwrap_nullable_union`.
  4. **`BETWEEN` produced no `QueryFilter`** (DataFusion does not lower it to `>= AND <=`), so range predicates skipped pruning entirely. → `Expr::Between` arm + shared `expr_column_name` / `expr_literal_json` helpers.
  Verified: `id >= 100` prunes 5/5 segments ("column max < filter min"), `id = 15` prunes 4/5, `id >= 40` keeps exactly the 40–49 segment, and query results are unchanged. Pinned by 3 Rust tests + 3 Python tests. ✅
- [x] **AWS Glue metadata location** *(reframed from "compute new metadata path from the snapshot")*: Glue is the only catalog whose commit API cannot hand back an authoritative metadata location — REST/Nessie return it, Hive/JDBC set it directly — so the client must supply it. The old code *reconstructed* it from the snapshot's `sequence-number`, which only matches the metadata version by coincidence. The writer now captures the path `save_to_store` returns and sends an explicit `set-metadata-location` update; Glue prefers it, falls back with a warning, and warns rather than no-op'ing when neither is available. ✅
- [x] **`BETWEEN` did not prune** — **FIXED**. The `Expr::Between` arm existed in `extract_filters_from_expr` (so `explain()` saw it), but `might_match_df_expr` — the function `prune_entries` actually calls — had no `Between` arm and fell through to `_ => true`. Added the arm; `BETWEEN 25 AND 32` now prunes 3/5 segments and results are unchanged. Pinned by `between_parses_and_prunes`. ✅
- [x] **Early Pruning for L2 Distance Scans** — **DONE**. Added `l2_distance_squared_early_exit(a, b, threshold)` (returns `None` the moment the running sum exceeds the threshold) and rewrote `vector_search_flat` to keep a streaming bounded top-k instead of full-scan-then-sort. The current k-th best distance is the early-exit threshold, so candidates that cannot enter the top-k are abandoned mid-accumulation. Ties preserve insertion order, matching the previous stable-sort-then-truncate. Pinned by 4 Rust tests incl. a streaming-vs-full-sort equivalence test. ✅
- [x] **Graph Construction Profiling Hooks** — **DONE**. Added lock-free `HnswProfile` counters (inserts, insert time, searches, search time, `search_layer` calls, distance evals) to `Hnsw`, updated from the parallel build/search paths with `Relaxed` atomics. Exposed via `Hnsw::profile()` / `profile_snapshot()` (with `avg_insert_micros` / `avg_search_micros`). Pinned by `test_profile_hooks_record_build_and_search`. ✅
- [x] **Configuration from SQL** — **DONE**. `VectorSearchConfig::extract_sql_hints` pulls the body of a `/*+ ... */` comment; `from_sql_hints` accepts both the bare `key=value` list and the `INDEX_HINT(...)` wrapper, handles all seven config keys, ignores unknown keys, and errors on malformed known values. `Table::sql` now applies parsed hints to the DataFusion session config so the vector-search optimizer rule can read them. Pinned by 7 Rust tests. ✅
- [x] **Surface the pruning breakdown in the DataFusion plan** — **DONE**. `HyperStreamExec` now carries an optional `pruning_summary` (built in `TableProvider::scan` from `QueryPlanner::classify_condition` over the pre-pruning segment list, metrics suppressed) and prints it as `pruning=[reason=count, ...]` in `DisplayAs`. `with_new_children` preserves it. One DataFusion `EXPLAIN` now shows the same reason breakdown as the engine's own `explain()`. Pinned by `test_scan_plan_surfaces_pruning_breakdown`. ✅
- [x] **First-commit segment had empty `column_stats`** — **FIXED** (found while building the above). On the very first commit `current_manifest.schemas` is still empty, so the manifest writer was handed an empty schema; `bounds_avro_values` then found no field ids, wrote no lower/upper bounds, and the reader reconstructed empty `column_stats` for that one segment — so stats pruning silently missed it. The writer now prefers `metadata.updated_schemas` (the schema being written) before falling back to the manifest's. Verified: all three segments in the test now carry stats and prune. ✅

**Next Steps**
- [ ] (none outstanding)

#### A2. Connector & Pushdown Enhancements

**Completed**
- [x] **Out-of-Core Index Ingestion**: HNSW and inverted index building use out-of-core (on-disk) processing and incremental batching. ✅ (v0.7.0)
- [x] **HNSW Hot Cache Optimization**: `IndexFileCache` stores fully deserialized `Arc<Hnsw>` graphs in memory rather than raw `Vec<u8>` bytes; kNN latency ~3-5ms. ✅ (v0.7.0)
- [x] **Trino Connector Sidecar Pushdown**: `trino-hyperstream` SPI evaluates filter predicates directly against sidecar `.hnsw` and `.idx` files before scanning parquet splits. ✅
- [x] **Micro-Batch Streaming Ingest Buffer**: Native 5–30s Iceberg snapshot buffer for streaming ingestion from Kafka and Kinesis. ✅ (v0.8.0)

**Next Steps**
- [ ] (none outstanding)

#### A3. Advanced Search & Query Features

**Completed**
- [x] **Zero-Copy Arrow IPC Vector Index**: HNSW graph traverses columnar Apache Arrow IPC structures instead of Rust heap pointers. ✅ (v0.8.0)
- [x] **Async Ingest Memory Buffer & WAL**: `_bulk` ingestion buffers documents in memory and flushes asynchronously via a Write-Ahead Log (WAL). ✅ (v0.8.0)
- [x] **TurboQuant™ Core Quantization**: Built-in scalar quantization (TQ4 / TQ8 with Fast Walsh-Hadamard Transform) for 4x memory compression. ✅ (v0.7.0)
- [x] **Composite Scalar Indexes**: Multi-column composite roaring bitmaps for frequent multi-column filter queries (e.g., `(tenant_id, status)`). ✅ (v0.7.0)
- [x] **Multi-Vector Search**: Query planner and scoring coordination across multiple embedding columns using Reciprocal Rank Fusion (RRF). ✅ (v0.7.0)

**Next Steps**
- [ ] (none outstanding)

#### A4. Native Ingest Orchestrator (tokio) — cluster-free bulk ingest
*A1 dependencies (MVCC commits, cross-partition compaction) are now complete — this is unblocked.* Spark stays for pre-write transforms and existing lake pipelines, but ingestion must not *depend* on it: a first-class `Table::ingest` / `hdb ingest` that plans, executes, and commits a bulk load entirely inside the engine.

**Completed**
- [x] **Working prototype**: whole-site Wikipedia demo fresh-process chunking (`scripts/prepare_demo.py`: 2M-row chunks, 118 s @ ~10 GB, per-chunk OCC commits).
- [x] **Work planner**: `Table::plan_ingest` enumerates parquet inputs into `(path, row_start, row_end)` work units sized by `chunk_rows` (the memory-budget knob). ✅
- [x] **Bounded worker pool**: `Table::ingest_async` runs a `buffer_unordered(parallelism)` pool; each worker reads its range and builds a *private* segment (data + indexes) via `HybridSegmentWriter`, so index builds run concurrently. ✅
- [x] **Commit strategy**: the coordinator commits each completed segment through the OCC manifest CAS (`CommitMetadata::skip_missing_remove_paths`), serialized so manifest versions stay ordered. ✅
- [x] **Resume & idempotency**: completed work-unit keys are recorded in a `_ingest_state.json` sidecar; a re-run skips them and resumes at the unit boundary. ✅
- [x] **Python surface**: `table.ingest(paths, chunk_rows=None, parallelism=None, index_all=False, resume=True, compact_after=False)` returns a report dict (`units_total/skipped/committed`, `rows_ingested`, `segments`). ✅
- [x] **CLI surface**: `hdb table ingest --uri … --input … [--plan] [--chunk-rows N] [--parallelism N] [--index-all] [--compact]`; `--row-start/--row-end` is the serverless thin-runner mode (`Table::ingest_range_async`), each runner committing independently via CAS. ✅
- [x] **Scheduled compaction**: `IngestOptions::compact_after` drives `rewrite_data_files` at the end of an ingest so segment/manifest counts stay bounded at TB scale. ✅

**Next Steps**
- [ ] **Memory discipline**: worker recycling policy (fresh process/task per memory budget) to reset glibc's unreturnable main-heap churn — or slab-allocating the HNSW/TQ builders so freed memory is actually reusable.
- [ ] **Allocator evaluation** (shared with the memory-discipline item, for long-lived daemons that rebuild indexes in-process): jemalloc vs glibc (mimalloc failed static-TLS under pyo3), `M_PURGE` on flush boundaries, or slab-allocating the HNSW/TQ builders.
- [ ] **Multi-machine mode (later)**: disjoint file subsets per node today; lease-based work stealing over an object-store lease file / Flight gateway later.

#### A5. GPU & Hardware Acceleration

**Completed**
- [x] **GPU-accelerated k-means centroid training**: `simple_kmeans` dispatches its assignment step to `gpu::compute_kmeans_assignment` when a GPU backend is usable, keeping the rayon CPU scan as the fallback. ✅
- [x] **Universal GPU PyPI Wheel**: Single universal Python wheel leveraging `cudarc` runtime dynamic loading (`libcuda.so`) and WGPU across Linux and macOS. ✅ (v0.7.0)
- [x] **GitHub Actions CUDA CI**: Automated CUDA build and test pipeline with `nvidia/cuda` Docker containers. ✅ (v0.7.0)

**Next Steps**
- [ ] **Find nvrtc/cudart from installed wheels**: cudarc 0.13.9 probes only `libnvrtc.so` / `.so.{12,11,10,1}`, so pip's `nvidia-*-cu13` layout (`nvidia/cu13/lib/libnvrtc.so.13`) is never found — the JIT path panics and silently falls back to CPU (the demo ships `scripts/create_cuda_shims.sh` and auto-re-execs with `LD_LIBRARY_PATH` as a workaround). Proper fix: resolve the library path ourselves (glob `site-packages/nvidia/*/lib`, honour `CUDA_HOME`/`CUDA_PATH`, try `.so.13`) and dlopen by absolute path, so `pip install` alone is enough — no env-var prefix.
- [ ] **GPU distance computation inside graph construction**: `hnsw_rs` builds neighbour graphs CPU-only and the CUDA backend has no HNSW kernel. Implementing GPU HNSW construction is research-grade; the realistically GPU-accelerable piece is the *distance computation during insertion (neighbor search)*, which cannot be wired in trivially because `hnsw_rs` owns that loop — it requires either a custom CUDA HNSW build or an IVF-flat GPU path for large clusters. Tracked as design work, deliberately not a dispatcher change.
- [ ] **GPU Acceleration for Sparse & Binary Vectors**: GPU acceleration is not yet implemented for sparse and binary vectors (`src/python_distance.rs`).

#### A6. Catalog & Interoperability

**Completed**
- [x] **Apache Polaris Integration**: OAuth2 client credentials grant flow (`/v1/oauth/tokens`) in `RestCatalogClient` (`src/core/catalog/rest.rs`) to support open Iceberg REST catalogs (Polaris, Lakekeeper). ✅ (v0.7.0)

**Next Steps**
- [ ] (none outstanding)

#### A7. Graph RAG & Lakehouse Graph Analytics [Free]

Native graph analytics on Iceberg edge tables with sidecar index acceleration. Replaces the need for Neo4j + Pinecone combos or Spark GraphX for knowledge graph and Graph RAG workloads. All core graph features ship in the free Community edition.

**Completed**
- [x] **Edge Table Schema Convention (5a)**: standard edge table layout (source, target, relation, weight, embedding), sidecar indexes (Roaring Bitmap on `source`/`target`, HNSW on `embedding`), and best-practices guide ([`docs/graph_rag_edge_tables.md`](docs/graph_rag_edge_tables.md)). ✅ (v0.8.0)
- [x] **Graph SQL Functions (5b)**: `PAGERANK`, `PERSONALIZED_PAGERANK`, `COMMUNITY_DETECT`, `GRAPH_NEIGHBORS`, `SUBGRAPH`, `CONNECTING_PATHS`, `NODE_SIMILARITY`, `CONNECTED_COMPONENTS`, `DEGREE_CENTRALITY`, `SHORTEST_PATH`. ✅ (v0.8.0)
- [x] **Graph RAG Pipeline Integration (5c)**: `GRAPH_RAG_SEARCH` (local/global, HippoRAG seed weighting, relation pruning, dual vector-graph), community summarization workflow (hierarchical Louvain pyramids), entity equivalence resolution (DSU). ✅ (v0.8.0)
- [x] **Python API (5d)**: `pagerank`, `personalized_pagerank`, `louvain_communities`, `graph_neighbors`, `subgraph`, `connecting_paths`, `resolve_entities`, `graph_rag_search`, `to_networkx`. ✅ (v0.8.0)
- [x] **dbt Macros (5e)**: `pagerank`, `personalized_pagerank`, `community_detect`, `graph_neighbors`, `subgraph`, `connecting_paths`, `shortest_path`, `connected_components`, `degree_centrality`, `node_similarity`, `topological_sort`. ✅ (v0.8.0)
- [x] **Search Gateway Graph Endpoints (5f)**: OpenSearch `_graph_search` DSL (Port 9200); Qdrant wire compatibility retained (Port 6333). ✅ (v0.8.0)

**Next Steps**
- [ ] (none outstanding)

#### A8. Correctness & Benchmarking Suite [Free]

Ensure that all HyperStreamDB features maintain mathematical correctness and benchmark speed against established industry standards. This prevents regressions and builds trust in the database.

**Completed**
- [x] **Graph Algorithms Suite (6a)**: accuracy validation vs NetworkX/petgraph; speed profiling (10x-50x vs NetworkX). ✅ (v0.8.0)
- [x] **Vector Search Suite (6b)**: recall vs latency benchmarks vs faiss/scikit-learn; L2/Cosine/Inner Product correctness. ✅ (v0.8.0)
- [x] **SQL Aggregates Suite (6c)**: aggregate consistency vs pandas/dask (nulls, extreme values). ✅ (v0.8.0)

**Next Steps**
- [ ] (none outstanding)

#### A9. Performance & Competitive Benchmarking

**Completed**
- [x] **100k Competitive Benchmarks vs. OpenSearch**: long-running benchmark runs on local SSD using docker-constrained environments (4 CPUs / 4GB RAM). ✅ (v0.7.0)
  - *Key Takeaways from 100K-doc benchmark*:
    - **Vector search is world-class and strictly faster**: HyperStreamDB query latencies are incredibly stable (P50: 1.94ms, P99: 4.26ms). It completely eliminates tail-latency spikes that plague OpenSearch (P99: 62.58ms), running up to 14.7x faster at the 99th percentile under tight memory constraints.
    - **Memory safety proven**: The engine safely loaded 100k HNSW vectors within the 4GB hard container limit without OOM crashing.
    - **Storage footprint**: 7.1x lower disk requirement (~26MB vs ~185MB) due to zero data lake duplication.
    - **Ingestion throughput**: OpenSearch handles bulk indexing faster (7,510 docs/s vs 4,419 docs/s) by deferring HNSW graph operations to background merges.
- [x] **1M Competitive Benchmarks vs. OpenSearch**: 1,000,000 document scaling benchmark under identical 4 CPU / 4GB RAM limits. ✅ (v0.7.0)
  - *Key Takeaways from 1M-doc benchmark*:
    - **Zero tail latency degradation**: HyperStreamDB latency remains completely flat from 100k to 1M (P50: 1.91ms, P99: 3.74ms).
    - **Catastrophic tail collapse eliminated**: OpenSearch suffers severe memory thrashing under 4GB RAM, causing P99 latency to spike to **478.77ms** (128x slower).
    - **Zero data duplication**: Requires only ~280MB storage vs OpenSearch's ~1,852MB (6.6x disk savings).
    - Documented comprehensively in [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md).

**Next Steps**
- [ ] (none outstanding)

#### A10. Packaging, Hardware & CI

**Completed**
- [x] **Universal GPU PyPI Wheel**: Single universal Python wheel leveraging `cudarc` runtime dynamic loading (`libcuda.so`) and WGPU across Linux and macOS. ✅ (v0.7.0)
- [x] **GitHub Actions CUDA CI**: Automated CUDA build and test pipeline with `nvidia/cuda` Docker containers. ✅ (v0.7.0)
- [x] **CI/CD Pipeline Maintenance**: Upgraded checkout actions to v5 for Node 24 compatibility, enforced Rust SecAudit resolutions, and DRY'd Python test workflows to use dynamically loaded wheel `[dev]` extras. ✅ (v0.8.0)

**Next Steps**
- [ ] (none outstanding)

### Part B — Ecosystem, Vertical & Commercial

#### B1. Client Ecosystem & Distribution
*Depends on a stable core API (Part A). Thin, dependency-light adapters over the existing Python API (`vector_search`, `hybrid_search`, `graph_rag_search`, `drift_search`) — no engine changes required.*

**Completed**
- [x] Cross-platform binary wheels on PyPI (`pip install hyperstreamdb`) for Linux (x86_64, aarch64) and macOS (Apple Silicon / Metal).

**Next Steps**
- [ ] **LangChain (`langchain-hyperstreamdb`)**:
  - `HyperStreamVectorStore`: Standard `VectorStore` interface (add / similarity search / MMR) mapping onto HNSW + TurboQuant sidecars, with metadata filters pushed down as RoaringBitmap predicates (`id IN (...)`).
  - `HyperStreamGraphRetriever`: `BaseRetriever` wrapping `table.graph_rag_search(...)` — local/global modes, PPR grounding, and prompt-ready `format_context()` injection.
  - Edge-table loader: Ingest documents/triplets into Iceberg doc + edge tables following the [`docs/graph_rag_edge_tables.md`](docs/graph_rag_edge_tables.md) schema convention.
- [ ] **LlamaIndex (`llama-index-vector-stores-hyperstreamdb`, `llama-index-graph-stores-hyperstreamdb`)**:
  - `HyperStreamVectorStore`: `BaseVectorStore` implementation with add/query mapped to the sidecar HNSW indexes and scalar-filter pushdown.
  - Property-graph store: `GraphStore` over edge tables (`subgraph`, `connecting_paths`, `graph_neighbors` UDAFs) enabling `PropertyGraphIndex` / HippoRAG-style retrievers on lakehouse data.
  - Two-level Graph-RAG retriever: Composite retriever mirroring the full-site Wikipedia demo pattern — 384-d seed index → CSR expansion → bitmap-filtered rerank.
- [ ] **Haystack (`hyperstream-haystack`)**:
  - `HyperStreamDocumentStore`: implement deepset's `DocumentStore` contract (`write_documents`, `filter_documents`, `delete_documents`, embedding retrieval) over HyperStreamDB tables, with metadata filters pushed down as RoaringBitmap predicates and embeddings served by the TQ HNSW indexes.
  - `HyperStreamEmbeddingRetriever`: dense/sparse (BM25) and hybrid (RRF) retrieval components usable in a Haystack pipeline.
  - Graph-RAG retriever component: wraps `graph_rag_search` (seed index → CSR expansion → bitmap-filtered rerank) for Haystack pipelines.
- [ ] **LangGraph & agent tooling** (orchestration layer — expose HyperStreamDB retrievers as graph nodes/tools):
  - Retrieval tools: `HyperStreamRetrieverTool`, `HyperStreamGraphRagTool`, `HyperStreamDriftTool` (typed tool wrappers with provenance: seed pages, PPR scores, hop paths).
  - Reference agent graph: `examples/langgraph_agentic_rag.py` — planner → hybrid retrieve → graph expand/rerank → synthesize, demonstrating agentic Graph RAG over the whole-site Wikipedia tables.
- [ ] **Pydantic AI (`hyperstream-pydantic-ai`)** — *up-and-comer track*: type-safe agent framework from the Pydantic team (the validation layer already under OpenAI SDK / LangChain). Thin adapter exposing HyperStreamDB retrievers as typed tools (`HyperStreamRetriever`, `HyperStreamGraphRagTool`) with Pydantic result models; low integration cost, high mindshare leverage with the Pydantic ecosystem.
- [ ] **Examples & Docs**: `examples/langchain_rag.py`, `examples/llamaindex_graph_rag.py`, `examples/haystack_pipeline.py`, `examples/pydantic_ai_rag.py` and the LangGraph agent above, with integration docs.

> **Framework prioritization (avoid sprawl):** cover the **most popular** first — LangChain + LlamaIndex are the must-haves, Haystack for enterprise RAG, LangGraph for orchestration — then add **one up-and-comer** (Pydantic AI) to leapfrog incumbents via the Pydantic ecosystem. All are thin adapters over the existing Python API, so each is cheap; cap the list here and add further frameworks only on demonstrated user demand.

#### B2. Codebase Intelligence & Model Context Protocol (MCP) Server
*Depends on core + client ecosystem.*

**Completed**
- [ ] (none outstanding)

**Next Steps**
- [ ] **MCP Server Implementation (`hyperstream-mcp`)**:
  - Protocol Support: Standard Model Context Protocol (JSON-RPC over stdio and SSE). (v0.9.0)
  - Tool: `code_search`: Hybrid BM25 (exact symbols/keywords) + HNSW vector search over codebase chunks. (v0.9.0)
  - Tool: `find_symbol`: Sub-millisecond exact definition and reference lookups powered by String Inverted Index. (v0.9.0)
  - Tool: `get_context`: Extract relevant code blocks, AST parent contexts, and neighboring functions. (v0.9.0)
  - Tool: `code_graph`: Query imports, calls, and dependency relationships via sidecar graph tables. (v0.9.0)
  - Language Parsers: Tree-sitter integration for AST-aware semantic chunking (Rust, Python, TS/JS, Go, Java, C++). (v0.9.0)
- [ ] **Git-Diff Incremental CI Indexer**:
  - CLI Subcommand `hyperstream index`: `--repo <path>`, `--diff-since <ref>`, `--target <uri>`. (v0.9.0)
  - Incremental Parquet & Overlay Appends: Write new code chunks and vector embeddings directly as an append delta; tombstone deleted chunks via Roaring Bitmaps. (v0.9.0)
  - Official GitHub Action (`hyperstreamdb/index-action@v1`): Ready-to-use GitHub Action for PR and merge workflows. (v0.9.0)
  - GitLab CI & Jenkins Examples: Provide standard CI pipeline configurations. (v0.9.0)
- [ ] **Feature Tiering: Local vs. Remote Lakehouse**:
  - [Free] Local Storage Backends: Direct support for local filesystem (`file://`) and developer MinIO instances. (v0.9.0)
  - [Free] Local MCP Server & Tooling: Full stdio/SSE MCP protocol support for local developer desktop tools (Cursor, Claude, Roo Code). (v0.9.0)
  - [Free] Git-Diff Incremental Indexing Engine: Fast incremental AST chunking and overlay generation on individual developer machines. (v0.9.0)
  - [Paid] Remote Cloud Object Storage Integration: Direct synchronization to cloud object storage (`s3://`, `gs://`, `az://`, `r2://`). (v0.9.0)
  - [Paid] Centralized Team Knowledge Cache: Shared team repository index across engineering organizations with access control and pre-computed embedding distribution. (v0.9.0)

#### B3. Vertical Lighthouse: High-Cardinality Scale Benchmark
*Depends on core high-cardinality filtering (A1) + ingest orchestrator (A4). Retained as the scale lighthouse + research platform (see [`business_plan/README.md`](business_plan/README.md)).*

**Completed**
- [x] **Raw scale-testing objective** — satisfied by the full-site Wikipedia Graph RAG demo ([`examples/web_ui/README.md`](examples/web_ui/README.md)): 51.8M live pages / 383M clean int64 edges, HNSW-TQ8 + BM25 (RRF `hybrid_search`), CSR graph index, PageRank/PPR, Louvain, and two-level Graph RAG (seed index → CSR expansion → bitmap-filtered rerank), with chunked fresh-process ingest, OCC commits, and compaction ([`scripts/prepare_demo.py`](scripts/prepare_demo.py)).

**Next Steps**
- [ ] High-cardinality scalar pre-filtering benchmark (millions of distinct keys / temporal predicates) against RoaringBitmap sidecars.
- [ ] End-to-end verification of hybrid scalar-vector queries under high data skew and temporal partitioning.
- [ ] Zero-copy PyTorch tensor feeding via Arrow Flight SQL gateway for deep learning feature pipelines.

#### B4. Enterprise Security & Compliance [Paid]
*Depends on core maturity (Part A).*

**Completed**
- [ ] (none outstanding)

**Next Steps**
- [ ] **[Paid] Row-Level Security (RLS) & Multi-Tenancy**: Sidecar-level tenant bitmap isolation (`.idx` intersection before reading Parquet).
- [ ] **[Paid] Dynamic Column Masking**: Role-based PII redaction on query and vector results.
- [ ] **[Paid] Customer-Managed Encryption Keys (CMEK)**: Envelope encryption for sidecar index files via AWS KMS, GCP KMS, or HashiCorp Vault.
- [ ] **[Paid] Cryptographic Audit Logging**: Tamper-evident hash chain recording queries across all three protocols (9200, 6333, 50051).
- [ ] **[Paid] SIEM Telemetry Export**: Native connector export to Splunk, Datadog, and AWS CloudWatch.
- [ ] **[Paid] Cross-Catalog Governance Propagation**: Unified RLS policies and audit synchronization across Polaris, Unity, and Glue catalogs.

#### B5. HyperStream Accelerator & Lifecycle Automation [Paid]
*Depends on core maturity (Part A).*

**Completed**
- [ ] (none outstanding)

**Next Steps**
- [ ] **[Paid] Fused SIMD & Tensor Core Kernels**: Hand-crafted AVX-512, ARM SVE, and Hopper/Blackwell FP8/FP4 fused kernels.
- [ ] **[Paid] GPUDirect Storage (GDS) Bypass**: Direct NVMe/S3 local cache streaming to GPU VRAM, bypassing host CPU/PCIe bottleneck.
- [ ] **[Paid] Sidecar Lifecycle Manager**: Autonomous 3-format coordinated compaction (Iceberg manifests + Parquet bin-packing + HNSW/Bitmap sidecars) with cost-aware S3 scheduling and recall drift rebalancing.

---

## 📣 Marketing & Go-To-Market

> **Trigger:** The full-site Wikipedia Graph RAG demo works end-to-end and the core engine (Phases 1–10) is complete. This is the moment to start public marketing — the product has a credible, reproducible flagship artifact and a stable core to point at.

### Positioning
- **Horizontal data infrastructure**, not a vertical app: Apache Iceberg V2/V3 + RoaringBitmap + HNSW/TQ8 overlays, exposed over OpenSearch / Qdrant / Arrow Flight SQL.
- **The proof point:** *"51.8M Wikipedia pages / 383M edges — hybrid vector + keyword + Graph RAG, served on a laptop under bounded RAM, with zero data duplication."*
- **Competitive wedge:** flat tail latency vs OpenSearch (P99 3.74ms vs 478.77ms at 1M docs), 6.6x–7.1x lower disk, no JVM / no OOM.

### Channels & Cadence
- **LinkedIn** (founder-led, 2–3 posts/week):
  - Short benchmark teardowns (P50/P99 charts, disk footprint) linking to the demo repo.
  - "Build in public" progress notes (ingest orchestrator, Graph RAG, 4 GB matrix).
  - Demo screen-captures (Browse / Semantic / Graph RAG / DRIFT tabs).
- **Medium** (long-form, 1–2/month):
  - "Running the entire English Wikipedia link graph on a laptop" — architecture + measured timings.
  - "Why tail latency is the real vector-search benchmark" — HyperStreamDB vs OpenSearch at 1M docs.
  - "Graph RAG on the lakehouse: seed index → CSR expansion → bitmap-filtered rerank."
  - "Iceberg-native vector search without a separate vector database."
- **Secondary**: Hacker News (Show HN), r/dataengineering, DuckDB / Iceberg communities, X/Twitter threads.

### Content Assets to Produce
- [ ] Benchmark chart pack (P50/P99 latency, disk footprint, ingest throughput) from [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md).
- [ ] 2–3 min demo video of the full-site Wikipedia Graph RAG UI ([`examples/web_ui/app.py`](examples/web_ui/app.py)).
- [ ] Reproducible "one-command" quickstart gist (`pip install hyperstreamdb` + 3 lines).
- [ ] Comparison one-pager vs OpenSearch / LanceDB / pgvector.

### Guardrails
- Keep `hyperstreamdb` a pristine, domain-agnostic Apache 2.0 engine; vertical/domain-specific work stays in separate downstream repos.
- Market the horizontal engine on measured, reproducible benchmarks only.

---

## Phase 9: Resource-Constrained Vector Benchmarking (4 GB RAM Matrix) ✅ COMPLETE

### Objectives
- Validate sustained vector ingestion and sub-second hybrid query latency under strict container memory limits (`docker run --memory=4g --cpus=4`).
- Document zero OOMs, zero GC pauses, and >95% Recall@10 on 1M vectors (768-dim Cohere/DBPedia and 64-dim Matryoshka embeddings).
- Formalize HNSW-aware hierarchical LRU caching (pinning upper layers $L > 0$ and RoaringBitmap metadata; evicting layer-0 raw vector chunks).

### Tasks
- [x] Reproducible Docker benchmark harness comparing HyperStreamDB (TQ8/TQ4) vs OpenSearch 2.x/3.x and LanceDB.
- [x] Automated measurement of RSS memory ceilings, ingest throughput (vectors/sec), and p95/p99 query latency.
- [x] Formal LRU cache budget configuration guide (`HYPERSTREAM_CACHE_CAP_BYTES`) ensuring predictable memory bounds on edge/container hosts.

---

## Phase 10: Streaming Commit, Delete Lifecycle & Concurrency Verification ✅ COMPLETE

### Objectives
- Formally verify HNSW index overlay stability across immutable Iceberg snapshot commits, partition splits, and position/equality deletes.
- Prove that position delete files mask deleted rows via RoaringBitmap masks during HNSW graph traversal without corrupting graph connectivity.

### Tasks
- [x] Integration test suite for Iceberg V2 position delete masking in vector graph scans (`tests/verify_mor_vector_deletes.rs`).
- [x] Incremental sidecar index append vs. compaction coordination under concurrent streaming writes.
- [x] Architecture documentation detailing the interaction between persistent HNSW overlays and Iceberg transaction manifests.

---

## Success Metrics

### Performance (Measured 2026-01-18)
| Metric | Target | Achieved | Notes |
|--------|--------|----------|-------|
| Ingest throughput | >100K rows/sec | **753K rows/sec** | NYC Taxi dataset |
| Query (indexed, p99) | <100ms | **85ms** | High-selectivity filter |
| Vector search | <50ms for k=10 | **~500ms/segment** | Use scalar filters to prune segments |
| Vector search (parallel) | <10s | **5.0s** | 100K vectors, 10 segments, 16 parallel readers |
| Compaction | <5min per 10GB | **4.91s** | 3M rows (~200MB) |

### Reliability
- ✅ Zero data loss (ACID writes via manifest versioning)
- ✅ Atomic commits (manifest-based transactions)
- ⬜ 99.9% uptime (requires production deployment)

### Usability
- ✅ <5 min to first query (single pip install + 3 lines of code)
- ✅ Pandas-compatible API (`table.to_pandas()`)
- ✅ Iceberg-compatible connectors (`spark-hyperstream` & `trino-hyperstream`)

---

## 🗺️ Roadmap Reconciliation & Status Summary

All core foundation phases (Phases 1–8) are **COMPLETE and verified in code**:
- **Phase 1: Real-World Dataset Benchmarks** — NYC Taxi (753k rows/s), Wikipedia (100k docs), 768D BERT embeddings.
- **Phase 2: Nessie Catalog Integration** — Git-like table branching and multi-table transactions.
- **Phase 3 & 3.5: Performance & Native DataFusion SQL Engine** — MoR/CoW deletion vectors, partition pruning, Index Nested Loop Joins, pgvector operators (`<->`, `<=>`, `<#>`).
- **Phase 4.5: Multi-Catalog Abstraction** — REST, AWS Glue, Hive Metastore, Unity Catalogs.
- **Phase 5: Connectors & Distributed Analytics** — Spark DataSource V2, Trino SPI connector, and split-level byte-range parallelism.
- **Phase 6: Operational Tooling & Observability** — `hdb` CLI REPL, `tracing-opentelemetry`, Prometheus metrics exporter (`/metrics`).
- **Phase 6.5: Ecosystem Gateways** — Dual-protocol search server (`hyperstreamdb-search` on ports 9200 & 6333), Arrow Flight SQL gateway (`hyperstreamdb-flight` on port 50051), and official dbt adapter (`dbt-hyperstreamdb`).
- **Phase 7: Cloud-Agnostic Concurrency & Durability** — `FileBasedLock` (`src/core/lock.rs`) using object storage CAS (`PutMode::Create`), OCC snapshot swaps with retries (`src/core/manifest/manager/commit.rs`), chaos testing (`tests/test_chaos.rs`).
- **Phase 8: Documentation Suite** — Complete Sphinx / ReadTheDocs setup in `docs/` with developer guides for SQL, Python, Iceberg V2/V3, GPU, and Concurrency.
- **Phase 9: 4 GB RAM Matrix** — Docker-constrained vector benchmarking vs OpenSearch/LanceDB.
- **Phase 10: Streaming Commit & Delete Lifecycle** — HNSW overlay stability across snapshots, partition splits, and position/equality deletes.

**Active Roadmap (core-first, dependency-ordered)** — see the [Active Roadmap](#-active-roadmap--core-product-first-dependency-ordered) section above:
- **Part A — Core Product**: A1 Core Engine Correctness & Concurrency → A2 Connector & Pushdown → A3 Advanced Search → A4 Native Ingest Orchestrator → A5 GPU & Hardware → A6 Catalog → A7 Graph RAG → A8 Correctness Suite → A9 Competitive Benchmarking → A10 Packaging & CI.
- **Part B — Ecosystem, Vertical & Commercial**: B1 Client Ecosystem → B2 Codebase Intelligence & MCP → B3 High-Cardinality Scale Lighthouse → B4 Enterprise Security [Paid] → B5 Accelerator & Lifecycle [Paid].

---

## Questions & Decisions

### ✅ Resolved
- **Catalog:** Pluggable multi-catalog (Nessie, REST, Glue, Hive, Unity)
- **Manifest Format:** Avro/JSON (semantic Iceberg V2/V3 compatibility)
- **Distributed Locking:** Vendor-neutral `FileBasedLock` using object storage CAS (`PutMode::Create`) with heartbeats & leases (no proprietary services like DynamoDB). Commits themselves are **lock-free** (OCC + rebase); the lock is only used for maintenance (expire/vacuum) serialization.
- **Filtering Style:** Pushdown via sidecar inverted and roaring bitmap indexes
- **Pruning:** Partition pruning + column-statistics pruning (min/max persisted through the Avro manifest as Iceberg `lower_bounds`/`upper_bounds`/`null_value_counts`), with per-reason reporting in `explain()`
- **Vector Index:** Standardized on HNSW-IVF with GPU acceleration (`cudarc` for CUDA, WGPU for Vulkan/Metal/DirectX)
- **SQL & Analytics:** DataFusion native integration + Arrow Flight SQL gateway + dbt adapter (`dbt-hyperstreamdb`)
- **REST APIs:** OpenSearch / Elasticsearch 7.10 + Qdrant compatibility via `hyperstreamdb-search`

### 🤔 Open
- Distributed compaction strategy (Spark job vs local async daemon)?
- Polaris catalog credential refresh token lifecycles?
- Graph RAG: Leiden vs. Louvain for community detection default? (Leiden is newer but more complex to implement)
- Graph RAG: Should `PAGERANK` return results as a materialized sidecar or as a transient DataFrame?

> **Note:** The former "Technical Debt & Missing Features" list has been folded into **A1. Core Engine Correctness & Concurrency** (next steps) so all outstanding core work lives in one dependency-ordered place.

---

**Last Updated:** 2026-09-22
**Status:** Phases 1–10 COMPLETE ✅ | **A1 Core Engine Correctness & Concurrency COMPLETE ✅** (MVCC lock-free commits, cross-partition compaction, index-join key types, row-value & OR-range pushdown, statistics pruning fixed, Time32/64, sparse IPC + SQL maps, Glue metadata location) | Active: Part A Core Product (Native Ingest Orchestrator — unblocked) + Part B Client Ecosystem | Planned: High-Cardinality Scale Lighthouse (scale objective satisfied by the full-site Wikipedia demo), Codebase Intelligence & MCP, Enterprise & Accelerator tiers
