# Real-World Testing & Production Readiness Plan

## Overview

This document outlines the step-by-step plan to take HyperStreamDB from PoC to production-ready.

**Timeline:** ~8 weeks  
**Current Phase:** Phase 5 COMPLETE ✅ | Next: Phase 6 - Operational Tooling

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

## Phase 2: Nessie Integration (Week 3)

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

## Phase 3: Performance Optimization (Weeks 4-5)

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
- [ ] Compare before/after metrics

- [ ] Compare before/after metrics

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

## Phase 4.5: Multi-Catalog Support (Weeks 6-7)

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
- [ ] Support OAuth2 client credentials flow in REST Catalog for Apache Polaris integration

---

## Phase 5: Spark/Trino Connector APIs (Weeks 8-9)

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
- [ ] Add partition support (optional)
- [x] Integration tests for file/split APIs
- [ ] Benchmark parallelism improvements

### Connector Development (Post-API)
- [ ] Spark DataSource V2 connector (Java/Scala)
- [ ] Trino Connector SPI implementation (Java)
- [ ] dbt adapter (Python, via Trino/Spark)

---

## Phase 6: Operational Tooling (Week 10)

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
- [ ] Add Prometheus metrics
- [ ] Add tracing spans
- [ ] Create Grafana dashboards
- [ ] Document monitoring setup

---

## Phase 7: Production Hardening (Week 11)

### Objectives
- Error handling & retries
- Distributed locking
- Data validation

### Hardening

#### 1. Error Handling
- Replace `println!` with `tracing`
- Add retry logic for S3
- Graceful degradation (index unavailable → full scan)
- Circuit breakers

#### 2. Concurrency Control
- DynamoDB-based distributed locks
- Optimistic concurrency for manifests
- Conflict resolution

#### 3. Data Validation
- Manifest integrity checks
- File existence validation
- Checksum verification

### Tasks
- [ ] Improve error handling
- [ ] Implement distributed locking
- [ ] Add data validation
- [ ] Chaos testing
- [ ] Load testing

---

## Phase 8: Documentation (Week 12)

### Objectives
- User documentation
- API reference
- Architecture docs
- ReadTheDocs setup

### Documentation Structure

#### 1. ReadTheDocs (Sphinx)
```
docs/
├── conf.py              # Sphinx configuration
├── index.rst            # Homepage
├── getting_started.rst  # Quick start
├── api/
│   ├── table.rst       # Table API
│   ├── catalogs.rst    # Catalog API (Nessie, REST, Glue, Unity)
│   └── sql.rst         # SQL API
├── guides/
│   ├── catalogs.rst    # Catalog comparison & usage
│   ├── indexing.rst    # Indexing strategies
│   └── performance.rst # Performance tuning
└── requirements.txt
```

#### 2. Rust Documentation (docs.rs)
- Auto-generated from `///` doc comments
- Published when crate is released to crates.io

### Tasks
- [ ] Set up Sphinx for ReadTheDocs
- [ ] Write getting started guide
- [ ] Document all catalog types (Nessie, REST, Glue, Unity)
- [ ] API reference (auto-generated from docstrings)
- [ ] Architecture documentation
- [ ] Performance benchmarks
- [ ] Add Rust doc comments (`///`)
- [ ] Configure ReadTheDocs build
- API reference
- Example applications

### Documentation

#### 1. User Docs
- Quickstart guide
- Architecture overview
- Performance tuning
- Troubleshooting

#### 2. API Reference
- Rust API docs (rustdoc)
- Python API docs (Sphinx)
- REST API spec (OpenAPI)

#### 3. Examples
- Vector search application
- Hybrid query examples
- Spark/Trino integration

### Tasks
- [ ] Write quickstart guide
- [ ] Generate API docs
- [ ] Create example apps
- [ ] Record demo videos
- [ ] Publish documentation site

---

---

## Future Phases (Q2 2026 - Federated Connectors)

### Objectives
- Support read-only federation for Iceberg, Hudi, and Delta Lake
- Build sidecar indexes for external data lakes
- Unified query layer

### Tasks
- [ ] Implement Hudi Table Reader
- [ ] Implement Delta Lake Reader
- [ ] Build "Index-Only" ingestion job (skips data copy)
- [ ] Add federated integration tests
- [ ] Feature: "Overlay Indexing" (Sidecar) for existing Iceberg tables
- [ ] Feature: "Universal Indexing" for Hudi & Delta Lake (XTable-style compatibility)
- [ ] Apache Polaris Integration: Add OAuth2 client credentials auth support to REST catalog for Apache Polaris compliance

### Index-Accelerated Engine Integrations
- [ ] **Arrow Flight SQL Gateway**: Build a unified, high-performance Arrow Flight SQL server enabling any JDBC/ODBC-compatible engine (Doris, StarRocks, Dremio) to leverage HyperStreamDB's index-accelerated query execution.
- [ ] **DuckDB Extension**: Develop a native `duckdb_hyperstream` extension in Rust/C++ to enable local DuckDB instances to scan tables using HyperStreamDB's indexes.
- [ ] **StarRocks/Doris C++ UDFs**: Create native UDFs for vector index search (`hyperstream_knn`) to run high-performance KNN queries inside MPP databases.
- [ ] **Trino Connector Predicate Pushdown**: Enhance the Trino connector to resolve filter predicates against sidecar `.hnsw` and `.idx` files before scanning parquet splits.

### Multi-Column Indexes & Composite Search
- [ ] **Composite Scalar Indexes**: Pre-computed roaring bitmaps for common multi-column filters (e.g. `(city, zip)`).
- [ ] **Multi-Vector Search**: Simultaneous ranking across multiple embedding columns with combined scoring.
- [ ] **Index-Accelerated Joins**: Phase 2 optimizations for multi-table composite filtering.

### GPU Build & Distribution Improvements
- [ ] **Replace `cust` with `cudarc`**: Migrate CUDA backend from `cust` (requires CUDA SDK at compile time) to `cudarc` (dynamically loads `libcuda.so` at runtime). This enables a single universal wheel with runtime CUDA detection — no more source builds for CUDA users.
- [ ] **Universal GPU wheel**: Once `cudarc` is integrated, ship one PyPI wheel that auto-detects CUDA, MPS, XPU, and ROCm at runtime via WGPU.
- [ ] **GitHub Actions CUDA CI**: Add CI job using `nvidia/cuda` Docker image to build and test CUDA-enabled wheels.


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
- ⬜ Iceberg-compatible connectors (Phase 3)

---

## Next Steps

**Phase 1 COMPLETE (2026-01-18):**
- ✅ NYC Taxi: 753K rows/sec ingest, 85ms indexed query
- ✅ Vector Search: 5.0s parallel search (100K vectors, 16 workers)
- ✅ Wikipedia Hybrid: Scalar + vector queries working

**Next Priorities:**
1. ✅ Implement inverted index for string columns (Completed)
2. ✅ Add native hybrid query support (scalar + vector in single query)
3. ✅ Continue Phase 3 optimizations (Iceberg-compatible API)
4. ✅ Phase 3.5: Native SQL Support (Parallel Scanning, Filter Pushdown)
5. ✅ Phase 4.5: Multi-catalog support (REST, Glue, Hive, Unity)
6. ✅ Phase 5: Spark/Trino connector APIs (file-level, split-level)
7. Phase 6: Operational tooling (Metrics, Observability)

---

## Questions & Decisions

### ✅ Resolved
- **Catalog:** Support multiple (Nessie, REST, Glue, Unity)
- **Manifest format:** JSON (semantic Iceberg compat)
- **Filtering style:** Polars-like (predicate pushdown)
- **Inverted Index Format:** Parquet (replaced JSON for 12x performance boost)
- **Vector Index:** Standardized on HNSW-IVF (removed legacy types)

### 🤔 Open
- Partition strategy for large tables?
- Distributed compaction (Spark job vs local)?
- Schema evolution migration path?

---

**Last Updated:** 2026-01-19
**Last Updated:** 2026-01-25
**Status:** Phase 5 COMPLETE ✅ | Next: Phase 6 - Operational Tooling
