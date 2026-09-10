# Real-World Testing & Production Readiness Plan

## Overview

This document outlines the step-by-step plan to take HyperStreamDB from PoC to production-ready.

**Timeline:** ~8 weeks  
**Current Phase:** Phases 1–8 COMPLETE ✅ | Active Roadmap: Polaris OAuth2, Trino Sidecar Pushdown & Multi-Vector Search

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
- [ ] Support OAuth2 client credentials flow in REST Catalog for Apache Polaris integration

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

## 🎯 Active Roadmap & Remaining Milestones

The following items represent the active, vetted roadmap for HyperStreamDB. Speculative dead ends (such as proprietary cloud locks, bespoke C++ database extensions, or third-party format readers) have been pruned in favor of standards-based interoperability.

### 1. Catalog & Interoperability
- [x] **[Free] Apache Polaris Integration**: Add OAuth2 client credentials grant flow (`/v1/oauth/tokens`) in `RestCatalogClient` (`src/core/catalog/rest.rs`) to support open Iceberg REST catalogs (Polaris, Lakekeeper). ✅ (v0.7.0)

### 2. Performance & Competitive Benchmarking
- [ ] **[Free] 100k / 1M Competitive Benchmarks vs. Elasticsearch 7.10**: Execute long-running benchmark runs on NVMe and MinIO S3 storage using `benchmarks/competitive/benchmark_es710.py` and document findings.

### 3. Connector & Pushdown Enhancements
- [ ] **[Free] Trino Connector Sidecar Pushdown**: Enhance `trino-hyperstream` SPI implementation to evaluate filter predicates directly against sidecar `.hnsw` and `.idx` files before scanning parquet splits.
- [ ] **[Free] Micro-Batch Streaming Ingest Buffer**: Native 5–30s Iceberg snapshot buffer for streaming ingestion from Kafka and Kinesis.

### 4. Advanced Search & Query Features
- [x] **[Free] TurboQuant™ Core Quantization**: Built-in scalar quantization (TQ4 / TQ8 with Fast Walsh-Hadamard Transform) for 4x memory compression in core open-source engine. ✅ (v0.7.0)
- [x] **[Free] Composite Scalar Indexes**: Multi-column composite roaring bitmaps for frequent multi-column filter queries (e.g., `(tenant_id, status)`). ✅ (v0.7.0)
- [x] **[Free] Multi-Vector Search**: Query planner and scoring coordination to search and rank across multiple embedding columns simultaneously using Reciprocal Rank Fusion (RRF). ✅ (v0.7.0)

### 5. Graph RAG & Lakehouse Graph Analytics [Free]

Native graph analytics on Iceberg edge tables with sidecar index acceleration. Replaces the need for Neo4j + Pinecone combos or Spark GraphX for knowledge graph and Graph RAG workloads. All core graph features ship in the free Community edition.

#### 5a. [Free] Edge Table Schema Convention
- [ ] **[Free] Standard Edge Table Layout**: Define standard Iceberg edge table schema (source_id, target_id, relation, weight, embeddings).
- [ ] **[Free] Sidecar Indexes**: Auto-generate sidecar indexes on `source_id` and `target_id` columns (Roaring Bitmap) for O(1) edge lookups.
- [ ] **[Free] Best Practices Guide**: Document edge table conventions (partitioning by relation type, sort order by source_id).

#### 5b. [Free] Graph SQL Functions (DataFusion UDFs)
- [ ] **[Free] `PAGERANK(edge_table, damping, max_iterations, tolerance)`**: Iterative PageRank over edge table.
- [ ] **[Free] `COMMUNITY_DETECT(edge_table, algorithm, resolution)`**: Louvain / Label Propagation community detection.
- [ ] **[Free] `GRAPH_NEIGHBORS(entity_id, edge_table, hops, direction)`**: 1–N hop neighborhood retrieval.
- [ ] **[Free] `NODE_SIMILARITY(node_a, node_b, edge_table, method)`**: Jaccard and overlap similarity via sidecar bitmap intersection.
- [ ] **[Free] `CONNECTED_COMPONENTS(edge_table)`**: Component labeling via iterative label propagation.
- [ ] **[Free] `DEGREE_CENTRALITY(edge_table, direction)`**: In-degree, out-degree, and total degree aggregation.

#### 5c. [Free] Graph RAG Pipeline Integration
- [ ] **[Free] `GRAPH_RAG_SEARCH(query_embedding, edge_table, doc_table, mode, community_col)`**: Combined graph + vector search (local and global modes).
- [ ] **[Free] Community Summarization Workflow**: SQL-driven pipeline to GROUP BY community_id and produce summary embeddings for global search.

#### 5d. [Free] Python API
- [ ] **[Free] `table.pagerank(damping=0.85, iterations=30)`**: DataFrame with PageRank scores.
- [ ] **[Free] `table.communities(algorithm='louvain', resolution=1.0)`**: Community assignments.
- [ ] **[Free] `table.graph_neighbors(entity_id, hops=2)`**: Neighbor entities + edges.
- [ ] **[Free] `table.graph_rag_search(query, mode='local', hops=2, top_k=10)`**: Combined graph + vector results.
- [ ] **[Free] `table.to_networkx()`**: Export to NetworkX `DiGraph` for ecosystem visualization.

#### 5e. [Free] dbt Macros (`dbt-hyperstreamdb`)
- [ ] **[Free] `{{ pagerank(ref('edges'), damping=0.85) }}`**: Materialize PageRank scores as an Iceberg table.
- [ ] **[Free] `{{ community_detect(ref('edges'), algorithm='louvain') }}`**: Materialize community assignments.
- [ ] **[Free] `{{ graph_neighbors(ref('edges'), entity_id, hops=2) }}`**: Neighborhood subgraph extraction.

#### 5f. [Free] Search Gateway Graph Endpoints
- [ ] **[Free] Qdrant API (Port 6333)**: Extend `/points/search` with `graph_filter` parameter for neighborhood-scoped vector search.
- [ ] **[Free] OpenSearch API (Port 9200)**: Extend `_search` DSL with `graph_neighbors` filter clause.

### 6. Packaging, Hardware & CI
- [ ] **[Free] Universal GPU PyPI Wheel**: Distribute a single universal Python wheel leveraging `cudarc` runtime dynamic loading (`libcuda.so`) and WGPU across Linux and macOS.
- [ ] **[Free] GitHub Actions CUDA CI**: Automated CUDA build and test pipeline with `nvidia/cuda` Docker containers.

### 7. Codebase Intelligence & Model Context Protocol (MCP) Server

#### 7a. [Free] MCP Server Implementation (`hyperstream-mcp`)
- [ ] **[Free] Protocol Support**: Standard Model Context Protocol (JSON-RPC over stdio and SSE).
- [ ] **[Free] Tool: `code_search`**: Hybrid BM25 (exact symbols/keywords) + HNSW vector search over codebase chunks.
- [ ] **[Free] Tool: `find_symbol`**: Sub-millisecond exact definition and reference lookups powered by String Inverted Index.
- [ ] **[Free] Tool: `get_context`**: Extract relevant code blocks, AST parent contexts, and neighboring functions.
- [ ] **[Free] Tool: `code_graph`**: Query imports, calls, and dependency relationships via sidecar graph tables.
- [ ] **[Free] Language Parsers**: Tree-sitter integration for AST-aware semantic chunking (Rust, Python, TS/JS, Go, Java, C++).

#### 7b. [Free] Git-Diff Incremental CI Indexer
- [ ] **[Free] CLI Subcommand `hyperstream index`**:
  - `--repo <path>`: Target repository directory.
  - `--diff-since <ref>`: Git diff mode (e.g. `HEAD~1`, `origin/main`) to only re-index changed files.
  - `--target <uri>`: Target storage URI (`file:///...`, `s3://...`).
- [ ] **[Free] Incremental Parquet & Overlay Appends**: Write new code chunks and vector embeddings directly as an append delta; tombstone deleted chunks via Roaring Bitmaps.
- [ ] **[Free] Official GitHub Action (`hyperstreamdb/index-action@v1`)**: Ready-to-use GitHub Action for PR and merge workflows.
- [ ] **[Free] GitLab CI & Jenkins Examples**: Provide standard CI pipeline configurations.

#### 7c. Feature Tiering: Local vs. Remote Lakehouse
- [ ] **[Free] Local Storage Backends**: Direct support for local filesystem (`file://`) and developer MinIO instances.
- [ ] **[Free] Local MCP Server & Tooling**: Full stdio/SSE MCP protocol support for local developer desktop tools (Cursor, Claude, Roo Code).
- [ ] **[Free] Git-Diff Incremental Indexing Engine**: Fast incremental AST chunking and overlay generation on individual developer machines.
- [ ] **[Paid] Remote Cloud Object Storage Integration**: Direct synchronization to cloud object storage (`s3://`, `gs://`, `az://`, `r2://`).
- [ ] **[Paid] Centralized Team Knowledge Cache**: Shared team repository index across engineering organizations with access control and pre-computed embedding distribution.

### 8. Enterprise Security & Compliance [Paid]
- [ ] **[Paid] Row-Level Security (RLS) & Multi-Tenancy**: Sidecar-level tenant bitmap isolation (`.idx` intersection before reading Parquet).
- [ ] **[Paid] Dynamic Column Masking**: Role-based PII redaction on query and vector results.
- [ ] **[Paid] Customer-Managed Encryption Keys (CMEK)**: Envelope encryption for sidecar index files via AWS KMS, GCP KMS, or HashiCorp Vault.
- [ ] **[Paid] Cryptographic Audit Logging**: Tamper-evident hash chain recording queries across all three protocols (9200, 6333, 50051).
- [ ] **[Paid] SIEM Telemetry Export**: Native connector export to Splunk, Datadog, and AWS CloudWatch.
- [ ] **[Paid] Cross-Catalog Governance Propagation**: Unified RLS policies and audit synchronization across Polaris, Unity, and Glue catalogs.

### 9. HyperStream Accelerator & Lifecycle Automation [Paid]
- [ ] **[Paid] Fused SIMD & Tensor Core Kernels**: Hand-crafted AVX-512, ARM SVE, and Hopper/Blackwell FP8/FP4 fused kernels.
- [ ] **[Paid] GPUDirect Storage (GDS) Bypass**: Direct NVMe/S3 local cache streaming to GPU VRAM, bypassing host CPU/PCIe bottleneck.
- [ ] **[Paid] Sidecar Lifecycle Manager**: Autonomous 3-format coordinated compaction (Iceberg manifests + Parquet bin-packing + HNSW/Bitmap sidecars) with cost-aware S3 scheduling and recall drift rebalancing.

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

---

## Questions & Decisions

### ✅ Resolved
- **Catalog:** Pluggable multi-catalog (Nessie, REST, Glue, Hive, Unity)
- **Manifest Format:** Avro/JSON (semantic Iceberg V2/V3 compatibility)
- **Distributed Locking:** Vendor-neutral `FileBasedLock` using object storage CAS (`PutMode::Create`) with heartbeats & leases (no proprietary services like DynamoDB)
- **Filtering Style:** Pushdown via sidecar inverted and roaring bitmap indexes
- **Vector Index:** Standardized on HNSW-IVF with GPU acceleration (`cudarc` for CUDA, WGPU for Vulkan/Metal/DirectX)
- **SQL & Analytics:** DataFusion native integration + Arrow Flight SQL gateway + dbt adapter (`dbt-hyperstreamdb`)
- **REST APIs:** OpenSearch / Elasticsearch 7.10 + Qdrant compatibility via `hyperstreamdb-search`

### 🤔 Open
- Distributed compaction strategy (Spark job vs local async daemon)?
- Polaris catalog credential refresh token lifecycles?
- Graph RAG: Leiden vs. Louvain for community detection default? (Leiden is newer but more complex to implement)
- Graph RAG: Should `PAGERANK` return results as a materialized sidecar or as a transient DataFrame?

---

**Last Updated:** 2026-09-09  
**Status:** Phases 1–8 COMPLETE ✅ | Active Next: Polaris REST OAuth2, Trino Sidecar Pushdown, Multi-Vector Search & Graph RAG  
