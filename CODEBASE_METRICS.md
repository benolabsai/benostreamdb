# HyperStreamDB Codebase Line Count & Test Analysis

*Generated: September 10, 2026*

---

## 1. Executive Summary

| Metric | Count | Description |
|:---|---:|:---|
| **Total Source Lines of Code (SLOC)** | **73,185** | Executable lines of code across all source languages (excluding blanks & comments) |
| **Total Physical Lines** | **96,236** | Complete line count in source files (includes 10,709 comments & 12,342 blank lines) |
| **Total Tracked Source Files** | **433** | Source code files tracked in git (Rust, Python, Java, Scala, Shell, SQL, GPU shaders, Web) |
| **Pure Production Code** | **48,197 SLOC** | Production database engine, connectors, APIs, and shaders (excluding all tests) |
| **Total Test Code** | **20,652 SLOC** | All test code across integration suites, connector tests, and inline unit tests |
| **Test-to-Production Ratio** | **1 : 2.33** | Test code comprises **~30.0%** of all executable code in the repository |
| **Configuration & Manifests** | **1,540 SLOC** | TOML and YAML configurations (Cargo manifests, CI/CD workflows, Docker Compose) |

---

## 2. Breakdown by Programming Language

| Language | Files | SLOC (Code) | Comments & Docs | Blank Lines | Total Lines | % of Code |
|:---|---:|---:|---:|---:|---:|---:|
| **Rust** (`.rs`) | 232 | **57,448** | 7,084 | 8,024 | 72,556 | 78.50% |
| **Python** (`.py`) | 118 | **11,295** | 2,954 | 3,337 | 17,586 | 15.43% |
| **Shell** (`.sh`) | 17 | **862** | 179 | 197 | 1,238 | 1.18% |
| **CSS** (`.css`) | 1 | **668** | 15 | 110 | 793 | 0.91% |
| **Java** (`.java`) | 11 | **615** | 56 | 162 | 833 | 0.84% |
| **SQL** (`.sql`) | 8 | **557** | 97 | 125 | 779 | 0.76% |
| **Scala** (`.scala`) | 18 | **541** | 64 | 152 | 757 | 0.74% |
| **HTML** (`.html`) | 1 | **350** | 0 | 15 | 365 | 0.48% |
| **Dockerfile** | 4 | **248** | 72 | 73 | 393 | 0.34% |
| **CUDA** (`.cu`) | 8 | **235** | 135 | 64 | 434 | 0.32% |
| **Metal** (`.metal`) | 7 | **155** | 34 | 41 | 230 | 0.21% |
| **OpenCL** (`.cl`) | 7 | **134** | 13 | 31 | 178 | 0.18% |
| **WGSL** (`.wgsl`) | 1 | **77** | 6 | 11 | 94 | 0.11% |
| **TOTAL (Source Code)** | **433** | **73,185** | **10,709** | **12,342** | **96,236** | **100.0%** |
| *YAML / TOML Manifests* | *29* | *1,540* | *166* | *169* | *1,875* | — |

---

## 3. Breakdown by Subsystem & Component

| Subsystem / Component | Files | SLOC (Code) | Comments | Blank | Total Lines | Primary Stack |
|:---|---:|---:|---:|---:|---:|:---|
| **Core Database Engine** (`src/`) | 169 | **45,356** | 5,748 | 6,193 | 57,297 | Rust, CUDA, Metal, OpenCL, WGSL |
| **Search Engine** (`hyperstreamdb-search/`) | 17 | **5,657** | 575 | 626 | 6,858 | Rust |
| **Arrow Flight Server** (`hyperstreamdb-flight/`) | 5 | **583** | 16 | 84 | 683 | Rust, Python |
| **Trino Connector** (`trino-hyperstream/`) | 11 | **615** | 56 | 162 | 833 | Java |
| **Spark Connector** (`spark-hyperstream/`) | 18 | **541** | 64 | 152 | 757 | Scala |
| **Python SDK & Client** (`python/`) | 2 | **506** | 251 | 142 | 899 | Python |
| **dbt Adapter** (`dbt-hyperstreamdb/`) | 13 | **444** | 29 | 105 | 578 | Python, SQL |
| **Enterprise Features** (`hyperstreamdb-enterprise/`) | 2 | **8** | 7 | 4 | 19 | Rust |
| **↳ Subtotal: Production Code** | **237** | **53,710** | **6,746** | **7,468** | **67,924** | |
| **Dedicated Test Suite** (`tests/`) | 156 | **13,940** | 2,885 | 3,586 | 20,411 | Python, Rust, Shell |
| **Benchmarks** (`benchmarks/`, `benches/`) | 8 | **1,830** | 272 | 410 | 2,512 | Python, Rust, Shell |
| **Build, CI, Docker & Admin Scripts** | 19 | **1,509** | 317 | 362 | 2,188 | Shell, Python, Dockerfile |
| **Examples & Demos** (`examples/`, `demo/`) | 10 | **1,127** | 452 | 378 | 1,957 | Python, Rust, SQL |
| **Marketing Site & Docs Scripts** | 3 | **1,069** | 37 | 138 | 1,244 | HTML, CSS, Python |

---

## 4. Core Engine Architecture Breakdown (`src/`)

Within the core database engine (`src/`), the 45,356 lines of code are distributed as follows:

| Engine Subsystem | Files | SLOC (Code) | Total Lines | Description & Key Responsibilities |
|:---|---:|---:|---:|:---|
| **Vector & Inverted Indexing** (`src/core/index/`) | 47 | **9,727** | 12,136 | HNSW graph, IVF-PQ quantization, SIMD/GPU acceleration, Bloom filters |
| **SQL & DataFusion Engine** (`src/core/sql/`) | 32 | **7,500** | 9,380 | pgvector operators, custom dialects, physical expression pushdown |
| **Table Engine & Concurrency** (`src/core/table/`) | 10 | **6,662** | 8,387 | Table locks, segment lifecycles, streaming ingestion buffers |
| **Iceberg Integration** (`src/core/iceberg/`) | 9 | **2,742** | 3,111 | REST catalog client, v2/v3 spec, manifest parsing & commit protocol |
| **Parquet Readers** (`src/core/reader/`) | 4 | **2,721** | 3,240 | Vectorized Parquet reader, column projection, filter pushdown |
| **PyO3 Python Engine Bindings** (`src/python/`) | 15 | **2,446** | 2,958 | Python C-API bindings for Arrow tables, sessions, and catalog |
| **Manifests & Metadata** (`src/core/manifest/`, `metadata.rs`) | 8 | **2,278** | 2,726 | Avro metadata serialization, partition summary records |
| **Execution Planner & Query Engine** (`planner.rs`, `query.rs`) | 3 | **2,358** | 3,167 | Logical/physical plan optimization, hybrid vector/scalar scoring |
| **Catalog Implementations** (`src/core/catalog/`, `nessie.rs`) | 9 | **1,789** | 2,185 | Hive metastore, Nessie, Memory, and File system catalogs |
| **CLI Binaries & Tools** (`src/bin/`) | 6 | **1,401** | 1,658 | REST gateway, admin CLI, layered index verification tools |
| **WAL, Compaction, Cache & Storage** | 6 | **2,044** | 2,746 | Write-Ahead Log, segment compaction, block cache, cloud storage |
| **Telemetry & Error Handling** | 5 | **426** | 648 | OpenTelemetry metrics, distributed tracing, error types |

---

## 5. Comprehensive Test Suite Breakdown

Across all testing layers, HyperStreamDB contains **20,652 SLOC of test code**.

### A. Test Code Distribution

| Test Layer | Files | SLOC (Code) | Primary Languages | Focus |
|:---|---:|---:|:---|:---|
| **Dedicated Test Suite** (`tests/`) | 155 | **13,934** | Python (6.9k), Rust (6.5k), Shell (566) | E2E, integration, property & chaos tests |
| **Inline Unit Tests** (`#[cfg(test)]`) | 40 | **5,513** | Rust native | Embedded unit tests in `src/` and sub-crates |
| **Sub-crate & Connector Test Suites** | 11 | **1,205** | Python, Java, Scala, Rust, SQL | Trino, Spark, Search, Flight, and dbt tests |
| **TOTAL TEST CODE** | **206** | **20,652** | | |

---

### B. Dedicated Test Suite Breakdown by Domain (`tests/`)

The 155 test files inside the dedicated `tests/` directory target specific functionality:

#### 1. Vector Search & Hardware Acceleration (34 files | 4,158 SLOC)
- **Multi-Hardware Parity**: Cross-platform verification ensuring numerical and recall parity across backends:
  - CUDA: `test_hyperstream_cuda.py`
  - Apple Silicon Metal/MPS: `test_mps_gpu.py`
  - Intel: `test_hyperstream_intel.py`
  - AMD ROCm: `test_hyperstream_rocm.py`
  - CPU SIMD: `test_hardware_parity.rs`
- **Distance Metric Kernels**: Correctness benchmarks and edge cases for L1, L2, Inner Product, Cosine, Hamming, and Jaccard distances.
- **Index Algorithms**: HNSW recall verification, layered index recovery, TurboQuant quantization, and GPU KMeans clustering.

#### 2. Data Ingestion, Schema Evolution & Partitioning (21 files | 2,105 SLOC)
- Partition transform validations (identity, bucket, truncate, hour/day transforms).
- Schema evolution compatibility across Iceberg metadata snapshots.
- Primary key acceleration, composite indexes, and write-buffer deduplication.

#### 3. Iceberg Format, REST Catalogs & Metadata (25 files | 1,876 SLOC)
- Iceberg v2/v3 specification compliance.
- Merge-on-Read (MOR) read and write workflows with equality/positional deletes.
- Puffin format index blob creation and attachment.
- REST catalog, AWS Glue, and Hive Metastore transaction commit verification.

#### 4. Performance Verification & Comparative Benchmarks (7 files | 981 SLOC)
- Direct comparison suites vs. Qdrant (`test_vs_qdrant.py`, `test_qdrant_direct_comparison.py`).
- Scan latency baselines, cold-read latency benchmarks, and parallel search scaling.

#### 5. Concurrency, WAL, Durability & Chaos (10 files | 837 SLOC)
- Multi-threaded writer contention and optimistic concurrency control.
- Chaos testing: simulated power losses during WAL flush, torn writes, crash recovery, and stale file cleanup.

#### 6. SQL Engine & DataFusion Pushdown (7 files | 492 SLOC)
- pgvector SQL operator semantics (`<->`, `<#>`, `<=>`).
- SQL UDF GPU acceleration and BM25 full-text pushdown.
- Complex aggregations, filtering, and ordering.

#### 7. Python PyO3 SDK & Integrations (15 files | 1,300 SLOC)
- Fluent client API verification, PyArrow schema consistency, docstrings, and export routines.
- Full real-world ingestion flows using NYC Taxi and Wikipedia datasets.

---

### C. Inline Unit Tests Breakdown (`#[cfg(test)]` in `src/`)

Unit tests written directly alongside production Rust modules total **5,513 SLOC** across 40 files:

| Module / Component | Test Files | Unit Test SLOC | Production SLOC | Major Features Tested |
|:---|---:|---:|---:|:---|
| **SQL Engine** (`src/core/sql/`) | 9 | **1,840** | 1,827 | Literal parsers (1,269 lines), vector operators (257 lines), function evaluation |
| **Search Engine** (`hyperstreamdb-search/`) | 7 | **1,081** | 2,307 | Query inference (387 lines), search handlers (312 lines), index caching, state |
| **Vector Index** (`src/core/index/`) | 11 | **903** | 4,655 | HNSW IO serialization (220 lines), distance calculations, quantization |
| **Execution Planner & Query Engine** | 2 | **305** | 1,546 | Plan generation and scoring in `query.rs` (302 lines) |
| **Compaction & Table Merge** | 2 | **436** | 603 | Segment compaction (205 lines), merge conflict resolution (231 lines) |
| **WAL, Storage & Cache** | 3 | **474** | 634 | WAL replay (174 lines), block cache eviction (153 lines), storage backends (147 lines) |
| **Table, Reader & Segments** | 3 | **304** | 3,540 | Segment state machines, Parquet reader filtering, table lock semantics |
| **Catalog & Locking** | 3 | **170** | 607 | In-memory catalog commit logic, concurrency primitives |
| **TOTAL INLINE UNIT TESTS** | **40** | **5,513** | — | |

---

### D. Sub-crate & Connector Dedicated Tests

| Project / Sub-crate | File | SLOC | Total Lines | Purpose |
|:---|:---|---:|---:|:---|
| `hyperstreamdb-search` | `tests/test_search_api.py` | **589** | 775 | Python HTTP client testing hybrid search endpoints |
| `trino-hyperstream` | `src/test/.../TrinoConnectorTest.java` | **225** | 306 | Trino SPI split generation and predicate pushdown |
| `hyperstreamdb-flight` | `tests/test_flight_service.rs` | **113** | 134 | Native Rust Arrow Flight DoGet / DoPut integration |
| `hyperstreamdb-flight` | `tests/test_flight_client.py` | **26** | 40 | Python pyarrow.flight client verification |
| `scripts/manual_tests` | `test_multi_cloud.py` | **132** | 212 | S3, GCS, and Azure multi-cloud integration tests |
| `spark-hyperstream` | `src/test/.../SparkMergeIntegrationTest.scala` | **76** | 101 | Spark DataFrame write/merge integration |
| `dbt-hyperstreamdb` | `test_project/models/*.sql` | **38** | 43 | dbt incremental, macro, and vector models |
| Standalone | `test_bloom.rs` | **6** | 6 | Bloom filter false-positive verification |
| **TOTAL** | **11 files** | **1,205** | **1,617** | |

---

## 6. Hardware Acceleration Shaders & Kernels

The core engine includes dedicated GPU and compute shaders across 4 vendor targets:

| Target Platform | File Extension | Files | SLOC (Code) | Total Lines | Implementations |
|:---|:---|---:|---:|---:|:---|
| **NVIDIA CUDA** | `.cu` | 8 | **235** | 434 | L2, Inner Product, Cosine, Hamming distance kernels |
| **Apple Metal** | `.metal` | 7 | **155** | 230 | MSL shaders for Apple Silicon MPS execution |
| **Khronos OpenCL** | `.cl` | 7 | **134** | 178 | Cross-vendor portable GPU compute kernels |
| **WebGPU / WGPU** | `.wgsl` | 1 | **77** | 94 | Cross-platform WGPU compute shader |
| **TOTAL GPU KERNELS** | | **23** | **601** | **936** | |

---

## 7. Connectors & External Integrations

| Ecosystem Connector | Primary Language | Source Files | SLOC (Code) | Test SLOC | Target System |
|:---|:---|---:|---:|---:|:---|
| **Trino Connector** | Java | 11 | **615** | 225 | Trino / Presto distributed SQL engine |
| **Spark Connector** | Scala | 18 | **541** | 76 | Apache Spark 3.x DataSourceV2 connector |
| **Python Client / SDK** | Python | 2 | **506** | — | High-level user-facing Python SDK |
| **dbt Adapter** | Python + SQL | 13 | **444** | 38 | dbt plugin for analytics engineering |
| **Arrow Flight** | Rust + Python | 5 | **583** | 139 | High-performance Arrow Flight RPC service |
| **TOTAL CONNECTORS** | | **49** | **2,689** | **478** | |

---

## 8. Methodology & Counting Standards

- **SLOC (Source Lines of Code)**: Measures actual executable code statements. Blank lines and comments (both single-line and multi-line blocks) are excluded.
- **Comments & Documentation**: Includes docstrings (`"""`), Rust doc comments (`///`, `//!`), and standard comments (`//`, `#`, `/* ... */`, `--`).
- **Inline Tests**: Rust unit test blocks demarcated by `#[cfg(test)] mod tests { ... }` within production source files were parsed, isolated, and categorized under test metrics rather than production metrics to present an accurate architectural distribution.
- **Excluded Non-Code Artifacts**: Build caches (`docs/build/`), test runner state (`.hypothesis/`), and Jupyter checkpoints (`.ipynb_checkpoints/`) were excluded from source metrics.
