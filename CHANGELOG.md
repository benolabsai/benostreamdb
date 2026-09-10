# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/rla3rd/hyperstreamdb/compare/v0.5.3...HEAD
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
