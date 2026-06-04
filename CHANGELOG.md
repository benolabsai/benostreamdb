# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/rla3rd/hyperstreamdb/compare/v0.5.0...HEAD
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
