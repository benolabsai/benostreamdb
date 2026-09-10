# Progress

Milestone log for HyperStreamDB. Newest entries first.

---

## 2026-09-10 — v0.7.0 Release & Production Hardening

HyperStreamDB v0.7.0 brings multi-vector search with RRF scoring coordination, composite scalar roaring bitmap indexes, Apache Polaris / Lakekeeper REST catalog OAuth2 authentication, and community TurboQuant™ scalar quantization. Additionally, following in-depth architectural and code review, significant correctness and production-hardening passes were executed across the engine.

### Shipped
- **Multi-Vector Search & Reciprocal Rank Fusion (RRF)**:
  - Concurrent multi-vector search across distinct vector columns with RRF score combination ($1 / (k + \text{rank} + 1)$).
  - DataFusion SQL optimizer and physical plan rewriter pushdown for multiple vector distance expressions (`VectorScanExec`).
  - Integration test suite: `tests/test_multi_vector_search.rs`.
- **Composite Scalar Roaring Bitmap Indexes**:
  - `IndexAlgorithm::CompositeBitmap` with exact `"identity"` tokenization to support multi-column point and range queries.
  - Multi-column index file generation and query filter rewriting in reader.
  - Integration test suite: `tests/test_composite_index.rs`.
- **Apache Polaris & Lakekeeper OAuth2 Client Credentials**:
  - Full client credentials grant flow (`/v1/oauth/tokens`) conforming to Iceberg REST Catalog spec.
  - Automatic bearer token caching and background refresh within 60s of expiration.
  - REST catalog unit test coverage: `src/core/catalog/rest.rs`.
- **Core Community TurboQuant™ (TQ4 & TQ8)**:
  - Scalar quantization integrated into HNSW indexing pipeline.
- **Production Hardening & Review Adaptations**:
  - **Dynamic Vector Metric Propagation**: Extracted metric from `IndexAlgorithm` into `HnswIvfIndex::build` (supporting `L2`, `Cosine`, `InnerProduct`, `L1`, `Hamming`, `Jaccard`), implemented `FromStr` on `VectorMetric`, and updated Puffin/Parquet index metadata deserialization.
  - **Global KNN Ordering**: Refactored `merge_and_rerank_vector_results` to guarantee monotonic ascending distance order across all returned batches using contiguous chunking rather than unordered `HashMap` bucketing.
  - **Metric Parity Test Suite**: Added `tests/test_vector_metrics_parity.rs` establishing 100% nearest-neighbor accuracy against exact brute-force ground truth across all 6 metrics.
  - **Strict Query Failure Semantics**: Segment vector search failures now fail the query immediately with actionable diagnostics rather than silently omitting rows.
  - **Zero-Warning Standard**: Cleaned up diagnostic `println!` statements in favor of structured `tracing::debug!`, verified 0 warnings under `#![deny(warnings)]` and clean `cargo fmt`.
  - **Iceberg Architecture Positioning**: Refined README and compliance tool to accurately position HyperStreamDB as an indexed lakehouse storage engine with an advisory/reconstructible index overlay.

---

## 2026-09-07 — OpenSearch / Elasticsearch 7.10-compatible Search API (M0–M3 complete)

The `hypersearch` add-on (`hyperstreamdb-search`) now implements the full v1
Elasticsearch/OpenSearch 7.10-compatible REST surface, plus a Qdrant-compatible API,
on top of the HyperStreamDB engine.

### Shipped
- **Workspace** — `hyperstreamdb-search` added as a Cargo workspace member (single
  lockfile, no version drift).
- **Cluster & metadata** — `GET /`, `GET /_health`, `GET /_cluster/health`,
  `GET /_cluster/stats`, `GET /_cat/indices`, `GET /metrics` (Prometheus).
- **Index CRUD** — `PUT /{index}` (with optional mapping), `GET /{index}`,
  `DELETE /{index}` (hard delete of all store objects).
- **Mapping** — `GET /{index}/_mapping` (Arrow → ES properties), `PUT /{index}/_mapping`
  (adds columns via `Table::add_column`; optional index registration).
- **Documents** — `POST /{index}/_doc[/{id}]` (schema-on-write auto-creation, `_id`
  primary key, duplicate → 400), `DELETE /{index}/_doc/{id}` → 501 (append-only).
- **Bulk** — `POST /_bulk` and `POST /{index}/_bulk` (NDJSON `index`/`create`/`delete`,
  batched per index, per-item status; `delete` → per-item 501).
- **Search** — `POST /{index}/_search` and `GET /{index}/_search?q=`:
  - `match` (BM25 Okapi over inverted indexes; multi-field OR-merge)
  - `knn` (HNSW; `k`, `num_candidates` → `ef_search`, `filter`)
  - hybrid `match` + `knn` (RRF fusion; `rrf_k` per-request or `HYPERSEARCH_RRF_K`)
  - `match_all`, `filter`/`bool` (`term`, `terms`, `range`, `exists`, `must_not`)
  - `_source` includes/excludes, `from`/`size`
- **Count & refresh** — `POST /{index}/_count` (filtered via planner / unfiltered via
  manifest), `POST /{index}/_refresh` and `POST /_refresh` (global).
- **Core BM25** — English analyzer (lowercase + punctuation tokenizer + stop-words),
  Okapi BM25 scoring with a per-segment doc-length sidecar, integrated into the
  keyword search path and hybrid RRF coordinator.
- **Qdrant API** — 8-endpoint Qdrant-compatible surface (collections + points) on a
  secondary listener (default port 6333), sharing the same `AppState`.
- **Telemetry** — `hypersearch_query_seconds{op}`, `hypersearch_bulk_items_total{status}`,
  `hypersearch_refresh_seconds`, request counters, per-route latency, cache hit/miss,
  and in-flight gauges.
- **Env config** — `HYPERSEARCH_STORAGE_URI`, `HYPERSEARCH_BIND`/`PORT`,
  `HYPERSEARCH_AUTO_REFRESH_SECS` (periodic flush), `HYPERSEARCH_RRF_K`,
  `HYPERSEARCH_INDEX_CACHE_GB`, `QDRANT_BIND`/`PORT`.
- **On-demand index-file cache** — [`hyperstreamdb-search/src/index_cache.rs`](hyperstreamdb-search/src/index_cache.rs):
  an LRU, size-capped (default 2 GiB) cache of index files keyed by
  `(index, segment_id, column, file, manifest_version)`, with `fetch_index_file`
  fetch-through and `hypersearch_index_fetch_bytes_total{kind}`. 9 unit tests
  cover eviction, LRU ordering, and invalidation (by index + by manifest version).
- **Benchmark (quick)** — `benchmark_es710.py --quick` validates the ES 7.10.2
  comparison pipeline end-to-end. Hypersearch search latency (86–120 ms p95) is
  within the 50–200 ms target; ES is faster on search (3–6 ms, in-memory) by
  design. See [BENCHMARK_FINDINGS.md](BENCHMARK_FINDINGS.md).

### Verification
- `cargo test -p hyperstreamdb-search --lib` — 45 unit tests pass.
- `pytest hyperstreamdb-search/tests/test_search_api.py` — 26 integration tests pass
  (cluster, health, metrics, doc writes, dups, schema evolution, match/knn/hybrid,
  filters, pagination, index CRUD, mapping, bulk, count, cat/stats, delete 501,
  global refresh, terms/must_not, `_source`, `q=`).
- `cargo clippy -p hyperstreamdb-search --all-targets` — clean (zero warnings).
- `cargo fmt -p hyperstreamdb-search --check` — clean.

### Documentation
- [GETTING_STARTED.md](GETTING_STARTED.md) — server + Python client quickstart.
- [OPENSEARCH_COMPATIBILITY.md](OPENSEARCH_COMPATIBILITY.md) — supported / unsupported
  matrix, error envelope, dashboards/observability, positioning.
- [README.md](README.md) — new "Search API" section.

### Remaining (follow-ups)
- 100k / 1M doc benchmark vs local ES 7.10 (local FS + MinIO S3) — long-running;
  commands and quick-run findings are in [BENCHMARK_FINDINGS.md](BENCHMARK_FINDINGS.md).
- `cargo audit` gate (CI; tool not installed locally).
- Version bump + tag (after the full 100k/1M benchmark).
