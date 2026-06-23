# Implementation Plan: Elasticsearch-Like Search Engine Layer (Add-On Product)

This document outlines the strategy for validating HyperStreamDB's production readiness and building an Elasticsearch-compatible REST API wrapper. 

The REST API is implemented as a **separate, optional add-on package (`hyperstreamdb-search`)** in the Cargo workspace. This ensures the core library remains lightweight, and users who only need library-level access or FFI bindings do not pull in HTTP dependencies.

---

## Architectural Thesis: Cost-Efficiency & Serverless Over Low-Latency

* **Not In-Memory**: Elasticsearch keeps all segments and caches aggressively pinned in RAM to achieve sub-millisecond query execution. This makes it operationally expensive and hard to run in serverless environments.
* **Target Workload (Website / Doc Search)**: For website search, document indexing, and corporate knowledge bases, query response latencies of **50–200ms** are perfectly acceptable.
* **Object Storage Native**: By designing for website search, we can load index files (`.hnsw` and `.idx`) and Parquet row groups **on-demand** from object storage (like S3/MinIO) or local disk cache. This enables an extremely cheap, serverless, scale-to-zero operational model.

---

## Step 1: Production Readiness & Quality Validation

Before implementing new features, we must establish a production-readiness baseline. This ensures the engine's core is secure, stable, and conforms to standard best practices.

### 1.1 Code Quality & Safety Checks
- **Security Audit**: Ensure `cargo audit` runs successfully in CI/CD with zero unignored vulnerabilities.
- **Static Analysis**: Enforce Clippy linting (`cargo clippy --all-targets --all-features`) with zero warnings or errors.
- **Formatting Standards**: Verify that all Rust files are styled per `cargo fmt`.
- **Code Test Coverage**: Ensure all Rust core unit and integration tests compile and run to completion (`cargo test --all-targets`).
- **Python Compatibility**: Run the full Python test suite (`pytest`) to ensure zero regressions in bindings, catalogs, and table reads/writes.

### 1.2 Documentation Readiness Check
- **API Reference Compilation**: Verify that `cargo doc --no-deps` builds cleanly without warnings or broken intra-doc links.
- **Getting Started Guide**: Ensure [GETTING_STARTED.md](GETTING_STARTED.md) is up-to-date with current APIs and dependency instructions.
- **API Coverage**: Confirm all public catalog methods, index structures, and configuration keys are fully documented.

---

## Step 2: Lexical Processing & Search Enhancements (Crate: `hyperstreamdb`)

To support Elasticsearch-like search capability, we implement base lexical processing helpers within the core library.

### 2.1 Text Analyzers & Tokenizers
- **New Module**: `src/core/index/analyzer.rs`
- **Features**:
  - Lowercase filter.
  - Standard whitespace/punctuation tokenizer.
  - Basic english stop-word filter.

### 2.2 BM25 Relevance Scoring
- **Implementation**: Calculate Okapi BM25 scores during inverted index lookup:
  $$\text{Score}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
- **Integration**: Integrate the scoring mechanism with the search planner to rank results.

---

## Step 3: The `hyperstreamdb-search` Add-On Crate

Create a new package `hyperstreamdb-search` in the Cargo workspace root.

### 3.1 Cargo Setup
- **New Crate**: [hyperstreamdb-search/Cargo.toml]
- **Dependencies**: `hyperstreamdb`, `axum`, `tower`, `tower-http`, `tokio`, `serde`, `serde_json`.

### 3.2 Document Ingestion & Schema Evolution
- **Endpoint**: `POST /<index_name>/_doc`
- **Dynamic Schema-Mapping**: Parse arbitrary JSON documents, infer field datatypes, and dynamically evolve the target Iceberg schema (without rewriting existing Parquet data).
- **Buffered Commits**: Route incoming documents to the active Memtable and Write-Ahead Log (WAL) to ensure low write latency.

### 3.3 Search API
- **Endpoint**: `POST /<index_name>/_search`
- **Query Parser**: Parse JSON-based query DSL matching standard search criteria:
  - `match` (lexical search using inverted index + BM25).
  - `knn` (HNSW vector search).
  - `filter` (roaring bitmap scalar pre-filtering).
- **On-Demand Loading**: Fetch the HNSW and inverted indexes from S3/MinIO on-demand, caching them locally in temp storage for future requests.

---

## Step 4: Verification & Benchmarking

### 4.1 Automated Validation
- Implement a test suite in `hyperstreamdb-search/tests/test_search_api.py` that verifies:
  - Document ingestion via HTTP `POST`.
  - Dynamic table creation and schema evolution.
  - Hybrid lexical + vector search queries.
  
### 4.2 Benchmark Analysis
- Measure ingestion throughput (docs/sec) and search latency (p95/p99) against a baseline local Elasticsearch instance, targeting the 50–200ms latency envelope.

---

## Step 5: Positioning: Strengths & Weaknesses vs. Elasticsearch

To effectively market this add-on, we must clearly define how it compares to Elasticsearch (ES) so users understand when to choose HyperStreamDB-Search.

### 5.1 Strengths (Competitive Advantages)
1. **Ultra-Low TCO (Total Cost of Ownership)**:
   * *Elasticsearch*: Requires expensive instance groups with huge memory allocations (JVM heaps) and fast, hot SSD storage. 
   * *HyperStreamDB*: Built for serverless object storage (S3/MinIO). When idle, it costs nothing. Index files (`.hnsw` and `.idx`) are fetched on-demand and cached locally.
2. **Open Data Lakehouse Integration**:
   * *Elasticsearch*: Uses a proprietary data format. Getting data out for analysis requires heavy ETLs or expensive scroll APIs.
   * *HyperStreamDB*: Underpinned by Apache Iceberg and Parquet. Other tools (DuckDB, Trino, Spark) can query the exact same files directly in the data lake without moving data.
3. **Dynamic Schema Evolution**:
   * *Elasticsearch*: Changing mapping types or structures often requires creating a new index and running a resource-heavy `_reindex` job.
   * *HyperStreamDB*: Inherits Iceberg's native schema evolution, allowing column additions, drops, and renaming instantly.
4. **GPU-Accelerated Hybrid Search**:
   * *Elasticsearch*: Vector search runs on standard JVM threads.
   * *HyperStreamDB*: Native vector indexing with SIMD/GPU acceleration for low-cost, high-scale HNSW execution.

### 5.2 Weaknesses (Where ES Wins & How to Position It)
1. **Sub-Millisecond Query Latency**:
   * *Elasticsearch*: Sub-5ms response times due to aggressive in-memory caching.
   * *HyperStreamDB*: 50–200ms response times due to S3-native fetch overhead.
   * *Positioning*: Position HyperStreamDB-Search for **website search, document catalogs, and log archives** where 50–200ms is imperceptible to users, but the 90% hosting cost reduction is highly compelling.
2. **Ecosystem & Dashboarding**:
   * *Elasticsearch*: Has mature Kibana integration for visualization and dashboarding.
   * *HyperStreamDB*: No proprietary visualization frontend.
   * *Leveraging Grafana, Prometheus & Apache Superset*:
     - **Prometheus Metrics Endpoint**: The `hyperstreamdb-search` REST server will expose a `/metrics` endpoint (via the `prometheus` crate, already a dependency) providing operational telemetry: query latency histograms, ingestion throughput counters, index cache hit rates, and active connection gauges.
     - **Grafana + Prometheus Stack**: Grafana natively scrapes Prometheus endpoints, giving operators real-time operational dashboards (query p95/p99, error rates, ingestion backpressure) out of the box with zero custom code.
     - **Grafana via DuckDB / Trino (Data Dashboards)**: For data-level dashboards, Grafana connects to DuckDB and Trino. Since our tables are standard Iceberg, Grafana can query and visualize HyperStreamDB data with full SQL support.
     - **Apache Superset**: The premier open-source BI tool for data lakes natively supports Trino, Spark, and DuckDB, offering a robust Kibana-like log viewing and dashboarding experience.
     - **Elasticsearch API Compatibility**: By ensuring our REST API matches standard Elasticsearch query endpoints, users can configure **Grafana's built-in Elasticsearch datasource** to query HyperStreamDB directly, providing zero-friction dashboard reuse.

