# OpenSearch / Elasticsearch Compatibility

`bsdb-search` (the `benostreamdb-search` add-on) speaks the **OpenSearch 1.x /
Elasticsearch 7.10** wire format. OpenSearch 1.x is the 7.10 fork, so the response
shapes, error envelopes, and query DSL target that dialect. `GET /` reports
`version.number = "7.10.2"` and `tagline = "You know, you search"`.

This document lists exactly what is supported, what is not, and how to work around
the gaps.

---

## Supported

### Cluster & metadata
| Endpoint | Notes |
|----------|-------|
| `GET /` | Cluster info; ES 7.10 version block + tagline. |
| `GET /_health`, `GET /_cluster/health` | Single-node health (`status: green`, 1 node, 1 primary shard per index). |
| `GET /_cluster/stats` | Aggregate index/node statistics. |
| `GET /_cat/indices` | Tab-separated index summary (`health status index pri rep docs.count docs.store store.size`). |
| `GET /_cat/cluster_manager`, `GET /_cat/master` | Active cluster manager / master node info (OpenSearch 3.x and ES compatible). |
| `GET /_cat/nodes` | Tab-separated node status, resource, and role summary. |
| `GET /_cat/shards` | Tab-separated shard allocation per index. |
| `GET /_cat/health` | Tab-separated cluster health and active shards. |
| `GET /_nodes*` | Node capabilities and HTTP coordinates for client connection sniffing. |
| `GET /_cluster/settings`, `PUT /_cluster/settings` | Cluster settings inspection and no-op acknowledgement. |
| `GET /_xpack`, `GET /_license` | Compatibility shims for official Elastic clients probing security/license status. |
| `GET /_ingest/pipeline*`, `PUT /_ingest/pipeline*` | Pipeline definition stubs acknowledging ingest pipeline configuration. |
| `GET /metrics` | Prometheus text format (0.0.4) operational telemetry. |

### Index management
| Endpoint | Notes |
|----------|-------|
| `PUT /{index}` | Create an index, optionally with `mappings.properties`. Pre-existing index → 400 `resource_already_exists_exception`. |
| `GET /{index}` | Returns `aliases`, `mappings`, and `settings`. Missing index → 404 `index_not_found_exception`. |
| `DELETE /{index}` | **Hard delete** — removes all store objects (manifest, metadata, data, indexes). |
| `GET /{index}/_mapping` | Renders the Arrow schema as ES properties (`text`, `long`, `double`, `boolean`, `date`, `dense_vector{dims}`, nested `object`, and `nested` array of objects). |
| `PUT /{index}/_mapping` | Adds new properties via `Table::add_column`; optional `indexes` block registers index algorithms. |
| `GET /{index}/_field_caps`, `GET /_field_caps` | Field capabilities API returning types, searchability, aggregatability, metadata status, and conflict mappings across indices. |

### Document lifecycle & updates
| Endpoint | Notes |
|----------|-------|
| `POST /{index}/_doc` | Index a doc with a server-generated id. Auto-creates the index on first write (schema-on-write). 201 `created` / 200 `updated`. |
| `POST /{index}/_doc/{id}` | Index a doc with a client-supplied id. Duplicate id → 400 `resource_already_exists_exception`. |
| `GET /{index}/_doc/{id}` | Retrieve document by `_id`. Returns standard ES metadata and `_source`. |
| `DELETE /{index}/_doc/{id}` | Soft/hard delete of document by `_id`. Returns standard ES `deleted` / `not_found`. |
| `POST /{index}/_delete_by_query` | Deletes matching documents across segments matching the query DSL. |
| `POST /{index}/_update/{id}` | Partial document update (`doc`), merge, and `doc_as_upsert`. |
| `POST /_reindex` | Copies documents from source to destination index, supporting query filtering, field selection, `max_docs`, and version conflict management. |
| `POST /_bulk`, `POST /{index}/_bulk` | NDJSON `index`, `create`, `update`, `delete` actions. Batched per index; per-item status in the response. |
| `POST /{index}/_refresh`, `POST /_refresh` | Flush the write buffer (memtable + WAL → segments + indexes) so new docs become searchable. |

### Index aliases
| Endpoint | Notes |
|----------|-------|
| `POST /_aliases` | Add/remove index aliases atomically. |
| `GET /_alias`, `GET /{index}/_alias` | Retrieve registered aliases across cluster or per index. |
| `PUT /{index}/_alias/{name}` | Create/assign alias to index. |
| `DELETE /{index}/_alias/{name}` | Remove alias from index. |

### Aggregations
`POST /{index}/_search` accepts an `aggs` (or `aggregations`) object, compiled
to SQL and executed with DataFusion. Aggregations run over the top-level
`filter` (the query clause scopes aggregations via filter handoff).

| Aggregation | Notes |
|-------------|-------|
| `terms` | `{field, size}` → `{buckets: [{key, doc_count}]}`, ordered by `doc_count` desc. |
| `histogram` | `{field, interval}` → numeric buckets. |
| `date_histogram` | `{field, calendar_interval\|fixed_interval}` → `date_trunc` buckets. |
| `range` | `{field, ranges: [{from, to, key?}]}` → range buckets. |
| `filter` | `{...filter clause...}` → `{doc_count}`. |
| `missing` | `{field}` → `{doc_count}` of null rows. |
| `avg`, `sum`, `min`, `max`, `value_count`, `cardinality` | `{field}` → `{value}`. |
| `stats`, `extended_stats` | `{field}` → `{count, min, max, avg, sum, sum_of_squares, variance, std_deviation}`. |
| `percentiles` | `{field, percents?, keyed?}` → `{values}` computed via DataFusion `approx_percentile_cont`. |
| `composite` | Multi-source bucket pagination (`terms`, `histogram`, `date_histogram`), `order`, `after_key` pagination cursor. |
| `significant_terms` | Foreground vs. background term frequency analysis with JLH significance scoring. |
| Nested `aggs` | Supported on `terms`, `histogram`, `range`, `filter`, `missing`, `composite`, and `significant_terms` buckets. |

### Search
| Feature | Notes |
|---------|-------|
| `POST /{index}/_search` | Full query DSL (below). |
| `GET /{index}/_search?q=` | Lucene-style query string parsed with AST tokenizer; honours `size` / `from`. |
| `POST /{index}/_count` | Document count, optionally filtered. |
| `match` | BM25 (Okapi) lexical search over inverted indexes. Multi-field `match` OR-merges per-field results. |
| `match_phrase` | Exact phrase matching with configurable positional `slop`. |
| `multi_match` | Multi-field full-text search across per-field BM25 inverted indexes. |
| `query_string`, `simple_query_string` | Lucene AST parser supporting boolean operators (`AND`, `OR`, `NOT`, `&&`, `\|\|`, `!`), field qualifiers (`field:val`), phrases, ranges (`[A TO B]`), wildcards (`*`), and grouping `()`. |
| `fuzzy` | Levenshtein edit-distance matching with `fuzziness` (`AUTO` or integer distance) and `prefix_length` pruning. |
| `knn` | HNSW vector search. `k`, `num_candidates` (→ `ef_search`), `filter`. |
| `hybrid` (`match` + `knn`) | Fused with Reciprocal Rank Fusion (RRF). `rrf_k` overridable per-request or via `BSDB_SEARCH_RRF_K`. |
| `match_all` | Returns all docs (uniform score 1.0). |
| `filter` / `bool` | `term`, `terms`, `range`, `exists`, `prefix`, `wildcard`, `regexp`, `ids`, `fuzzy`, `nested`, and `bool { must, filter, should, must_not }`. |
| `sort`, `search_after` | Multi-field sorting (`asc`/`desc`), `_score`, `_id`, and cursor-based deep pagination (`search_after`). |
| `scroll` | Keyset-backed stateless pagination via `_search?scroll=1m`, `POST /_search/scroll`, `GET /_search/scroll`, and `DELETE /_search/scroll` for export tools and migration scripts. |
| `highlight` | Snippet term highlighting (`<em>...</em>`) for matching full-text clauses. |
| `_source` filtering | `_source: { includes, excludes }` (dot-prefixed includes keep nested fields). |
| `from` / `size` | Pagination (default `size` 10, `from` 0). |
| `has_child` / `has_parent` | Relational and graph-neighborhood query DSL support (`type`/`parent_type`, `query`, `min_children`, `max_children`, and optional CSR `edge_index` traversal). |
| `nested` | Nested query DSL for both single objects (`user.age`) and arrays of objects with independent per-element predicate matching. |
| `_id` | Explicit document id (reserved `_id` primary-key column). Synthesized `{segment_id}:{row_id}` fallback for pre-existing tables without an id column. |

### Snapshot & restore
| Endpoint | Notes |
|----------|-------|
| `GET /_snapshot`, `GET /_snapshot/_all` | List registered snapshot repositories. |
| `GET /_snapshot/{repo}` | Retrieve snapshot repository configuration. |
| `PUT /_snapshot/{repo}`, `POST /_snapshot/{repo}` | Register or update snapshot repository. |
| `DELETE /_snapshot/{repo}` | Unregister snapshot repository and clear cached metadata. |
| `POST /_snapshot/{repo}/_verify` | Verify snapshot repository connectivity. |
| `PUT /_snapshot/{repo}/{snap}`, `POST /_snapshot/{repo}/{snap}` | Create snapshot, checkpointing target Iceberg tables and committing Parquet files. Supports `wait_for_completion`. |
| `GET /_snapshot/{repo}/{snap}`, `GET /_snapshot/{repo}/_all` | Retrieve snapshot metadata across cluster or per repository. |
| `GET /_snapshot/{repo}/{snap}/_status` | Retrieve snapshot completion status and shard progress. |
| `POST /_snapshot/{repo}/{snap}/_restore` | Restore snapshot checkpoint. |
| `DELETE /_snapshot/{repo}/{snap}` | Delete snapshot record. |

### Error envelope & client handshake
Errors use the ES shape:
```text
{ "error": { "type": "<exception>", "reason": "..." }, "status": <http_code> }
```
Mapped types include `index_not_found_exception` (404), `resource_already_exists_exception`
(400), `search_context_missing_exception` (404), `repository_missing_exception` (404),
`snapshot_missing_exception` (404), and `illegal_argument_exception` (400).

#### Client product validation headers
All HTTP responses include compatibility headers so official client SDKs (Python, Node.js, Go, Java) pass initialization and validation checks without error:
- `x-elastic-product: Elasticsearch`
- `x-opensearch-version: 3.0.0`

#### Graceful fallback for unmapped endpoints
Any unmapped route or method triggers a global Axum fallback returning HTTP 400 with a standard ES error envelope:
```json
{
  "error": {
    "type": "illegal_argument_exception",
    "reason": "No handler found for uri [/unimplemented/route] and method [POST]"
  },
  "status": 400
}
```
This ensures upstream clients receive valid, parseable JSON error envelopes rather than unformatted 404/501 or text/HTML responses, preventing client JSON decoder crashes.

---

## Not supported (v1)

| Feature | Behavior / workaround |
|---------|----------------------|
| ILM (index lifecycle) | Not implemented. |
| Authentication / security | Not implemented. Bind to `127.0.0.1` (default) or put a reverse proxy with auth + TLS in front. |
| Multi-node / sharding / replicas | Single-node local daemon; for distributed workloads, shared Iceberg catalogs on S3/GCS with Flight SQL provide horizontal scale-out. |
| Kibana | No proprietary UI. See "Dashboards" below. |

---

## Dashboards & observability

- **Grafana Elasticsearch datasource** — point it at `http://<host>:9200` to reuse
  existing Elasticsearch dashboards for basic search queries (the wire format is
  compatible for the supported endpoints).
- **Grafana Prometheus datasource** — scrape `GET /metrics` for operational
  dashboards (query p50/p95/p99, error rates, ingestion throughput, cache hit
  rates, in-flight requests).
- **Data dashboards** — because tables are standard Iceberg/Parquet, Grafana /
  Superset / DuckDB / Trino can query the same data directly for full SQL
  analytics (including aggregations that the REST API does not expose).

---

## Positioning

BenoStreamDB-Search is **not** a drop-in replacement for an in-memory, sub-millisecond
Elasticsearch cluster. It is positioned for **website search, document catalogs, and
log archives** where a 50–200 ms query latency envelope is imperceptible to users, and
the object-storage-native, scale-to-zero operational model (on-demand index fetch from
S3/RustFS, Iceberg/Parquet data-lake format, dynamic schema evolution, GPU-accelerated
index builds) delivers a large TCO reduction versus a JVM + hot-SSD ES deployment.


---

## Elasticsearch integration design

_(merged from the former `OPENSEARCH_COMPATIBILITY.md`)_


This document outlines the strategy for validating BenoStreamDB's production readiness and building an Elasticsearch-compatible REST API wrapper. 

The REST API is implemented as a **separate, optional add-on package (`benostreamdb-search`)** in the Cargo workspace. This ensures the core library remains lightweight, and users who only need library-level access or FFI bindings do not pull in HTTP dependencies.

---

## Architectural Thesis: Cost-Efficiency & Serverless Over Low-Latency

* **Not In-Memory**: Elasticsearch keeps all segments and caches aggressively pinned in RAM to achieve sub-millisecond query execution. This makes it operationally expensive and hard to run in serverless environments.
* **Target Workload (Website / Doc Search)**: For website search, document indexing, and corporate knowledge bases, query response latencies of **50–200ms** are perfectly acceptable.
* **Object Storage Native**: By designing for website search, we can load index files (`.hnsw` and `.idx`) and Parquet row groups **on-demand** from object storage (like S3/RustFS) or local disk cache. This enables an extremely cheap, serverless, scale-to-zero operational model.

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
- **Getting Started Guide**: Ensure [INSTALLATION.md](INSTALLATION.md) is up-to-date with current APIs and dependency instructions.
- **API Coverage**: Confirm all public catalog methods, index structures, and configuration keys are fully documented.

---

## Step 2: Lexical Processing & Search Enhancements (Crate: `benostreamdb`)

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

## Step 3: The `benostreamdb-search` Add-On Crate

Create a new package `benostreamdb-search` in the Cargo workspace root.

### 3.1 Cargo Setup
- **New Crate**: [benostreamdb-search/Cargo.toml]
- **Dependencies**: `benostreamdb`, `axum`, `tower`, `tower-http`, `tokio`, `serde`, `serde_json`.

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
- **On-Demand Loading**: Fetch the HNSW and inverted indexes from S3/RustFS on-demand, caching them locally in temp storage for future requests.

---

## Step 4: Verification & Benchmarking

### 4.1 Automated Validation
- Implement a test suite in `benostreamdb-search/tests/test_search_api.py` that verifies:
  - Document ingestion via HTTP `POST`.
  - Dynamic table creation and schema evolution.
  - Hybrid lexical + vector search queries.
  
### 4.2 Benchmark Analysis
- Measure ingestion throughput (docs/sec) and search latency (p95/p99) against a baseline local Elasticsearch instance, targeting the 50–200ms latency envelope.

---

## Step 5: Positioning: Strengths & Weaknesses vs. Elasticsearch

To effectively market this add-on, we must clearly define how it compares to Elasticsearch (ES) so users understand when to choose BenoStreamDB-Search.

### 5.1 Strengths (Competitive Advantages)
1. **Ultra-Low TCO (Total Cost of Ownership)**:
   * *Elasticsearch*: Requires expensive instance groups with huge memory allocations (JVM heaps) and fast, hot SSD storage. 
   * *BenoStreamDB*: Built for serverless object storage (S3/RustFS). When idle, it costs nothing. Index files (`.hnsw` and `.idx`) are fetched on-demand and cached locally.
2. **Open Data Lakehouse Integration**:
   * *Elasticsearch*: Uses a proprietary data format. Getting data out for analysis requires heavy ETLs or expensive scroll APIs.
   * *BenoStreamDB*: Underpinned by Apache Iceberg and Parquet. Other tools (DuckDB, Trino, Spark) can query the exact same files directly in the data lake without moving data.
3. **Dynamic Schema Evolution**:
   * *Elasticsearch*: Changing mapping types or structures often requires creating a new index and running a resource-heavy `_reindex` job.
   * *BenoStreamDB*: Inherits Iceberg's native schema evolution, allowing column additions, drops, and renaming instantly.
4. **GPU-Accelerated Hybrid Search**:
   * *Elasticsearch*: Vector search runs on standard JVM threads.
   * *BenoStreamDB*: Native vector indexing with SIMD/GPU acceleration for low-cost, high-scale HNSW execution.

### 5.2 Weaknesses (Where ES Wins & How to Position It)
1. **Sub-Millisecond Query Latency**:
   * *Elasticsearch*: Sub-5ms response times due to aggressive in-memory caching.
   * *BenoStreamDB*: 50–200ms response times due to S3-native fetch overhead.
   * *Positioning*: Position BenoStreamDB-Search for **website search, document catalogs, and log archives** where 50–200ms is imperceptible to users, but the 90% hosting cost reduction is highly compelling.
2. **Ecosystem & Dashboarding**:
   * *Elasticsearch*: Has mature Kibana integration for visualization and dashboarding.
   * *BenoStreamDB*: No proprietary visualization frontend.
   * *Leveraging Grafana, Prometheus & Apache Superset*:
     - **Prometheus Metrics Endpoint**: The `benostreamdb-search` REST server will expose a `/metrics` endpoint (via the `prometheus` crate, already a dependency) providing operational telemetry: query latency histograms, ingestion throughput counters, index cache hit rates, and active connection gauges.
     - **Grafana + Prometheus Stack**: Grafana natively scrapes Prometheus endpoints, giving operators real-time operational dashboards (query p95/p99, error rates, ingestion backpressure) out of the box with zero custom code.
     - **Grafana via DuckDB / Trino (Data Dashboards)**: For data-level dashboards, Grafana connects to DuckDB and Trino. Since our tables are standard Iceberg, Grafana can query and visualize BenoStreamDB data with full SQL support.
     - **Apache Superset**: The premier open-source BI tool for data lakes natively supports Trino, Spark, and DuckDB, offering a robust Kibana-like log viewing and dashboarding experience.
     - **Elasticsearch API Compatibility**: By ensuring our REST API matches standard Elasticsearch query endpoints, users can configure **Grafana's built-in Elasticsearch datasource** to query BenoStreamDB directly, providing zero-friction dashboard reuse.

