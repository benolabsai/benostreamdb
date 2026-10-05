# BenoStreamDB Roadmap & Architecture Status

**Vision**: Make Apache Iceberg executable as a high-performance database with rebuildable secondary index overlays (CSR graph, HNSW vector, BM25 text, Roaring bitmaps).

---

## 🚀 Launch Candidate (pre-announcement milestone)

The benchmark report already demonstrates a stronger technical story than the
feature list alone: transactional correctness, crash recovery, multi-writer
concurrency, object-store faults, skew, soak behaviour, and competitive
retrieval performance. **The recommendation is therefore to announce sooner
rather than delay for more features** — provided the benchmark methodology and
reproducibility are clean enough to publish.

### The product story

> **Add specialized search and graph indexes directly to transactional Iceberg
> data, without turning your lakehouse into a collection of separate
> databases.**

The overlay-index architecture *is* the product story, not any individual
index. The vector benchmark, graph benchmark, crash tests, multi-writer tests,
and object-store resilience tests collectively support that story far better
than a long feature checklist.

**Positioning discipline.** Do not market the benchmark numbers as broad
claims. The SIFT comparison is only 20,000 vectors, and the dedicated
in-memory engines are optimised for a narrower problem than an Iceberg-native
architecture. Say:

> **"BenoStreamDB delivers sub-millisecond vector search while maintaining
> transactional Iceberg tables and persistent overlay indexes."**

— not *"BenoStreamDB is faster than FAISS."*

### Do now (priority order)

| Priority | Item | Theme |
| :--- | :--- | :--- |
| ✅ Done | **JSON-path inverted index** — `IndexAlgorithm::JsonPath` builds a `(path, value) -> row_ids` Puffin overlay; the planner rewrites `json_contains` / `json_exists` / `json_path_exists` / `json_extract_path_text(...) = 'value'` into index lookups, with a full-scan fallback for unindexed paths. Completes the story (SQL + BM25 + vector + graph + JSON-path indexes, all rebuildable overlays on Iceberg). | [Theme 2](#theme-2-semi-structured--json) |
| 🔴 High | **MCP server** — excellent demo surface for the AI story | [Theme 1](#theme-1-ai-agent-tools--client-ecosystem) |
| 🔴 High | **SQL graph traversal** (`graph_neighbors(...)`) — turns the graph capability into something immediately demonstrable | [Theme 3](#theme-3-graph-overlays) |
| 🔴 High | **Documentation + benchmark reproducibility** — treat benchmarks as product work | [Benchmarks](#benchmarks-as-product-work) |
| 🟠 High | **Predicate-filtered `Table::subscribe()`** — only if reactive/streaming is going to be a headline | [Theme 4](#theme-4-reactive-lakehouse-streaming) |

### Do not block the announcement on

These are valuable but do not establish the core differentiation, or are
substantial engineering for little initial marketing benefit:

- **Arrow Flight subscriptions** — build `Table::subscribe()` first; add Flight
  once someone actually needs remote subscriptions.
- **Typed global entity URNs** — powerful architecture, hard to explain and
  validate initially.
- **Full tokenizer suite** (code / multilingual / CJK) — useful, but not core
  differentiation.
- **LangChain / LlamaIndex integrations** — MCP is the stronger first
  integration; add these when there is demand.
- **Kafka / RabbitMQ / NATS adapters** — integration work, not differentiation.
- **Iceberg v3 `variant`** — standards alignment, not a compelling launch
  differentiator.

> **Enterprise features live in the business plan, not this roadmap.** Security,
> governance, autonomous maintenance, HA/DR, and observability are an
> **operational control plane around the engine**, sequenced separately for the
> commercial product. See
> [`business_plan/ENTERPRISE_ROADMAP.md`](../business_plan/ENTERPRISE_ROADMAP.md).

---

## 🎯 Architecture & Status at a Glance

| Status | Capability Area | Key Components / Interfaces |
| :--- | :--- | :--- |
| ✅ **Production Ready** | **Core Lakehouse Engine** | Apache Iceberg V2/V3 format, Parquet storage, DataFusion ANSI SQL, MVCC lock-free commits, WAL, Ingest RAM backpressure. |
| ✅ **Production Ready** | **Concurrency & Durability** | Unified atomic write buffer (`PendingWrite`), Table-wide maintenance barriers (`maintenance_lock`), Fenced vacuum/orphan GC. |
| ✅ **Production Ready** | **Vector Search Overlays** | Zero-copy Arrow IPC HNSW & IVF-PQ, TurboQuant™ (TQ4/TQ8), SIMD intrinsics, GPU acceleration (`benostream-gpu-ann`: CUDA, Metal, Vulkan). |
| ✅ **Production Ready** | **Text & Scalar Overlays** | Inverted BM25 Okapi indexes, Roaring Bitmap scalar indexes, Composite multi-column filters, Statistics pruning. |
| ✅ **Production Ready** | **Core Graph Primitives** | Zero-copy Memory-Mapped CSR Graph Index (`MmapCsrGraph`), Core Graph Traversal operator (`Table::graph_neighborhood`), Graph SQL UDFs. |
| ✅ **Production Ready** | **JSON Path Functions & Index** | PostgreSQL `json_*` UDFs (`json_extract_path`, `json_contains`, `json_path_query`, …) over `Utf8` columns, plus a `json_path` `(path, value) -> row_ids` Puffin overlay with planner pushdown. |
| ✅ **Production Ready** | **Lakehouse Connectors** | Native Apache Spark connector (`spark-benostreamdb`), Trino connector (`trino-benostreamdb`), Official dbt adapter (`dbt-benostreamdb`). |
| ✅ **Production Ready** | **Ecosystem Interfaces** | Optional Arrow Flight SQL server (`server/flight_sql`), Contrib Search Gateway (`contrib/benostreamdb-search`: OpenSearch 7.10 & Qdrant REST). |
| 🚀 **Launch Candidate** | **AI-Native & Search Extensions** | MCP server, SQL graph traversal, predicate-filtered subscriptions. |
| 📋 **Deferred** | **Breadth** | Arrow Flight subscriptions, typed global URNs, tokenizer suite, LangChain/LlamaIndex, broker adapters, Iceberg v3 `variant`. |
| 🏢 **Business Plan** | **Enterprise Control Plane** | Auth/RBAC/RLS, audit, autonomous maintenance, observability, HA/DR, and CMEK. |
---

## 🏛️ Architectural Foundations (Completed)

BenoStreamDB implements an **indexed, compute-disaggregated lakehouse architecture**:
1. **The Overlay Invariant**: Parquet data files in Apache Iceberg are the authoritative source of truth. All secondary indexes (CSR graph, HNSW vector, BM25 text, Roaring bitmaps) are **advisory, rebuildable sidecar overlays**. If an index is lost or deleted, queries fall back to Parquet scans while the overlay rebuilds.
2. **Serverless-First, Server-Optional**: The core database is an in-process, embeddable library (Python/Rust/JVM) operating directly against object storage (S3/GCS/Azure/Local). Arrow Flight SQL provides an optional single-process network access layer for remote clients and BI tools.
3. **Lakehouse Independence**: All core connectors (`dbt-benostreamdb`, `spark-benostreamdb`, `trino-benostreamdb`) and optional servers (`server/flight_sql`) operate independently of contrib compatibility packages.

### Key Completed Milestones
- **Core Concurrency Hardening (H1, H2, H3, M1)**:
  - Unified `pending_writes: Arc<RwLock<Vec<PendingWrite>>>` ensuring atomic WAL-to-buffer pairing.
  - Table-level `maintenance_lock` establishing an exclusive barrier between concurrent writes and destructive operations (`truncate`, `vacuum`, `compaction`).
  - Strict manifest load error propagation across all mutation paths.
  - Regression verified under parallel stress testing (`tests/test_h1_h2_concurrency_regression.rs`).
- **Core Graph Engine**:
  - Memory-mapped Compressed Sparse Row (`.graph_v2.csr.{offsets,edges,dict}`) engine.
  - Core `Table::graph_neighborhood()` traversal operator with fast-path CSR execution, relation filtering, and unindexed edge fallback.
  - Graph-scoped vector and BM25 queries callable directly from Python, SQL, and search APIs.
- **Native Ingest Orchestrator (`Table::ingest_async`)**:
  - Cluster-free bulk ingestion with memory-budgeted chunking (`malloc_trim` heap policy), lock-free OCC manifest commits, and resume idempotency.
- **Multi-Catalog Resolution**:
  - Seamless integration with Apache Polaris, Nessie, AWS Glue, Hive Metastore, and file-based Iceberg catalogs.
- **PostgreSQL `json` Path Functions**:
  - `json_extract_path` / `json_extract_path_text` (variadic path), `json_contains` (recursive `@>`), `json_exists`, `json_typeof`, and `json_path_exists` / `json_path_query` (jsonpath subset: `$.a.b[0]`, `[*]`). Implemented in `src/core/sql/udf/json.rs`.

---

## 📋 Active Roadmap Themes

Themes are ordered by Launch Candidate priority.

### Theme 1: AI Agent Tools & Client Ecosystem
*Native interfaces for AI frameworks and autonomous agent runtimes.* — 🔴 **High**

- [ ] **Official Model Context Protocol (MCP) Server (`contrib/benostreamdb-mcp`)**:
  - Dedicated MCP server exposing BenoStreamDB tables, schemas, SQL queries, and hybrid vector/graph retrieval as native tools for Claude Desktop, Cursor, and Antigravity.
  - The demo that matters: an agent that (1) discovers a table, (2) inspects schema, (3) executes SQL, (4) performs vector retrieval, (5) traverses an edge table, and (6) combines the results.
- [ ] **LangChain & LlamaIndex Partner Integrations** — 🟢 **Deferred**:
  - `langchain-benostreamdb`: LangChain-compatible `VectorStore` and `Retriever` implementations with metadata filtering.
  - `llama-index-vector-stores-benostreamdb`: LlamaIndex vector store index adapter.
- [ ] **Broker Event Adapters** — 🟢 **Deferred**:
  - Pluggable Kafka, RabbitMQ, and NATS consumers implementing the `WorkCoordinator` trait for automated event-driven lakehouse ingestion.

### Theme 2: Semi-Structured & JSON
*First-class access to JSON documents without forcing a rigid column schema.* — 🔴 **High**

> **Current state.** JSON is stored as a `Utf8` string column (text, not a
> decomposed binary form, so the functions use PostgreSQL's `json_*` names, not
> `jsonb_*`). The search gateway
> (`contrib/benostreamdb-search/src/infer.rs`) already infers an Arrow schema
> from sample documents (ES 7.10 dynamic mapping).

- [x] **JSON Path Functions** (shipped):
  - `json_extract_path` / `json_extract_path_text` (variadic path),
    `json_contains` (recursive `@>`), `json_exists`, `json_typeof`, and
    `json_path_exists` / `json_path_query` (jsonpath subset: `$.a.b[0]`, `[*]`).
    Implemented in `src/core/sql/udf/json.rs`; DataFusion 52 ships no JSON
    module.
- [x] **JSON-Path Inverted Index** (shipped) — 🔴 **High**:
  - A rebuildable overlay over selected JSON paths (e.g. `$.user.id`) that
    accelerates equality/containment filters without a full scan, following the
    Overlay Invariant. `IndexAlgorithm::JsonPath { paths }` builds a
    `(path, value) -> row_ids` Parquet overlay in `src/core/index/json_path.rs`,
    packed into the segment's Puffin bundle under the `json_path` category. The
    planner rewrites `json_contains` / `json_exists` / `json_path_exists` /
    `json_extract_path_text(...) = 'value'` into index lookups
    (`HybridReader::query_json_path_first`), falling back to a full scan for
    unindexed paths. The index records a presence marker and one level of array
    elements so every lookup is a *superset* of the true matches; the filter is
    re-applied above the scan. Verified by a differential oracle against a full
    scan (`tests/test_json_path_index.rs`).
  - **Scope discipline:** equality on a configured path set (`$.user.id`,
    `$.customer.id`, `$.metadata.foo`) plus containment. The full JSONPath
    universe (recursive descent, filters) is out of scope.
  - The story it unlocks: *"Add an index to an existing Iceberg table without
    rewriting the table."*
- [ ] **Iceberg v3 `variant` Mapping** — 🟢 **Deferred**:
  - Map semi-structured columns to Iceberg v3 `variant` (the spec-native
    semi-structured type) so JSON round-trips through catalogs instead of being
    a bespoke engine-only type; Arrow-side representation is `Struct`/`Utf8`
    with path pushdown. This is a storage-fidelity improvement — it still needs
    the JSON-path index above to make containment fast.

### Theme 3: Graph Overlays
*Elevating knowledge graphs into a first-class lakehouse data modeling primitive.* — 🟠 **High**

> **Partially delivered.** The programmatic edge-table API has shipped:
> `Table.create_edge_table(...)`, `Table.from_networkx(...)`, and
> `Table.to_networkx(...)` create and round-trip standard edge tables with
> automatic endpoint/embedding sidecar indexes. The declarative DDL form
> (`table_type = 'edge'`) and typed global URNs below remain open.

- [ ] **SQL Graph Traversal Operator Extensions** — 🟠 **High**:
  - Expand DataFusion table functions to expose graph walks directly in `FROM`
    clauses. This is the demo that matters — *"the graph is an overlay on your
    lakehouse data, not a separate graph database"*:
    ```sql
    SELECT * FROM graph_neighbors('edges', seeds => [101], hops => 2);
    ```
- [ ] **Declarative Edge Table DDL** — 🟠 **High**:
  - First-class table metadata options (`table_type = 'edge'`, `src_col`, `dst_col`) that automatically register and maintain forward and reverse CSR indexes on segment commits.
- [ ] **Typed Global Entity URNs (`table:id`)** — 🟢 **Deferred**:
  - Extend the CSR graph dictionary to support composite entity identifiers (e.g. `users:101`, `orders:5002`) so a single CSR graph overlay can seamlessly traverse across heterogeneous Iceberg tables.

### Theme 4: Reactive Lakehouse Streaming
*Bringing change data capture (CDC) and reactive streaming to the lakehouse.* — 🟠 **High**

- [ ] **`Table::subscribe()` Core API** — 🟠 **Medium**:
  - Internal broadcast channel (`tokio::sync::broadcast`) emitting committed `RecordBatch` streams and metadata events on snapshot commits.
- [ ] **Predicate-Filtered Subscriptions** — 🟠 **High**:
  - Pushdown filter evaluation against incoming streaming batches before emitting to subscribers. This is what makes Live Queries genuinely useful rather than merely technically interesting.
- [ ] **Arrow Flight SQL Streaming Subscriptions** — 🟢 **Deferred**:
  - Implement Flight `DoExchange` / `DoGet` streaming endpoints allowing remote clients (Python, DuckDB, web dashboards) to subscribe to table diffs in real time without polling. Build this once somebody actually needs remote subscriptions.

### Theme 5: Domain-Specific Text Analyzers for BM25
*Specialized tokenization pipelines for enterprise code, legal, and multilingual knowledge bases.* — 🟢 **Deferred**

- [ ] **Configurable Table Analyzer Configurations**:
  - Allow table schemas to declare custom text analyzers in metadata (`analyzer = 'code_camelcase'`, `stopwords = 'en'`).
- [ ] **Built-in Tokenizer Suite**:
  - **Code Tokenizer**: Splits on CamelCase, snake_case, and symbol delimiters for source code search.
  - **Multilingual Stemming**: Integration of Snowball stemmers for English, Spanish, French, and German.
  - **CJK Segmentation**: Character/bi-gram tokenization for Chinese, Japanese, and Korean corpora.

---

## Benchmarks as Product Work

Benchmarks and demonstrations are part of the product, not marketing polish.
The Launch Candidate requires **excellent end-to-end benchmarks** and
**documentation/tutorials** that show the three workflows concretely.

- [ ] **End-to-end benchmark suite** — 🔴 **High**:
  - Vector ANN (SIFT/HNSW/TurboQuant), graph analytics, SQL OLAP, lexical &
    hybrid (BEIR), production concurrency/maintenance/recovery, and the
    shared-envelope Docker competitor matrix (FAISS, hnswlib, LanceDB, pgvector,
    OpenSearch, Neo4j+GDS, cuGraph).
- [ ] **Workflow tutorials** — 🔴 **High**:
  - One runnable tutorial per workflow: (1) AI/semantic retrieval via MCP,
    (2) JSON/document retrieval with a JSON-path index, (3) graph traversal via
    `graph_neighbors(...)`.
