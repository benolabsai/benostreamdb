# BenoStreamDB Roadmap & Architecture Status

**Vision**: Make Apache Iceberg executable as a high-performance database with rebuildable secondary index overlays (CSR graph, HNSW vector, BM25 text, Roaring bitmaps).

---

## 🎯 Architecture & Status at a Glance

| Status | Capability Area | Key Components / Interfaces |
| :--- | :--- | :--- |
| ✅ **Production Ready** | **Core Lakehouse Engine** | Apache Iceberg V2/V3 format, Parquet storage, DataFusion ANSI SQL, MVCC lock-free commits, WAL, Ingest RAM backpressure. |
| ✅ **Production Ready** | **Concurrency & Durability** | Unified atomic write buffer (`PendingWrite`), Table-wide maintenance barriers (`maintenance_lock`), Fenced vacuum/orphan GC. |
| ✅ **Production Ready** | **Vector Search Overlays** | Zero-copy Arrow IPC HNSW & IVF-PQ, TurboQuant™ (TQ4/TQ8), SIMD intrinsics, GPU acceleration (`benostream-gpu-ann`: CUDA, Metal, Vulkan). |
| ✅ **Production Ready** | **Text & Scalar Overlays** | Inverted BM25 Okapi indexes, Roaring Bitmap scalar indexes, Composite multi-column filters, Statistics pruning. |
| ✅ **Production Ready** | **Core Graph Primitives** | Zero-copy Memory-Mapped CSR Graph Index (`MmapCsrGraph`), Core Graph Traversal operator (`Table::graph_neighborhood`), Graph SQL UDFs. |
| ✅ **Production Ready** | **Lakehouse Connectors** | Native Apache Spark connector (`spark-benostreamdb`), Trino connector (`trino-benostreamdb`), Official dbt adapter (`dbt-benostreamdb`). |
| ✅ **Production Ready** | **Ecosystem Interfaces** | Optional Arrow Flight SQL server (`server/flight_sql`), Contrib Search Gateway (`contrib/benostreamdb-search`: OpenSearch 7.10 & Qdrant REST). |
| 📋 **Active Roadmap** | **Reactive & Search Extensions** | Reactive Flight Subscriptions (Live Queries), Declarative Edge Tables, Domain-Specific BM25 Analyzers, AI Agent MCP Server. |

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

---

## 📋 Active Roadmap Themes

### Theme 1: Reactive Lakehouse Streaming & Arrow Flight Subscriptions ("Live Queries")
*Bringing change data capture (CDC) and reactive streaming to the lakehouse.*

- [ ] **`Table::subscribe()` Core API**:
  - Internal broadcast channel (`tokio::sync::broadcast`) emitting committed `RecordBatch` streams and metadata events on snapshot commits.
- [ ] **Arrow Flight SQL Streaming Subscriptions**:
  - Implement Flight `DoExchange` / `DoGet` streaming endpoints allowing remote clients (Python, DuckDB, web dashboards) to subscribe to table diffs in real time without polling.
- [ ] **Predicate-Filtered Subscriptions**:
  - Pushdown filter evaluation against incoming streaming batches before emitting to subscribers.

### Theme 2: Declarative Edge Tables & Multi-Table Graph Overlays
*Elevating knowledge graphs into a first-class lakehouse data modeling primitive.*

> **Partially delivered.** The programmatic edge-table API has shipped:
> `Table.create_edge_table(...)`, `Table.from_networkx(...)`, and
> `Table.to_networkx(...)` create and round-trip standard edge tables with
> automatic endpoint/embedding sidecar indexes. The declarative DDL form
> (`table_type = 'edge'`) and typed global URNs below remain open.

- [ ] **Declarative Edge Table DDL**:
  - First-class table metadata options (`table_type = 'edge'`, `src_col`, `dst_col`) that automatically register and maintain forward and reverse CSR indexes on segment commits.
- [ ] **Typed Global Entity URNs (`table:id`)**:
  - Extend the CSR graph dictionary to support composite entity identifiers (e.g. `users:101`, `orders:5002`) so a single CSR graph overlay can seamlessly traverse across heterogeneous Iceberg tables.
- [ ] **SQL Graph Traversal Operator Extensions**:
  - Expand DataFusion table functions to expose graph walks directly in `FROM` clauses:
    ```sql
    SELECT * FROM graph_neighbors('edges', seeds => [101], hops => 2);
    ```

### Theme 3: Domain-Specific Text Analyzers for BM25
*Specialized tokenization pipelines for enterprise code, legal, and multilingual knowledge bases.*

- [ ] **Configurable Table Analyzer Configurations**:
  - Allow table schemas to declare custom text analyzers in metadata (`analyzer = 'code_camelcase'`, `stopwords = 'en'`).
- [ ] **Built-in Tokenizer Suite**:
  - **Code Tokenizer**: Splits on CamelCase, snake_case, and symbol delimiters for source code search.
  - **Multilingual Stemming**: Integration of Snowball stemmers for English, Spanish, French, and German.
  - **CJK Segmentation**: Character/bi-gram tokenization for Chinese, Japanese, and Korean corpora.

### Theme 4: AI Agent Tools & Client Ecosystem
*Native interfaces for AI frameworks and autonomous agent runtimes.*

- [ ] **Official Model Context Protocol (MCP) Server (`contrib/benostreamdb-mcp`)**:
  - Dedicated MCP server exposing BenoStreamDB tables, schemas, SQL queries, and hybrid vector/graph retrieval as native tools for Claude Desktop, Cursor, and Antigravity.
- [ ] **LangChain & LlamaIndex Partner Integrations**:
  - `langchain-benostreamdb`: LangChain-compatible `VectorStore` and `Retriever` implementations with metadata filtering.
  - `llama-index-vector-stores-benostreamdb`: LlamaIndex vector store index adapter.
- [ ] **Broker Event Adapters**:
  - Pluggable Kafka, RabbitMQ, and NATS consumers implementing the `WorkCoordinator` trait for automated event-driven lakehouse ingestion.
