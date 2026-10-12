# BenoStreamDB Roadmap & Architecture Status

**Vision**: Make Apache Iceberg executable as a high-performance database with rebuildable secondary index overlays (CSR graph, HNSW vector, BM25 text, Roaring bitmaps).

---

## 🎯 Architecture & Status at a Glance

| Status | Capability Area | Key Components / Interfaces |
| :--- | :--- | :--- |
| ✅ **Developer Preview** | **Core Lakehouse Engine** | Apache Iceberg V2/V3 format, Parquet storage, DataFusion ANSI SQL, MVCC lock-free commits, WAL, Ingest RAM backpressure. |
| ✅ **Developer Preview** | **Concurrency & Durability** | Unified atomic write buffer (`PendingWrite`), Table-wide maintenance barriers (`maintenance_lock`), Fenced vacuum/orphan GC. |
| ✅ **Developer Preview** | **Vector Search Overlays** | Zero-copy Arrow IPC HNSW & IVF-PQ, TurboQuant™ (TQ8; TQ4 retiring), SIMD intrinsics, GPU acceleration (`benostream-gpu-ann`: CUDA, Metal, Vulkan). |
| ✅ **Developer Preview** | **Text & Scalar Overlays** | Inverted BM25 Okapi indexes, Roaring Bitmap scalar indexes, Composite multi-column filters, Statistics pruning. |
| ✅ **Developer Preview** | **Core Graph Primitives** | Zero-copy Memory-Mapped CSR Graph Index (`MmapCsrGraph`), Core Graph Traversal operator (`Table::graph_neighborhood`), Graph SQL UDFs. |
| ✅ **Developer Preview** | **JSON Path Functions & Index** | PostgreSQL `json_*` UDFs (`json_extract_path`, `json_contains`, `json_path_query`, …) over `Utf8` columns, plus a `json_path` `(path, value) -> row_ids` Puffin overlay with planner pushdown. |
| ✅ **Developer Preview** | **Lakehouse Connectors** | Native Apache Spark connector (`spark-benostreamdb`), Trino connector (`trino-benostreamdb`), Official dbt adapter (`dbt-benostreamdb`). Full engine function-surface parity across Python/dbt/Spark/Trino, guarded by `tests/python/test_function_parity.py`. |
| ✅ **Developer Preview** | **Ecosystem Interfaces** | Optional Arrow Flight SQL server (`server/flight_sql`), Contrib Search Gateway (`contrib/benostreamdb-search`: OpenSearch 7.10 & Qdrant REST), stateless auth (API key + JWT, shared module). |
| ✅ **Developer Preview** | **AI-Native & Search Extensions** | MCP server (`contrib/benostreamdb-mcp`), SQL graph traversal (`FROM graph_neighbors(...)`), `Table::subscribe()` change feed + predicate-filtered subscriptions. |
---

## 🏛️ Architectural Foundations (Completed)

BenoStreamDB implements an **indexed, compute-disaggregated lakehouse architecture**:
1. **The Overlay Invariant**: Parquet data files in Apache Iceberg are the authoritative source of truth. All secondary indexes (CSR graph, HNSW vector, BM25 text, Roaring bitmaps) are **advisory, rebuildable sidecar overlays**. If an index is lost or deleted, queries fall back to Parquet scans while the overlay rebuilds.
2. **Serverless-First, Server-Optional**: The core database is an in-process, embeddable library (Python/Rust/JVM) operating directly against object storage (S3/GCS/Azure/Local). Arrow Flight SQL provides an optional single-process network access layer for remote clients and BI tools.
3. **Lakehouse Independence**: All core connectors (`dbt-benostreamdb`, `spark-benostreamdb`, `trino-benostreamdb`) and optional servers (`server/flight_sql`) operate independently of contrib compatibility packages.

### Key Completed Milestones
- **Core Concurrency Hardening**:
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
- **Cross-connector function parity**:
  - The engine's full custom function surface (vector scalar UDFs, vector
    aggregates, JSON, graph UDAFs) is exposed consistently across Python, dbt,
    Spark, and Trino, and guarded by `tests/python/test_function_parity.py`
    (source of truth: `udf::registered_function_names()`). The Spark DSv2
    `FunctionCatalog` exposes 30 functions, and a pass-through reader
    (`option("query", "<engine SQL>")`) makes the vector aggregates and graph
    UDAFs — which have no DSv2 scalar equivalent — reachable from Spark. The
    engine's ad-hoc `Table::sql()` context now registers the full surface via the
    shared `udf::register_all_custom_udfs` helper.
- **Ingest back-pressure liveness**:
  - The ingest RAM high-water mark is now baseline-relative (it bounds the
    engine's *growth* since the write began) with a bounded fail-open grace
    (`BSDB_INGEST_BACKPRESSURE_GRACE_SECS`, default 60s), so a writer can no
    longer block forever when process RSS is already at/above the cap for reasons
    the write cannot relieve.
- **MCP Server (`contrib/benostreamdb-mcp`)**:
  - Dedicated MCP server exposing BenoStreamDB tables, schemas, SQL queries, and
    hybrid vector/graph retrieval as native tools for Claude Desktop, Cursor, and
    Antigravity. Windows `.msi` installer via `cargo-wix`.
- **JSON-Path Inverted Index**:
  - `IndexAlgorithm::JsonPath { paths }` builds a `(path, value) -> row_ids`
    Parquet overlay in `src/core/index/json_path.rs`, packed into the segment's
    Puffin bundle under the `json_path` category. The planner rewrites
    `json_contains` / `json_exists` / `json_path_exists` /
    `json_extract_path_text(...) = 'value'` into index lookups
    (`HybridReader::query_json_path_first`), falling back to a full scan for
    unindexed paths. Verified by a differential oracle against a full scan
    (`tests/test_json_path_index.rs`).
- **SQL Graph Traversal Operator Extensions**:
  - DataFusion table functions expose graph walks directly in `FROM` clauses
    (`graph_neighbors`, `graph_shortest_path`, `graph_all_shortest_paths`,
    `graph_subgraph`, `graph_connecting_paths`), following the in-memory /
    out-of-core `GraphMode` pattern. Implemented in
    `src/core/sql/graph_udf/graph_table_functions.rs`; reachable from Python,
    dbt, Spark (pass-through), Trino, and Flight SQL.
- **Declarative Edge Table DDL**:
  - First-class table metadata (via `Manifest.properties`): `table_type`
    (`node` | `edge` | `table`), `src_col`, `dst_col`, `relation_col`,
    `weight_col`, `id_col`, `label_col`. `create_edge_table` /
    `create_node_table` stamp the convention; `Session.list_graph_tables()`
    enumerates them. Declaring `table_type = 'edge'` auto-configures the forward
    and reverse CSR graph indexes (`Table::ensure_edge_indexes_async`).
- **`Table::subscribe()` + predicate-filtered subscriptions**:
  - Per-table `tokio::sync::broadcast` channel emitting committed `RecordBatch`es
    plus a `Commit` marker; exposed in Python, as the `subscribe_events` SQL table
    function, a Flight streaming ticket, and an MCP tool.
    `Table::subscribe_filtered("age > 30")` re-applies the predicate above each
    batch.
- **End-to-end benchmark suite**:
  - Vector ANN (SIFT/HNSW/TurboQuant), graph analytics, SQL OLAP, lexical &
    hybrid (BEIR), production concurrency/maintenance/recovery, and the
    shared-envelope Docker competitor matrix (FAISS, hnswlib, LanceDB, pgvector,
    OpenSearch, Neo4j+GDS, cuGraph).
- **Workflow tutorials**:
  - One runnable tutorial per workflow: (1) AI/semantic retrieval via MCP,
    (2) JSON/document retrieval with a JSON-path index, (3) graph traversal via
    `graph_neighbors(...)`. Python versions are Jupyter notebooks
    (`examples/tutorials/*.ipynb`, verified to execute end-to-end); the
    SQL/connector forms (Spark/Trino/dbt/Flight/MCP/Rust) are markdown pages
    under `docs/tutorials/`.

---

## 📋 Active Roadmap Themes

Themes are ordered by priority.

### Theme 1: Deprecate & Remove TQ4 Quantization Engine Option — 🔴 **Highest Priority**
*Retire 4-bit TurboQuant (`hnsw_tq4` / `tq4`) across the engine, APIs, and Puffin sidecars.*

> **Rationale & Benchmark Findings.** Comprehensive benchmarking demonstrated that 4-bit TurboQuant (`tq4`) performs significantly worse in recall compared to full-precision `hnsw` and 8-bit TurboQuant (`tq8`), without delivering any compelling added value or Pareto efficiency. `tq8` already provides ~4× compression with resilient recall, and product quantization (`pq`) serves extreme compression use cases far more effectively.
>
> While TQ4 has been removed from benchmark suites, competitor matrixes, and reports for the upcoming release, its full engine code removal is scheduled as the highest priority item for the next release (respecting the current feature freeze).

- [ ] **Remove `IndexAlgorithm::HnswTq4` variant and parser** — 🔴 **Highest Priority**:
  - Deprecate and remove `IndexAlgorithm::HnswTq4`, `IndexAlgorithm::hnsw_tq4()`, and parser mappings (`"hnsw_tq4"`, `"tq4"`).
- [ ] **Remove TQ4 quantization pipeline from HNSW/IVF and Segment writers** — 🔴 **Highest Priority**:
  - Clean up `QuantizerImpl::TurboQuant` 4-bit path in `src/core/index/hnsw_ivf.rs`, `PUFFIN_BLOB_VECTOR_HNSW_TQ4` in `src/core/puffin.rs`, and segment index writers.
- [ ] **Clean connector and language bindings** — 🔴 **Highest Priority**:
  - Remove Python `table.quantize(..., "tq4")`, PyO3 helper parsing, MCP tool prompts, Spark/dbt catalog descriptions, and surface parity tests.
- [ ] **Clean documentation and test suites** — 🔴 **Highest Priority**:
  - Remove `differential_ann_tq4_recall` and `explain_reports_tq4_index_access_path` tests, update vector API documentation.

### Theme 2: Multi-Precision Vector Indexes
*Let one vector column serve both exact and compressed search.* — 🔴 **High**

> **Current state.** A vector column holds a **single** index: adding a second
> replaces the first (see `Table::add_index` in `src/core/table/index_config.rs`),
> and the default is now full-precision `hnsw`. Quantized variants are opt-in:
> `hnsw_tq8` (TurboQuant 8-bit, ≈4× smaller) and `hnsw_pq` — trading recall for
> a smaller index and cheaper distance computation. The reader already ranks
> multiple index files for a column (`vector_search_index_raw`), but with
> float-first precedence and no per-query override.

- [ ] **Multiple vector indexes per column + query-time selection** — 🟠 **High**:
  - Allow a column to carry a float `hnsw` (exact / best recall) **and** a
    TurboQuant `hnsw_tq8`/`hnsw_pq` index, chosen per query, e.g.
    `table.vector_search(..., index='tq8')` — exact precision for
    recall-critical reads, the compressed index for latency-sensitive scans.
  - Requires: (1) `add_index` to **append** a vector variant instead of replacing
    the whole family; (2) a query-time selector threaded through
    `vector_search_index`/`vector_search_index_raw`/`vector_search_scored` and
    the Python/SQL surfaces; (3) defining precedence when no selector is given
    (full precision first).

### Theme 3: Overlay Indexes on Read-Only Delta & Hudi (Native Rust Translator)
*Attach the overlay indexes to existing Delta / Hudi tables — writing only indexes, never data.* — 🔴 **High**

> **Current state.** Delta and Hudi tables are Parquet under the hood, and the
> overlay index engine already treats any Parquet file as data. The authoritative
> data files and the advisory index overlays are stored completely separately
> (`s3://my-index-bucket/overlays/`), so the upstream table does not even need to
> know BenoStreamDB exists. The existing
> `Table.register_external(overlay_uri, iceberg_metadata_uri)` path already
> attaches an overlay to an external, read-only Iceberg table without write
> permission on the upstream storage or catalog.
>
> **Approach: a read-side `TableFormat` trait, not a metadata materializer.**
> XTable/UniForm solve a *bidirectional* metadata-sync problem (Delta↔Iceberg↔Hudi
> with write-back); we need one direction, read-only, in-process. So we do **not**
> materialize Iceberg metadata files. Instead, generalize the read path behind a
> `TableFormat` trait that produces the existing `Manifest`/`ManifestEntry`/
> `DeleteFile` structures directly: `IcebergFormat` is today's code, and
> `DeltaFormat`/`HudiFormat` parse the native logs. Nothing is written upstream
> and no Avro manifests are generated — the overlay bucket holds only our
> sidecars. Both existing translators are JVM-based (**Apache XTable** is Java;
> **Delta UniForm** is Scala/Java), so a native Rust implementation avoids a JVM
> runtime dependency entirely.

- [ ] **Extract a read-side `TableFormat` trait** — 🟠 **High**:
  - `load_snapshot()` → data files + delete files + schema + partition spec +
    snapshot id; `read_deletes()`; `partition_transform()`. `IcebergFormat` is a
    lift-and-shift of today's read path (pure refactor, no behaviour change).
- [ ] **`DeltaFormat` (read-only)** — 🟠 **High**:
  - Parse `_delta_log/NNN.json` (newline-delimited `add`/`remove`/`metaData`/
    `protocol` actions) + `.checkpoint.parquet`; merge-on-read via **deletion
    vectors** (Delta 3.x, RoaringBitmap `.bin` — we already have Roaring). The
    quick win: a bounded, well-documented parser.
- [ ] **`HudiFormat` (read-only)** — 🟢 **Deferred**:
  - Parse the `.hoodie/` timeline (commit/deltacommit instants) + base Parquet +
    MOR **log files** (Avro `HoodieLogBlock`). COW first, MOR later; Hudi 1.0's
    timeline format change is a moving target. **The long pole.**
- [ ] **Incremental-on-staleness builds (tail the log, don't `LIST`)** — 🟠 **High**:
  - The upstream writer is external (Databricks/Flink), so we cannot hook its
    stream. A background crawler **tails the latest `_delta_log` entry / timeline
    instant** and compares it to the last-seen version — no full `LIST`, no
    per-file `stat`; steady-state cost is one small `GET` per poll. On change, it
    incrementally builds indexes for the **new files only** (HNSW append-insert +
    BM25 posting merge). Poll interval adapts to the observed change rate
    (backoff + jitter), with an explicit `refresh()` for query-time staleness.
- [ ] **Out-of-band append correctness (the Overlay Invariant)** — 🟠 **High**:
  - Until the background build catches up, the planner searches the existing
    HNSW/BM25 index for the old data and falls back to a Parquet scan for the
    newly appended files — 100% snapshot correctness at all times.
- [ ] **Out-of-core sidecars (mmap)** — 🟢 **Deferred**:
  - Extend the existing `MmapCsrGraph` pattern to the HNSW/BM25 sidecars: mmap
    the graph/postings, keep only the hot top-level (centroids, entry points)
    resident, and page in on demand via `BSDB_DISK_CACHE_DIR`. This is the only
    way a sidecar can exceed host memory.

### Theme 4: Hardware-Accelerated CPU Distance Kernels via SimSIMD (ARM NEON, AVX-512, VNNI) — 🔴 **High**
*Deliver hand-tuned SIMD vector distance execution on ARM64 and modern x86 without manual assembly maintenance.*

> **Current state & Performance Gap.** Vector distance calculations in `src/core/index/distance.rs` dominate HNSW graph traversal and point queries. Today, x86_64 uses manual 256-bit AVX2/FMA intrinsics, but:
> 1. **ARM64 (Apple Silicon, AWS Graviton 3/4, Windows ARM / Snapdragon X Elite)** currently lacks explicit SIMD intrinsics in `distance.rs` and falls back to scalar loop unrolling (`dot_product_portable`), missing out on 1.5×–2.5× throughput gains from hardware vector registers.
> 2. **AVX-512** on modern AMD Zen 4/5 and Intel Xeon/Core processors is unutilized, leaving 512-bit registers (16 floats/instruction) and 32 ZMM registers untapped.
> 3. **Quantized Vector Math**: CPU lacks hardware VNNI (`int8` dot products) and half-precision (`f16`/`bf16`) SIMD instructions, forcing software conversions.
>
> **Approach: Integrate Unum's `simsimd` crate.** `simsimd` provides production-hardened, zero-dependency C/assembly kernels with automatic runtime CPU dispatching across AVX2, AVX-512, AVX-512-FP16, ARM NEON, and ARM SVE. Integrating `simsimd` accelerates CPU-bound HNSW traversals across all platforms while eliminating ~400 lines of fragile `unsafe` architecture-specific intrinsics.

- [ ] **Integrate `simsimd` dependency into `benostreamdb` and `benostream-gpu-ann`** — 🟠 **High**:
  - Add `simsimd` (Apache-2.0, pure Rust bindings with compiled SIMD kernels) to workspace dependencies.
- [ ] **Wire runtime dispatch in `src/core/index/distance.rs`** — 🟠 **High**:
  - Replace manual `dot_product_avx2`, `l2_distance_squared_avx2`, and `dot_product_portable` with `simsimd::spatial` / `simsimd::binary` dispatches.
  - Automatically route L2, Inner Product (Dot), Cosine, Hamming, and Jaccard distances through hardware-detected kernels.
- [ ] **Native Quantized CPU Distance Acceleration (`i8` / `f16` / `bf16`)** — 🟠 **High**:
  - Expose hardware-accelerated integer dot products (`VNNI` / `ARM SDOT`) and `f16` distance kernels for TurboQuant / quantized vector scans on CPU.
- [ ] **Multi-Architecture Benchmarking** — 🟡 **Medium**:
  - Measure search latency improvements across Apple Silicon M-series (NEON), AWS Graviton 4 (SVE), AMD Zen 4/5 (AVX-512), and Windows on ARM.

### Theme 5: GPU Scaling (single-query multi-GPU + cross-call batching) — 🟠 **Medium**

> **Shipped in 0.12.0.** The multi-GPU device pool
> (`gpu::set_gpu_device_pool`, wired into Trino/Spark) round-robins devices
> across **worker threads**, and each kernel call is already batched (one
> H2D → launch → D2H over the whole vector set, with a 50k floor:
> `GPU_DISPATCH_THRESHOLD`).
>
> **Remaining.** The round-robin only spreads work across **concurrent**
> dispatches — a single large query still runs on one device
> (`HnswIvfIndex::search` makes one synchronous `compute_distance` call), and
> there is no overlap of transfer with compute across calls.

- [ ] **Single-query multi-GPU sharding** — shard the candidate set (e.g.
  `HnswIvfIndex::search`'s centroid / distance batch) across the devices in the
  pool, launch concurrently on each, and merge top-k. This is what turns the
  pool from a concurrency-only feature into a genuine per-query speedup.
- [ ] **Cross-call batching + streams** — overlap H2D / compute / D2H across
  successive batches with CUDA streams instead of one synchronous round trip
  per call; pipeline rather than block.
- [ ] **Avoid redundant transfers** — keep vectors/centroids device-resident and
  cache the uploaded query. Today each call re-uploads the query and copies the
  whole batch to host (`htod_copy(vectors.to_vec())`).
- [ ] **Adaptive dispatch threshold** — `GPU_DISPATCH_THRESHOLD` (50k) is a
  coarse floor; make it adaptive to the measured launch/transfer overhead so
  mid-size batches can still benefit without paying H2D/D2H for nothing.

### Theme 6: AI Agent Tools & Client Ecosystem
*Native interfaces for AI frameworks and autonomous agent runtimes.* — 🟠 **Medium**

- [ ] **LangChain & LlamaIndex Partner Integrations** — 🟢 **Deferred**:
  - `langchain-benostreamdb`: LangChain-compatible `VectorStore` and `Retriever` implementations with metadata filtering.
  - `llama-index-vector-stores-benostreamdb`: LlamaIndex vector store index adapter.
- [ ] **Broker Event Adapters** — 🟢 **Deferred**:
  - Pluggable Kafka, RabbitMQ, and NATS consumers implementing the `WorkCoordinator` trait for automated event-driven lakehouse ingestion.

### Theme 7: Graph Overlays
*Elevating knowledge graphs into a first-class lakehouse data modeling primitive.* — 🟠 **Medium**

> **Partially delivered.** The programmatic edge-table API has shipped:
> `Table.create_edge_table(...)`, `Table.from_networkx(...)`, and
> `Table.to_networkx(...)` create and round-trip standard edge tables with
> automatic endpoint/embedding sidecar indexes. The graph UDAFs
> (`graph_neighbors`, `graph_pagerank`, `graph_shortest_path`, …) are now
> reachable from every connector surface — including Spark, via the pass-through
> reader. Typed global URNs below remain open.

- [ ] **Typed Global Entity URNs (`table:id`)** — 🟢 **Deferred**:
  - Extend the CSR graph dictionary to support composite entity identifiers (e.g. `users:101`, `orders:5002`) so a single CSR graph overlay can seamlessly traverse across heterogeneous Iceberg tables.

### Theme 8: Semi-Structured & JSON
*First-class access to JSON documents without forcing a rigid column schema.* — 🟢 **Deferred**

> **Current state.** JSON is stored as a `Utf8` string column (text, not a
> decomposed binary form, so the functions use PostgreSQL's `json_*` names, not
> `jsonb_*`). The search gateway
> (`contrib/benostreamdb-search/src/infer.rs`) already infers an Arrow schema
> from sample documents (ES 7.10 dynamic mapping).

- [ ] **Iceberg v3 `variant` Mapping** — 🟢 **Deferred**:
  - Map semi-structured columns to Iceberg v3 `variant` (the spec-native
    semi-structured type) so JSON round-trips through catalogs instead of being
    a bespoke engine-only type; Arrow-side representation is `Struct`/`Utf8`
    with path pushdown. This is a storage-fidelity improvement — it still needs
    the JSON-path index above to make containment fast.

### Theme 9: Reactive Lakehouse Streaming
*Bringing change data capture (CDC) and reactive streaming to the lakehouse.* — 🟢 **Deferred**

- [ ] **Arrow Flight SQL Streaming Subscriptions** — 🟢 **Deferred**:
  - Implement Flight `DoExchange` / `DoGet` streaming endpoints allowing remote clients (Python, DuckDB, web dashboards) to subscribe to table diffs in real time without polling. Build this once somebody actually needs remote subscriptions.

### Theme 10: Domain-Specific Text Analyzers for BM25
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
**documentation/tutorials** that show the three workflows concretely. Both have
shipped — see **Key Completed Milestones**.

