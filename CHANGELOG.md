# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic
Versioning](https://semver.org/spec/v2.0.0.html).

## [0.12.0]

### Benchmarks
- **Report: "Index Precision" column** — the vector tables now report the
  quantization (`f32`/`tq8`/`tq4`/`pq`) rather than the algorithm name, and the
  `_tq8`/`_tq4`/`_pq` suffix is dropped from the Engine column (the precision
  column carries it). The §4 BEIR table is sorted by dataset then QPS, with
  lexical BM25 rows reading `-` (no vector index) and hybrid `f32`. The SQL
  harness now carries the dataset name (recovered from the filename for older
  records) and duplicate SQL runs are collapsed.
- **Competitor matrix expanded** — added **Qdrant**, **Milvus** and **Weaviate**
  (vector) and **Memgraph** (MAGE) and **Kùzu** (embedded) graph engines, all
  under the shared Docker envelope. The report gains a **§10 "Competitor
  Configurations"** reference table (index/engine setup + measurement layer) and
  **§2** now renders the graph matrix (incl. Neo4j + GDS) with a `layer` column.
- Re-ran the graph (SNAP + synthetic) and vector (8 ANN-Benchmarks datasets)
  matrices under the shared envelope after the fixes below.

### Fixed
- **Neo4j graph benchmark measured the driver, not the database** —
  `_graph_neo4j` timed `gds.*.stream` with `list(res)`, i.e. Python marshalling
  of 100k+ Bolt records (PageRank read as ~3.7 s on web-Google). It now runs
  native GDS `.mutate` in the Neo4j JVM (~0.1 s of compute) and returns a
  summary row; results carry a `layer` describing the measurement layer.
- **pgvector L2 queries did a full sequential scan** — the query used `<=>`
  (cosine) against a `vector_l2_ops` index, so Postgres could not use the HNSW
  index. Fixed to `<->` to match the opclass (QPS ~3x higher; recall now
  properly approximate). Raised `shared_buffers`/`maintenance_work_mem` and the
  container `shm_size` so the HNSW build is not throttled by the Postgres
  defaults or Docker's 64 MB `/dev/shm`.
- **hnswlib built dot-product datasets in L2 space** — now maps
  `inner_product` → `ip`; LanceDB maps it to `dot` and applies the `ef`
  (HNSW) / `nprobes` (IVF_PQ) recall knob, which it previously ignored.
- **DuckDB / DataFusion ignored the cgroup envelope** — DuckDB now pins
  `threads`/`memory_limit` and DataFusion `target_partitions` to
  `BENCH_CPUS`/`BENCH_MEM` (they otherwise used every host core).
- **Benchmark env over-reported host resources** — `env.cores`/`env.ram_gb` now
  read the cgroup CPU/memory limits. The containers were always capped at
  8 CPU / 16 GiB, but the *reported* values previously showed 32 CPUs / 121 GB.
- **Memgraph loader hung / pegged a core** — the un-indexed
  `MERGE (n:Node {id})` load was O(n²); a `:Node(id)` index is now created
  first (500k-edge load: minutes/hang → ~5 s). Kùzu's default 8 TB `max_db_size`
  and host-sized buffer pool also failed to mmap under the cgroup cap; both are
  now sized from the envelope.

### Added
- **Workflow tutorials** — one runnable tutorial per core workflow: AI/semantic
  retrieval via MCP, JSON/document retrieval with a JSON-path index, and graph
  traversal via `graph_neighbors(...)`. Python versions are Jupyter notebooks
  (`examples/tutorials/*.ipynb`, verified to execute end-to-end); the
  SQL/connector forms (Spark/Trino/dbt/Flight/MCP/Rust) are markdown pages under
  `docs/tutorials/`.
- **GPU device selection on every surface** — one shared core mapping
  (`gpu::context_from_device_str`, accepting `auto` | `cpu` | `cuda[:N]` |
  `mps`/`metal` | `intel`/`xpu` | `rocm`/`hip`) now backs the Spark/Trino JNI
  bridges, the Python binding (`benostreamdb.set_gpu_device` / `gpu_device`),
  the dbt adapter (profile `gpu_device`), and the Flight SQL + MCP servers
  (`BSDB_GPU_DEVICE` / `BENOSTREAM_GPU_DEVICE` at startup). Previously only
  Python/Spark/Trino exposed a device; Flight, dbt, and MCP had none.
- **Multi-GPU execution** — `gpu::set_gpu_device_pool` (+ `set_gpu_device_pool_from_str`)
  installs a round-robin pool of devices; each engine worker thread is assigned
  one the first time it asks for a context, so a single query spreads across
  GPUs. Wired into the Trino connector (`benostream.gpu-device` may be a
  comma-separated list; `BenoStreamDBJNIBridge.setGpuDevicePool`) and the Spark
  JNI bridge. A single device keeps the previous process-wide behaviour.
- **Reactive subscriptions (`Table::subscribe()`)** — a per-table
  `tokio::sync::broadcast` change feed that publishes committed `RecordBatch`es
  plus a `Commit` marker on every successful flush. `subscribe_filtered("age > 30")`
  re-applies a SQL predicate above each batch (DataFusion) and yields only
  matching rows (commit markers are suppressed for filtered subscriptions);
  `Subscription::close()` unsubscribes deterministically. Exposed on **every**
  surface: native Python (`Table.subscribe()` → `Subscription.recv`/`try_recv`/
  `close`, context-manager), the `subscribe_events` DataFusion table function
  (reachable from Python/dbt/Spark/Trino/Flight), a Flight SQL streaming ticket
  (`type.googleapis.com/benostreamdb.Subscribe`), and an MCP `subscribe_events`
  tool. The feed is in-process.
- **Declarative edge tables auto-configure their CSR overlays** — declaring
  `table_type = 'edge'` with `src_col`/`dst_col` (via `SET TBLPROPERTIES` or
  Python `set_property`/`set_properties`) now materialises the forward (source)
  and reverse (target) CSR graph indexes through
  `Table::ensure_edge_indexes_async` (idempotent; no-op for non-edge tables).
- **Spark stored procedures are now real (not stubs)** — the
  `CALL benostream.system.*` surface routes end-to-end into the engine over JNI:
  `add_index` (with an `algorithm` argument: `hnsw`, `hnsw_pq`, `hnsw_tq4`,
  `hnsw_tq8`, `bm25`, `bloom`, `bitmap`, `composite_bitmap`, `csr_graph`,
  `json_path`), `drop_index`, `build_index`, `rebuild_index`, `set_primary_key`,
  the new **`drop_primary_key`**, `show_indexes` (real JSON from the manifest),
  and `compact`. Previously each native entry point only logged and returned a
  sentinel, so index/PK management silently did nothing.
- **`IndexAlgorithm::from_name`** — parses a user-supplied algorithm name or
  physical category (`vector`, `lexical`, `scalar`, `graph_v2`, `bloom`) into an
  `IndexAlgorithm`; the single string→algorithm mapping shared by the connectors.
- **`Table::list_index_files` / `Table::rebuild_index`** — engine helpers backing
  the connector `show_indexes` and `rebuild_index` procedures.
- **dbt macro coverage for the whole engine function surface** — new/expanded
  macros in `dbt-benostreamdb`: vector distances (`l2_distance`,
  `cosine_distance`, `inner_product`, `l1_distance`, `hamming_distance`,
  `jaccard_distance`), transforms (`vector_add/sub/mul/concat`, `vector_dims`,
  `vector_norm`, `l2_normalize`, `binary_quantize`, `subvector`,
  `vector_to_binary`), aggregates (`vector_sum`, `centroid`, `vector_median`,
  `vector_stddev`, `vector_min`, `vector_max`), sparse (`sparse_to_vector`,
  `vector_to_sparse`), lexical search (`bm25_score`, `tf_idf`,
  `keyword_search`), JSON (`json_extract_path`, `json_extract_path_text`,
  `json_contains`, `json_exists`, `json_typeof`, `json_path_exists`,
  `json_path_query`), and graph (`triangle_count`, `modularity`,
  `closeness_centrality`, `betweenness_centrality`, `all_shortest_paths`,
  `leiden` via `community_detect`, `drift_search`, `regional_drift`). The
  broken `topological_sort` macro (the engine registers no such function) was
  removed. Verified by new dbt models (`test_all_functions`,
  `test_vector_aggregates`, `test_search_functions`, `test_json_functions`,
  `test_all_graph_functions`) — 8/10 models green, including all 20 graph
  macros.
- **`CREATE TABLE AS SELECT` in the engine** — `create_table` now derives the
  target schema from the source query, creates the table, and populates it via
  `INSERT INTO ... SELECT` (executed through the DataFusion context to keep the
  async call graph acyclic). The query text is sliced from the original
  statement (`extract_ctas_query`) because sqlparser's AST re-render mangles
  parenthesized `WITH` queries — the form dbt emits. Covered by 5 unit tests.
- **Unified default schema (`default`)** — the engine's DDL default
  (`catalog_ddl::DEFAULT_SCHEMA`), DataFusion's unqualified-name search path
  (`with_default_catalog_and_schema`), Spark's V2 default namespace, and the
  dbt adapter's profile default now all agree on `default` (lakehouse
  convention), so `CREATE TABLE t` and `INSERT INTO t` resolve identically
  across every connector surface.
- **`Session(warehouse=...)` Python binding** — the embedded session accepts a
  warehouse base location (as the Flight server does via `BSDB_WAREHOUSE`), so
  DDL/CTAS can derive table URIs; the dbt adapter passes the profile `path`
  through (`:memory:` maps to the engine's in-memory store).
- **Native Spark connector (v0.12.0)** — the Spark DataSource V2 connector is
  now fully native: catalog/DDL (`BenoStreamCatalog` via JNI
  `listSchemas`/`listTables`/`createTable`/`dropTable`), reads (engine SQL via
  `openQuery` + Arrow C Data Interface with filter/projection pushdown), writes
  (`appendBatch`), and **row-level operations** — `DELETE` / `UPDATE` / full
  `MERGE` matrix through `SupportsDelta` using primary-key row identity
  (matched deletes become engine predicate deletes; inserts/updated rows are
  appended as Arrow batches). Verified on Spark 3.5.9 and 4.2.0 with
  whole-stage codegen enabled. Spark 4.x builds carry **no
  `iceberg-spark-runtime` dependency**; the 3.5 profile links it `provided`
  only for `CALL` support (Spark 3.5 lacks the native DSv2 procedure API). One
  Scala 2.12 artifact serves Spark 3.5.x and one Scala 2.13 artifact serves
  Spark 4.0/4.1/4.2 (`DeltaWriter`/`SupportsDelta` APIs are identical). On
  Spark 4.x, `BenoStreamProcedureCatalog` extends the native catalog so tables,
  namespaces, and `system.*` procedures share a single catalog.
- **Spark catalog functions (`FunctionCatalog`)** —
  `benostream.system.cosine_distance` / `l2_distance` / `dot_product` /
  `vector_distance` / `sparse_dot_product` / `hybrid_score` /
  `reciprocal_rank_fusion` resolve natively from SQL, DataFrames, and PySpark
  with zero session registration (works on Spark 3.4+ including 3.5 and 4.x).
  `SHOW FUNCTIONS IN benostream.system` lists them; `BenoStreamFunctions.register`
  remains for unqualified session names. pgvector operators are intentionally
  not exposed in Spark SQL (grammar rejects them; use the function spelling,
  which matches the Trino/DataFusion names).
- **Search gateway (`contrib/benostreamdb-search`)** — a standalone,
  embeddable HTTP service exposing Elasticsearch/OpenSearch-compatible and
  Qdrant-compatible REST APIs over BenoStreamDB tables. Covers document
  get/delete, bulk ingest, phrase/multi-match search, custom sort, index
  aliases, nested queries, `search_after` pagination, highlighting, regexp
  queries, snapshots, payload indexing, `best_score` recommendation, and
  prefetch RRF fusion. Position-aware inverted indexes back phrase search, and
  sparse-vector + hybrid (BM25 + kNN) collections are supported.
- **Stateless authentication & RBAC** — API-key and JWT authentication for the
  search gateway, Qdrant, and Flight SQL (`src/core/auth.rs`), with role-based
  access control.
- **Arrow Flight SQL server (`server/flight_sql`)** — an optional Flight SQL
  endpoint over the engine's DataFusion session, with DDL/DML executed inside
  `GetFlightInfo` (ADBC cancels the follow-up `DoGet` for DML) and catalog /
  warehouse startup configuration.
- **OTLP telemetry export** — `src/telemetry/otlp.rs` adds OpenTelemetry
  protocol metrics export.
- **dbt adapter standalone PyPI distribution** — `dbt-benostreamdb` ships as a
  standalone package with its own `pyproject.toml`.
- **PostgreSQL `json` path functions** — `json_extract_path` /
  `json_extract_path_text` (variadic path), `json_contains` (recursive `@>`),
  `json_exists`, `json_typeof`, and `json_path_exists` / `json_path_query`
  (jsonpath subset: `$.a.b[0]`, `[*]` wildcards). JSON is stored as a `Utf8`
  string column, so the functions use PostgreSQL's `json_*` names (not
  `jsonb_*`, which implies a decomposed binary representation). DataFusion 52
  ships no JSON module, so these are registered as scalar UDFs in
  `src/core/sql/udf/json.rs`.
- **JSON-path inverted index overlay** — `IndexAlgorithm::JsonPath { paths }`
  builds a `(path, value) -> row_ids` Parquet overlay for selected JSON paths in
  a `Utf8` column (`src/core/index/json_path.rs`), packed into the segment's
  Puffin bundle under the `json_path` category. The planner rewrites
  `json_contains`, `json_exists`, `json_path_exists`, and
  `json_extract_path_text(...) = 'value'` predicates into index lookups
  (`HybridReader::query_json_path_first`), falling back to a full scan for
  unindexed paths. The index records a presence marker and one level of array
  elements so every lookup is a *superset* of the true matches — the filter is
  re-applied above the scan, preserving the Overlay Invariant. Verified by a
  differential oracle against a full scan (`tests/test_json_path_index.rs`).
- **Detached Overlay Catalog** —
  `Table.mount_external_iceberg(table_uri, overlay_storage_uri)` mounts an
  external, read-only Apache Iceberg table and writes its derived overlay
  indexes into a separate writable prefix, without any write access to the
  upstream lake or catalog. A persisted `detached_catalog.json` sidecar pins the
  upstream snapshot ID, and `Table.sync_detached_overlay()` /
  `Table.start_detached_sync_worker(interval)` incrementally reconcile
  out-of-band upstream appends and compaction. Correctness is preserved by the
  checksum lineage contract: stale indexes are rejected in favour of a Parquet
  scan. Implemented in `src/core/table/detached.rs` with Python bindings and
  integration tests.
- **BenoStreamDB Native DRIFT Search (Global & Regional)** — Ported Microsoft
  GraphRAG's DRIFT (Dynamic Reasoning and Inference with Flexible Traversal)
  algorithm onto native graph primitives. Exposes `Table.drift_search(...)` and
  `Table.regional_drift_search(...)` in Python, corresponding Rust core methods,
  and `drift_search` / `regional_drift` SQL UDAFs. Both support an adaptive
  `GraphMode` (`auto` / `in_memory` / `out_of_core` / `cached`) that dynamically
  selects between in-memory structures and mmap-backed CSRs depending on the
  available memory budget. Includes vector-based community search, early PPR
  convergence checks, score decay, and returns full search traces with PPR
  scores and community assignments.
- **First-class graph traversal primitives in the core engine** —
  `src/core/table/graph.rs` exposes core Rust APIs (`load_graph_index`,
  `shortest_path`, `connecting_paths`, `graph_neighbors`, `subgraph_edges`)
  alongside robust Python result containers (`PathResult`, etc.) that directly
  integrate with `.to_pandas()` and `.to_arrow()`. All graph operations unify
  under a shared `GraphView` trait for memory-adaptive multi-modal execution.
- **Graph RAG vector re-ranking & Neo4j-style global search** —
  `graph_rag_search(..., rerank_vector=True, rerank_k=N)` re-ranks local-mode
  candidate node sets using restricted vector searches.
  `graph_rag_search(mode="global")` now selects communities via vector search
  over community report embeddings (HNSW) and recursively descends the hierarchy
  to leaves, falling back to heuristics when embeddings are missing.
- **New Python `Table` helpers** — `find_entities(...)` (full-text index with
  ILIKE fallback), and `lookup_entities(...)` (bulk id → name resolution).
- **`commit_synced_snapshot`** — commits an externally synchronized Iceberg
  snapshot with full reconciliation against the current manifest (retries on
  conflict, preserves delete files).
- **`docs/architecture_review_response.md`** — evaluation and remediation plan
  for the concurrency review (H1–H3) and the repository restructuring.
- **Full SQL DDL / maintenance surface over the core `Table` API** — a new
  interception layer (`src/core/sql/catalog_ddl.rs`) parses with
  `GenericDialect`
  and dispatches to the core `Table` API before DataFusion planning (mirroring
  `merge_into`), wired into `session.sql_to_df` / `is_ddl` / `get_schema`.
  Reachable from Rust, Python, and the Flight SQL gateway:
  - **Catalog**: `CREATE/DROP DATABASE`, `CREATE/DROP SCHEMA` (DataFusion +
    external catalog via the new `Catalog::create_namespace` / `drop_table`).
  - **Tables**: `CREATE TABLE` (catalog-backed, `LOCATION`, `WITH` options),
    `DROP TABLE` (drops from the catalog too), `TRUNCATE`.
  - **Indexes**: `CREATE INDEX` (single/composite, default Bitmap),
    `ALTER TABLE ADD INDEX`, `DROP INDEX`.
  - **Primary keys**: `ALTER TABLE ADD/DROP PRIMARY KEY`.
  - **Schema evolution**: `ADD/DROP/RENAME/ALTER COLUMN`.
  - **Maintenance**: `OPTIMIZE`/`COMPACT`, `VACUUM`, `MSCK REPAIR TABLE`,
    `DELETE FROM`, and `ALTER TABLE ... EXECUTE <action>` procedures.
  - **Session settings**: `SET/SHOW benostream.<key>` — allowlisted,
    session-scoped, and never writes process environment variables.
- **Vector companion aggregates** — `centroid`, `vector_min`, `vector_max`,
  `vector_stddev`, `vector_median` (`src/core/sql/udf/vector_stats.rs`), the
  natural companions to `vector_sum` / `vector_avg`.
- **Text-scoring UDFs** — `bm25_score(text, query)` and `tf_idf(text)`
  (`src/core/sql/udf/text.rs`), corpus-free variants for ranking rows against a
  single query.
- **Python bindings for the new aggregates** — `centroid`, `vector_min`,
  `vector_max`, `vector_stddev`, `vector_median`, `bm25_score`, `tf_idf` in
  `python/benostreamdb/__init__.py`, keeping the Rust, Python, and SQL surfaces
  1:1.
- **`plans/sql_ddl_surface.md`** — design spec for the SQL surface, and a full
  SQL language guide (all statements, UDFs, and pgvector syntax) in
  `server/flight_sql/README.md`.
- **`Table::reindex_inverted_column`** — a *targeted* reindex that rebuilds only
  a column's lexical index blob in place, upgrading legacy 2-column
  (position-less) inverted indexes to the 3-column position-aware format without
  touching the segment's other indexes (notably the HNSW vector indexes). The
  analyzer recorded in the existing blob is preserved, so query semantics are
  unchanged — only the O(1) term-frequency path is enabled. Driven by the new
  `reindex_inverted` binary.
- **`relocate_table` binary** — rewrites a moved table's absolute paths to its
  new location across all three places Iceberg stores them: the table metadata
  JSON (`location`, `snapshots[].manifest-list`, `metadata-log`), the
  BenoStream manifest entries, and the Avro manifest list / manifest files. An
  optional `--catalog-type` step repoints a catalog's metadata-location via a
  `set-metadata-location` commit.
- **Trino connector: real metadata + SQL pushdown over JNI** — the connector no
  longer hardcodes `listSchemaNames`/`listTables` or silently returns mock data
  on `UnsatisfiedLinkError`. New JNI entry points (`listSchemas`, `listTables`,
  `openQuery`/`readQueryBatch`/`closeQuery`) let the connector enumerate the
  warehouse and run each scan through the engine's DataFusion session, so the
  pushed-down predicate travels as a SQL `WHERE` clause and the planner applies
  the full index/vector-search pushdown. The split is now `(tableUri, sql)`
  instead of a file range, and the page source streams the query result.
- **`jni_bridge` fuzz target** — fuzzes the JNI bridge's untrusted-input
  parsing (the `[{name,type,nullable}]` schema JSON and the Arrow type-name
  mapping), shared by the Trino and Spark connectors. The pure-Rust helpers
  moved to `core::jni_util` so they can be fuzzed without a JVM. The
  `benostreamdb-search` dependency is now optional in the fuzz workspace (only
  the Qdrant target needs it), so the other targets build independently.
- **Spark catalog functions cover the whole engine surface** — the DSv2
  `FunctionCatalog` now exposes 30 functions (distances incl. `l1`/`hamming`/
  `jaccard` and the `inner_product` alias, transforms, `bm25_score`/`tf_idf`,
  and the seven `json_*` functions) under `benostream.system.*`, and a new
  pass-through reader (`BenoStreamQueryTable`, `option("query", "<engine SQL>")`)
  makes the vector aggregates and graph UDAFs — which have no DSv2 scalar
  equivalent — reachable from Spark. The engine's ad-hoc `Table::sql()` context
  now registers the full custom function surface (vector scalar UDFs, JSON,
  vector aggregates, graph UDAFs) via a single shared helper
  (`udf::register_all_custom_udfs`), so pass-through queries no longer fail with
  "Invalid function".
- **Cross-surface function-parity test** — `tests/python/test_function_parity.py`
  treats `udf::registered_function_names()` as the source of truth and asserts
  every core function is exposed (or explicitly documented as a gap) across the
  Python session, dbt macros, Spark catalog/pass-through, and Trino. Adding a
  core UDF without wiring a surface now fails the test.
- **SQL graph traversal table functions** — `graph_neighbors`,
  `graph_shortest_path`, `graph_all_shortest_paths`, `graph_subgraph`, and
  `graph_connecting_paths` are now DataFusion table functions, so a graph walk
  is a `FROM` source (`SELECT * FROM graph_neighbors('edges', '101', 2, 'auto')`)
  rather than an aggregate. Implemented in
  `src/core/sql/graph_udf/graph_table_functions.rs`; each follows the same
  in-memory / out-of-core `GraphMode` pattern as the graph UDAFs and accepts
  optional trailing `source`/`target` column names (auto-detected otherwise).
  Registered in the engine session, the ad-hoc `Table::sql()` context, and the
  Python `execute_sql` path, so they are reachable from Python, dbt, Spark
  (pass-through), Trino, and Flight SQL. The parity test now also covers the
  Flight SQL surface.
- **Surface parity beyond functions** — `tests/python/test_surface_parity.py`
  asserts the engine's **index algorithms** (`IndexAlgorithm::all_names()`),
  **DDL statements** (`catalog_ddl::handled_ddl_statements()`), and **table
  actions** (`catalog_ddl::table_action_names()`) are consistent across Python,
  dbt, Spark, Trino, and Flight SQL. New `registered_index_algorithms()` /
  `registered_ddl_statements()` / `registered_table_actions()` bindings expose
  the source-of-truth sets. The Python index parser now accepts every engine
  algorithm (`composite_bitmap`, `json_path`, and `csr_graph` as a string were
  missing).
- **dbt DDL + graph table-function macros** — `create_index` / `drop_index`
  (`macros/ddl.sql`) expose the index lifecycle, and `graph_neighbors_table` /
  `graph_shortest_path_table` / `graph_all_shortest_paths_table` /
  `graph_subgraph_table` / `graph_connecting_paths_table` (`macros/graph.sql`)
  emit the `FROM graph_*(...)` table functions, so a graph walk is a dbt model
  source. Verified by a new `test_surface_macros` model.
- **Table-type metadata (declarative edge/node tables)** — a table can declare
  its role and endpoints in its properties: `table_type` (`node` | `edge` |
  `table`), `src_col`, `dst_col`, `relation_col`, `weight_col`, `id_col`,
  `label_col`. Set with `ALTER TABLE t SET TBLPROPERTIES (...)` or Python
  `Table.set_property` / `set_properties` (backed by `Manifest.properties`,
  metadata-only commits). `create_edge_table` / `create_node_table` stamp the
  convention; the graph functions resolve endpoints **metadata-first** (then the
  standard name candidates); and `Session.list_graph_tables()` enumerates the
  node/edge tables — the discovery primitive an MCP agent uses. New
  `Table.table_type()` / `graph_metadata()` / `edge_endpoints()` (metadata-first)
  bindings.

### Changed
- **Puffin compound bundles are the index storage format.** Every secondary
  index — lexical (BM25 + doc-length), scalar inverted
  (Int32/Int64/Float/Date/Bool), vector (HNSW-IVF centroids/graph/mapping), and
  graph (CSR offsets/edges/dict) — is packed into a single `{segment}.puffin`
  container per segment, reducing object counts by 75%+. Readers load vector
  indexes via `HnswIvfIndex::load_from_puffin`, graph indexes via
  `MmapCsrGraph::from_bytes`, and lexical/scalar indexes via byte-range reads.
- **Query parallelism scales to the CPU budget.** DataFusion `target_partitions`
  and the SQL scan partition count now derive from
  `effective_target_partitions()` (the process-visible CPU count, respecting
  cgroup limits) instead of a hard-coded `4` / the framework default.
  `BSDB_TARGET_PARTITIONS` overrides for tuning.
- **Community detection now defaults to Leiden** — `communities`,
  `update_communities`, `graph_rag_search(community_algorithm=...)`,
  `summarize_communities(algorithm=...)`, and the Rust `communities_csr*`
  bindings default to `"leiden"` (connected communities) instead of `"louvain"`.
- **Repository restructured into a three-tier layout.** The core workspace is
  now strictly the database engine plus first-class data-platform connectors,
  with experimental integrations moved out of the root package:
  - `benostreamdb-search` → `contrib/benostreamdb-search` (optional HTTP search
    gateway; `contrib/*` is excluded from the root Cargo package).
  - `benostreamdb-flight` → `server/flight_sql` (optional Arrow Flight SQL
    server).
  - `trino-config/` → `trino-benostreamdb/etc/`; new `spark-benostreamdb/` and
    per-integration READMEs (`dbt-benostreamdb`, `spark-benostreamdb`,
    `trino-benostreamdb`).
  - The root Cargo workspace now contains only the core engine,
    `benostream-gpu-ann`, `server/flight_sql`, and
    `contrib/benostreamdb-search`.
  - CI trimmed to test the core engine and the Tier-2 integrations (Trino,
    Spark,
    dbt).
- README documents three deployment modes (embedded core, optional Arrow Flight
  SQL server, future distributed) and clarifies that Flight SQL is
  single-process.
- ROADMAP condensed from ~811 to ~100 lines, removing stale historical
  benchmarks.
- **Flight SQL executes DDL and DML inside `GetFlightInfo`** — ADBC cancels the
  follow-up `DoGet` for DML, so the statement would otherwise never run.
- **`BenoStreamTableProvider::insert_into`** — `INSERT INTO` now writes and
  commits the rows, so they are immediately visible to subsequent scans.
- **Flight SQL startup loads `CatalogConfig` and `BSDB_WAREHOUSE`** so
  `CREATE TABLE` lands in the instance's catalog.
- **`Table::checkpoint_async`** — an async WAL checkpoint safe to call from
  within a Tokio runtime (the sync `checkpoint` uses `blocking_lock`).
- **`Catalog` trait** gains `create_namespace` and `drop_table` (default no-ops;
  implemented for REST, JDBC, Glue, Hive, and Unity).

### Security
- **SQL injection fix (A03)** — the DDL/maintenance interception layer no longer
  interpolates untrusted identifiers into SQL; statements are parsed and
  dispatched through the core `Table` API.
- **SSRF guard (A10)** — `BSDB_SSRF_GUARD` blocks requests to link-local,
  private, and non-public endpoints, including Azure/GCP metadata endpoints and
  IP-literal hosts.
- **Dependency-advisory tracking plan (A06)** — documented in
  `docs/DEPENDENCY_RISK.md`.
- **Stateless API-key + JWT auth** — the search gateway, Qdrant, and Flight SQL
  reject unauthenticated requests by default (see `src/core/auth.rs`).

### Fixed
- **Inner-product vector search returned the *least* similar results** — the IVF
  coarse search computes cluster distances via `gpu::compute_distance`, whose
  CPU fallback returned `+dot` for `InnerProduct` while the explicit fallback and
  the flat scan use `-dot` (smaller = more similar). A query was therefore routed
  to the *least* similar clusters. Fixed in `compute_cpu` and in the CUDA, Metal,
  and WGSL `inner_product_kernel`s (all now negate). On `lastfm-64-dot`
  BenoStreamDB's recall@10 went from **0.065 → 0.712**.
- **`DistDot` did not normalize** — the HNSW inner-product distance was
  `(1 - dot).max(0)`, which is not a metric, so the hnswlib-style
  neighbour-selection heuristic built a graph dominated by high-norm hubs and
  recall collapsed on unnormalized vectors. It now computes `1 - cosine`
  (inner product on L2-normalized vectors), which is what the `hnsw_rs` crate's
  `DistDot` expects.
- **`DistL2` scalar fallbacks returned the squared distance** — the `f64`/`i32`/
  `u32`/`u16`/`u8` impls omitted the `.sqrt()` the `f32` impl has, so the
  reported distance was inconsistent across element types.
- **IVF centroid routing used L2 for the Cosine metric** — `find_closest_centroid`
  mapped `L2 | Cosine => l2_distance_squared`, but the engine does not
  L2-normalize on insert, so cosine queries were routed by L2 while the HNSW
  graph used true cosine. Cosine now routes by `cosine_distance`.
- **The search gateway treated the ES `_count` endpoint as a write** — the
  read-shaped-POST allowlist matched `/count` but not `_count`, so a read
  required the `admin` role. Extracted into a tested `is_write_request` helper.
- **ClickHouse could not load nullable Parquet columns** — the SQL adapter built
  non-nullable ClickHouse types from the Parquet schema, so a nullable column
  (e.g. NYC TLC's `passenger_count`) failed with "Unable to create native array".
  Nullable Parquet fields now map to `Nullable(...)`.
- **Quickstart container had an unsafe default network posture** — the search
  images defaulted `BSDB_QDRANT_BIND=0.0.0.0` and the quickstart compose
  published `9200`/`6333`/`50051` on all host interfaces with no credentials
  (the server fails closed, so it would exit rather than serve). The image
  defaults are now loopback, the quickstart publishes on `127.0.0.1` only, and
  it sets a local dev `BSDB_API_KEY`.
- **`CREATE INDEX ... USING <algorithm>` silently ignored most algorithms** —
  the engine's parser accepts `CREATE INDEX` but **drops the `USING` clause**
  (`ci.using` is always `None`), and the mapped set was only `hnsw*`/`bm25`, so
  every other algorithm (including `json_path`, `hnsw_pq`, `bloom`,
  `csr_graph`, `composite_bitmap`) became a scalar Bitmap. The algorithm is now
  read from the raw SQL via the shared `IndexAlgorithm::from_name`, so every
  engine algorithm is reachable from DDL. A trailing `WITH (paths = '$.a,$.b')`
  (which the generic dialect also rejects) is stripped and hand-parsed to supply
  the JSON paths a `json_path` index needs.
- **Flaky crash at interpreter exit on GPU workloads** — a background index
  build (which may use the GPU) could still be running on a tokio worker when
  the runtime was torn down, faulting the worker. `Table` background-task queues
  are now registered globally and **drained at exit** (bounded, via the Python
  `atexit` handler `benostreamdb.shutdown_gpu`), and hardware GPU backends are
  leaked on drop so cudarc never releases a CUDA resource after the driver has
  deinitialised. Repeated GPU exit runs went from ~2-4/10 crashes to 0/12.
- **Spark connector did not compile on Spark 4.0/4.1** — the shared
  `spark-4` catalog marked `listProcedures` as `override`, but that member only
  exists in Spark 4.2 (it is absent in 4.0/4.1 and abstract from 4.2). Dropping
  the `override` modifier (legal when implementing an abstract member and a
  plain new method on 4.0/4.1) makes one source root build against every 4.x
  minor. Verified with the offline `mvn test` matrix on 3.5/4.0/4.2.
- **Spark connector did not compile on Spark 3.5 (Scala 2.12)** —
  `BenoStreamCatalogFunctions` used `view.mapValues`, which does not exist on
  Scala 2.12's `IterableView`; replaced with a plain `map`.
- **Dead Spark JNI stubs removed** — `queryIndexIn` (always returned `{}`) and
  `commitPositionDeletes` (a no-op) had no callers: the connector prunes with the
  SQL that `openQuery` plans and commits deletes as predicate removals. Both the
  native functions and their `@native` declarations are gone, and the Spark docs
  now describe the real pushdown path.
- **Compiler/clippy warnings cleared** — an unreachable `java.math.BigDecimal`
  arm in the Spark filter literal builder (already covered by `java.lang.Number`),
  plus `clippy::useless_vec`, `clippy::manual_is_multiple_of`, and
  `clippy::large_enum_variant` in `bench_throttling` / `iceberg_rest`.
- **Graph table functions assumed the default catalog** — `resolve_table` looked
  up an unqualified / 2-part name only in the session's default catalog, so a
  table registered under a different catalog (e.g. the dbt adapter's `database`)
  was reported "not found". It now falls back to searching every catalog/schema
  for the table name.
- **Graph functions assumed `source`/`target` column names** — the Python
  `Table.pagerank()` / `personalized_pagerank()` / `shortest_path()` wrappers
  hard-coded `source`/`target` in their generated SQL, so an edge table using
  `src`/`dst` (or any other convention) failed. They now auto-detect the
  endpoint columns via `edge_endpoint_columns()`. The core
  `Table::graph_neighbors` / `shortest_path` / `connecting_paths` gained
  `_with_columns` variants (and `subgraph_edges_with_columns`), and the Python
  `core_*` bindings expose optional `source_column`/`target_column`, so every
  graph entry point can name its endpoints explicitly.
- **Ingest RAM back-pressure could block a writer forever** — the high-water
  check compared *absolute* process RSS against `BSDB_MAX_INGEST_RAM_GB`, so once
  the process baseline (a prior test in the same process, caches, the runtime
  itself) was already at/above the cap and no background task could reclaim it,
  `write` spun indefinitely. The check is now baseline-relative (it bounds the
  engine's *growth* since the write began) with a bounded fail-open grace
  (`BSDB_INGEST_BACKPRESSURE_GRACE_SECS`, default 60s) that logs an error and
  proceeds when RSS is stuck above the cap for reasons the write cannot relieve.
  The soak suite's memory-backpressure test now completes instead of hanging.
- **`graph_betweenness_centrality` panicked** — the result builder indexed the
  score map with `cb[&node]` for every node, but only intermediate vertices
  have an entry, so any graph with a non-intermediate node aborted with
  "no entry found for key" (a NO_PANIC_POLICY violation). Missing nodes now
  score `0.0`.
- **`vector_to_binary` / `vector_to_sparse` rejected `List` inputs** — both
  required `FixedSizeList`, so a `::FLOAT[]` cast failed ("could not cast array
  of type List(Float32) to FixedSizeListArray" / "Expected FixedSizeListArray").
  They now accept both layouts; `sparse_to_vector(vector_to_sparse(v))`
  round-trips correctly.
- **Vector aggregate UDFs over table columns** — `vector_sum`, `vector_avg`,
  `centroid`, `vector_median`, `vector_stddev`, `vector_min`, `vector_max` all
  failed on real tables: the `count` state field was declared as
  `List(Float32)` while the accumulator emits `UInt64`, and scan batches
  carried the table schema's nested Iceberg metadata (`iceberg.id`) while the
  plan's schema is metadata-free, so DataFusion's aggregate state coalescing
  panicked inside arrow's `coalesce` kernel. The scan now rebuilds batches
  against the plan schema (casting away nested field metadata), and the count
  state field is `UInt64`. All seven aggregates verified with correct values.
- **Vector UDF signature/return-type bugs** — `vector_add`/`vector_sub`/
  `vector_mul`/`vector_concat`/`subvector` declared scalar `Float32` arguments
  (rejecting every list-typed call) and `l2_normalize` plus the element-wise
  ops declared the input type while producing `FixedSizeList(Float32)`, tripping
  DataFusion's result-type assertion. Signatures now accept list arguments and
  the implementations emit `List(Float32)`.
- **dbt adapter: graph macros were never executable** — `subgraph`,
  `personalized_pagerank`, and `connecting_paths` used
  `map('format', "arrow_cast(%s, ...)")`, which Jinja applies in the wrong
  direction (the item becomes the format string), raising "not all arguments
  converted during string formatting". Replaced with a shared
  `_bsdb_arrow_array` helper. `pagerank` also omitted the `UInt64` casts on
  source/target that every other graph macro applies, so it failed with
  "Expected UInt64Array for sources". The CI test had hardcoded the rendered
  SQL, which is why neither bug was caught.
- **dbt adapter: SQL guard vs. dbt-issued scripts** — the engine's
  `sanitize_sql` rejects `;`/`--` anywhere; the adapter now strips `--` line
  comments (quote-aware) and a single trailing `;` before execution while still
  rejecting embedded semicolons (multi-statement protection preserved).
- **dbt adapter: drop-then-create cycle** — the embedded session starts with an
  empty relation cache, so `load_cached_relation` never saw warehouse-resident
  tables and the drop was skipped, leaving `Table already exists`. The `table`
  materialization now always issues an idempotent `drop table if exists`.
- **Engine DDL: quoted identifiers and `DROP TABLE IF EXISTS`** — `split_three`
  / `split_db_schema` now strip identifier quoting (dbt renders fully-quoted
  names), which previously leaked `"` into warehouse paths and broke catalog
  resolution; `DROP TABLE IF EXISTS` tolerates an unresolvable catalog path and
  deletes the underlying data (mirroring the FFI `dropTable`), so re-created
  tables start clean.
- **`SHOW TABLES` listed the `.keep` namespace placeholder** — `list_subdirs`
  (`src/core/ffi.rs`) now skips dot-prefixed entries, so the `CREATE SCHEMA`
  `.keep` marker no longer surfaces as a table in Trino/Spark `SHOW TABLES`
  (covered by `list_subdirs_tests::hidden_keep_marker_is_not_listed`).
- **H1 — write/WAL atomicity**: `write_buffer` and `pending_wal_tx_ids` unified
  into a single `pending_writes: Arc<RwLock<Vec<PendingWrite>>>` so a batch and
  its WAL transaction id are always taken together at flush time, structurally
  guaranteeing idempotent WAL replay.
- **H2 — `truncate_async()` race**: added a table-wide `maintenance_lock`
  (`tokio::sync::RwLock`); writes take a read lock while `truncate`,
  `rewrite_data_files`/compaction, `vacuum`, `delete`, `add_index`/`drop_index`,
  `update_schema`, `rollback_to_snapshot`, and `remove_orphan_files` take a
  write
  lock, preventing concurrent writes from being silently discarded.
- **H3 — destructive truncate error suppression**: `truncate_async()` no longer
  substitutes an empty manifest on load failure; the error is propagated.
  `flush_async` likewise propagates manifest-load errors instead of
  `unwrap_or_default()`.
- **Vacuum fail-closed**: `vacuum()` aborts (rather than deleting every data
  file) when a manifest version or its entries cannot be read, including during
  re-validation.
- **Orphan cleanup preserves delete files**: `remove_orphan_files` now treats
  position/equality delete files referenced by the manifest as live, so they are
  no longer reaped.
- **`ManifestManager::load_all_entries` relativizes inline manifest entries**:
  entries stored directly in the manifest JSON (e.g. after a `relocate_table`)
  are now normalized against the table URI, matching the Avro-entry handling.
  Without this, an absolute inline `file_path` reached the writer, whose
  `finish_indexing` derived a bogus nested `file:` output path.
- **Equality-delete read failures propagate**: `HybridReader` now returns an
  error instead of warning-and-continuing when an equality delete file cannot be
  read.
- **Lock robustness**: an unparseable lock payload is now treated as expired
  (with a warning) instead of being silently ignored.
- **Graph UDAFs are mode-invariant again**: DataFusion fans an aggregate out
  over `target_partitions` input partitions, and every empty partition emits a
  partial state whose scalar arguments are the accumulator defaults. The graph
  UDAFs adopted those defaults unconditionally in `merge_batch`, so a late
  empty partial could clobber the real arguments (e.g. `directed=true` reset to
  `false`, or `node2=3` reset to `0`) and the result depended on the
  non-deterministic merge order. `GraphAccumulatorBase::state_has_edges` now
  lets each UDAF skip scalar-argument adoption for empty partials.
- **`SubgraphView`/`CachingGraph`/`MmapCsrGraph` report `all_nodes`/`num_edges`**:
  the CSR-backed graph views previously fell back to the trait defaults
  (empty/zero), so global algorithms (`triangle_count`, `degree_centrality`,
  `pagerank`, `connected_components`, …) returned different answers in
  `out_of_core`/`cached`/`auto` than in `in_memory`. They now mirror
  `SimpleGraph`'s source-node semantics.
- **`CachingGraph::get_neighbors_into` caches only the appended slice**: it
  previously cached the caller's whole append buffer, so a reused scratch
  buffer could poison later cache hits.
- **`graph_preferential_attachment` uses one degree definition**: the
  retained-edges and resolved-graph branches now both count undirected degree.
- **Sparse row hydration no longer reads the whole Parquet file**:
  `fetch_rows_with_distances` read the entire file (up to 500 MB) into
  `BLOCK_CACHE` to return a handful of rows. A kNN/BM25 query hydrating 10 of
  100k rows now uses Parquet row selection instead, so it reads only the needed
  rows. This is the dominant cost of a gateway-style vector query (the search
  gateway's kNN path hydrates the top-k payloads). Small files (< 64 MB) are
  still cached whole: row selection reads whole row groups anyway, so caching a
  small file is strictly cheaper than re-reading it per query (this restores the
  ~10x kNN throughput regression the sparse-only guard introduced on small
  tables).
- **Ingest RAM back-pressure no longer deadlocks on caller-held memory**:
  `write_with_durability_async` compared *process* RSS against
  `max_ingest_ram_gb`, so a caller holding a large frame (e.g. a 1M-row Python
  DataFrame) pushed RSS over the limit and the write blocked forever — the
  caller cannot free the frame until the write returns. `Table` now has a
  `caller_reserved_bytes` accounting, set automatically by the Python `write`
  bindings from the incoming frame and by the JNI `appendBatch`/`mergeRows`
  from the Arrow batch, that is subtracted from RSS so the engine
  back-pressures on its own memory only. The ANN harness also writes in chunks.
- **Trino connector packaging**: the connector now runs in the stock
  `trinodb/trino:468` image. Fixed the SPI-version mismatch (compiled against
  Trino 468, updating the changed `ConnectorMetadata` signatures), the nested
  plugin ZIP (flattened), the host-glibc native lib (manylinux_2_28 build, with
  `arrow/pyarrow` gated behind the `python` feature so the JNI build does not
  link libpython), the non-serializable transaction handle (now an enum), the
  missing `libstdc++` (copied from a UBI stage for Arrow's JNI lib), and Arrow
  14's `MemoryUtil` reflection on Java 23 (`--add-opens` in `jvm.config`).
- **Trino connector surface area**: exercised the full connector through Trino
  468 (23/23 checks). Fixed a `NullPointerException` in `applyFilter` for
  single-value equality predicates (Trino 468's `ConstraintApplicationResult`
  rejects a null `ConnectorExpression`; now passes `Constant.TRUE`). Added a
  hidden `_bsdb_row_id` column (the Iceberg `$row_id` pattern) so the merge row
  id is not one of the data columns, switched the merge sink to the SPI
  `MergePage` helper, ordered merge deletes before inserts, and overrode the
  4-arg `beginMerge`. Implemented `CREATE SCHEMA` and `DROP TABLE` (native
  `createSchema`/`dropTable` over the object store). Verified: metadata
  (`SHOW`/`DESCRIBE`/`SHOW CREATE TABLE`/`information_schema`), reads,
  predicate pushdown (`=`/`<`/`IN`/`BETWEEN`), aggregates, `EXPLAIN`, `CTAS`,
  `INSERT … VALUES`/`SELECT`, `DELETE`, `MERGE`, `CREATE SCHEMA`, `DROP TABLE`.
- **Trino `MERGE` is fully supported** (all Trino clauses: `WHEN MATCHED THEN
  UPDATE`/`DELETE`, `WHEN NOT MATCHED THEN INSERT`, multi-clause and `AND`
  conditions). The `UPDATE` branch was previously a silent no-op: Trino's
  planner compares `ColumnHandle`s across separate `getColumnHandles()` calls
  (`QueryPlanner.planMerge` does `mergeCaseSetColumns.indexOf(dataColumnHandle)`),
  and the connector's handles had no `equals`/`hashCode`, so the lookup always
  missed and the planner fell back to the pre-update target row (and to `NULL`
  for `INSERT`). Every connector handle type now implements value equality.
  Verified end-to-end with value assertions: `UPDATE` writes the new value,
  `DELETE` removes the row, `INSERT` writes the source values, and a conditional
  clause is correctly skipped. The hidden row-id column now carries the primary
  key's own type (numeric, `VARCHAR`, or boolean), and the delete predicate is
  quoted for strings, so non-numeric keys work too.
- **Trino `DROP TABLE` now removes the table from `SHOW TABLES`**: the native
  `list_subdirs` derives names from the objects actually present instead of
  `list_with_delimiter`'s common prefixes, so a local filesystem's leftover empty
  directory no longer makes a dropped table reappear. `dropTable` also
  invalidates the manifest caches (`ManifestManager::invalidate_caches`) so a
  `CREATE TABLE` immediately after a `DROP TABLE` no longer reports "already
  exists" (the `LATEST_VERSION_CACHE`/`MANIFEST_CACHE` entries have short TTLs
  but were not cleared on drop). Covered by Rust unit tests
  (`core::ffi::list_subdirs_tests`).

### Tests
- `tests/test_index_lifecycle_invariants.rs` — indexes survive **compaction** and
  **snapshot rollback**, `DROP TABLE` removes index files, and `vacuum` removes
  orphaned index files.
- `tests/test_h1_h2_concurrency_regression.rs` — H1/H2/H3 regression coverage
  (truncate-vs-write race, `PendingWrite` atomicity under parallel writers, safe
  error bubbling).
- Orphan-cleanup-preserves-delete-files, `commit_synced_snapshot`
  reconciliation,
  and vacuum fail-closed tests.
- `tests/test_catalog_ddl.rs` — 28 tests covering every statement group, error
  cases, idempotency, whitespace robustness, and the `SET`/`SHOW` allowlist.
- `tests/test_vector_aggregates.rs` — 5 tests for the new aggregates and UDFs.
- `tests/python/test_vector_aggregates.py` — correctness of the Python helpers
  against numpy (vector aggregates) and hand-computed references (BM25/TF-IDF).
- `server/flight_sql/tests/test_flight_randomized_workload.py` — a randomized
  differential workload over the wire (INSERT/DELETE/OPTIMIZE/VACUUM/index/PK),
  asserting the visible id set matches an independent model after every step.

### Benchmarks
- **Full plan dataset matrix.** The benchmark suite now covers the datasets the
  plan calls for, all under the shared Docker envelope: vector — all eight
  ANN-Benchmarks sets (`sift`, `gist`, `fashion-mnist`, `mnist`, `glove-100`,
  `glove-200`, `nytimes`, `lastfm`); lexical — BEIR `scifact`, `nfcorpus`,
  `arguana` (plus the `scifact` hybrid RRF run); graph — SNAP `web-Google`,
  `roadNet-CA`, `com-LiveJournal` (500k-edge slices) against NetworkX and Neo4j
  GDS; SQL — ClickBench, TPC-H `lineitem`, and NYC TLC. `docker_bench.sh` gained
  `--datasets` (vector), `--graph-edges-host` (real edge lists), and
  `--sql-parquet-host`/`--sql-query`/`--sql-dataset` (real Parquet + query).
- **BEIR lexical/hybrid comparisons now run under the same Docker envelope.**
  The BEIR harness (`benchmarks/beir/run.py`, `run_hybrid.py`) is wired into the
  shared-envelope matrix as `docker_bench.sh --workload beir`: BenoStreamDB,
  Tantivy, and the server-backed OpenSearch all run in containers under one
  `--cpus`/`--memory` profile (previously BenoStreamDB/Tantivy ran in-process on
  the host while OpenSearch ran in a separately-sized container, so the
  comparison was not resource-matched). BEIR is CPU-bound (Okapi BM25 has no
  GPU path; the hybrid dense half is an HNSW query), so there is no GPU pass and
  every engine reports `backend: cpu`. The harness gained `--data-dir` (for the
  read-only mount), an auto-download of the BEIR corpus when absent, a
  self-describing `Resource Envelope` line, and a per-engine `backend` field
  (the backend the algorithm actually ran on). Each engine also emits a
  competitor-schema JSON record (`--json-dir`), which `generate_summary.py`
  rolls into the consolidated benchmark report as a table alongside the
  vector/graph/SQL rows. CI runs the embedded `benostreamdb,tantivy` baseline on
  the single runner.
- **The GPU pass now runs every engine.** `docker_bench.sh --gpu`/`--both` runs
  the same engine set as the CPU pass instead of only the GPU-capable ones, so
  the CPU-only competitors (pgvector, LanceDB, OpenSearch, Neo4j CPU GDS,
  DuckDB) run on CPU inside the GPU container under the same envelope. This
  makes the GPU/no-GPU distinction explicit in the results rather than hiding
  it. Restrict with `--gpu-engines` for a smaller pass.
- **LanceDB is reported with both of its ANN algorithms.** The vector adapter
  now runs LanceDB's **default** disk-ANN index (IVF_PQ, `lancedb`) *and* its
  scalar-quantized HNSW variant (`lancedb_hnsw`), instead of only the
  HNSW-shaped one. On the 20k SIFT run the default IVF_PQ index reaches
  recall@10 0.492 vs 0.924 for HnswSq — reporting only the HNSW variant
  overstated LanceDB's recall. The BEIR hybrid adapter also builds the IVF_PQ
  index (its docstring claimed a vector index but none was created).
- **Docker competitor matrix runs end-to-end.** The shared-envelope runner
  (`benchmarks/competitors/docker_bench.sh`) now builds and runs every
  participant — FAISS, hnswlib, LanceDB, pgvector, OpenSearch, and
  BenoStreamDB — under one `--cpus`/`--memory` profile. Fixes: `.dockerignore`
  no longer excludes the harness; the runner image installs `build-essential`
  (hnswlib builds from source) and `opensearch-py`; the runner logic moved to
  `run_bench.sh` (YAML folding mangled the inline compose `command:`);
  pgvector's `CREATE INDEX` inlines its integers (DDL cannot take bind
  parameters); the OpenSearch adapter uses `knn_vector` mappings and the
  `query.knn` search shape; and the rollup only includes results written during
  the current run. The BenoStreamDB harness now emits the same JSON record as
  the competitors (previously it wrote markdown into a `.json` path, so the
  rollup silently dropped our engine), so BenoStreamDB appears in the rollup
  with its index size.
- **Like-for-like vector comparison.** The competitor adapters return ids only
  (no payload fetch), so the rollup now reports BenoStreamDB's *pure-index*
  number rather than its full search+row-fetch number, and LanceDB projects
  only `id` in its search. Previously the rollup compared BenoStreamDB's
  full-search QPS against the competitors' search-only QPS.
- **cuGraph on Blackwell.** The `cugraph-cu12` wheels have no sm_120 kernels;
  the GPU runner now installs `cugraph-cu13`, and the adapter sets
  `renumber=True` and `rmm.reinitialize(managed_memory=True)`. Verified against
  NetworkX (max abs diff 2.6e-06, identical top-5). `networkx` now pulls in
  `scipy`.
- **cuGraph now runs through the normal compose path.** The harness applied
  `RLIMIT_AS` (the RAM envelope) to every engine, but CUDA/RAPIDS managed memory
  reserves far more *virtual* address space than physical RAM, so the `dlopen`
  of `libcugraph.so` failed with a misleading "cannot open shared object file".
  `apply_envelope` now skips `RLIMIT_AS` for GPU engines (`cugraph`, or any
  `--device gpu` run); the container cgroup
  (`deploy.resources.limits.memory`) already enforces the physical RAM envelope.
  The `docker run` workaround in `docker_bench.sh` is removed, and the GPU pass
  now builds the base runner image before the GPU image (the GPU image is
  `FROM bsdb-bench-runner:latest`, so a stale base shipped an old harness).
- **Matrix wheels are manylinux.** `--build-wheel` now builds with
  `maturin --zig --compatibility manylinux_2_28`, so the wheel loads on the
  slim runner image regardless of the host's (newer) glibc. The release
  pipeline already ships manylinux wheels via `maturin-action`.
- **Trino SQL competitor.** `_sql_trino` connects to a `trino` compose service
  (Hive connector, file metastore) and registers the shared Parquet as an
  external table `t`, so the shared SQL runs unchanged. Config in
  `benchmarks/competitors/trino/etc/`.
- **Iceberg / Delta round-trip harness.** `benchmarks/iceberg_roundtrip/run.py`
  writes a table with BenoStreamDB (Iceberg) and reads it back through Spark or
  Trino (Iceberg), or compares against Delta Lake — the Tier-2 table-format
  interop baseline.

## [0.11.1] - 2026-09-29

### Added
- **GPU-Native Index Construction crate (`benostream-gpu-ann`) (A11 Complete)**
  —
  A standalone, community-ready Rust library delivering high-performance
  GPU-native
  approximate nearest neighbor (ANN) vector indexing. Designed from first
  principles
  to target all three major GPU execution backends: NVIDIA CUDA (via `cudarc`
  and
  dynamic NVRTC runtime compilation), Apple Silicon Metal (via `metal-rs`), and
  cross-platform AMD/Intel/Vulkan GPUs (via `wgpu` and optimized WGSL compute
  shaders),
  with an automatic fallback to AVX2/NEON Rayon CPU SIMD.
  - **Stage 1 (GPU IVF-Flat)**: Coarse Voronoi k-means partitioning with
    parallel GPU
    centroid scans and flat cluster search, bypassing graph construction
    entirely.
  - **Stage 2 (GPU-Accelerated HNSW)**: Solved the long-standing roadmap
    challenge
    without forking `hnsw_rs`. Owns flat contiguous memory buffers natively in
    Rust
    and offloads multi-layer candidate frontier distance evaluations to GPU
    shaders.
    Achieves **100% Recall@10** on active RTX 3090 hardware (`WGPU_Default`).
  - **Stage 3 (GPU-Native CAGRA)**: Fixed-degree regular graph (`[N *
    graph_degree]`)
    engineered specifically for GPU memory hierarchy and single-transaction
    coalesced
    warp memory loads. Features GPU Voronoi clustering, GPU 2-hop neighbor
    refinement
    (NN-Descent iterations), and anisotropic edge pruning. Achieves **96.75%
    Recall@10**.
  - **Unified Public API**: Provides `VectorIndex`, `IndexBuilder`, and
    `Algorithm`
    abstractions with 100% bidirectional CPU ↔ GPU interoperability. An index
    built
    on CPU can be queried on GPU, and an index built on GPU can be queried on
    CPU.
  - **Comprehensive Metric Parity**: Strict differential parity verified across
    all 6
    distance metrics (L2, Cosine, InnerProduct, L1, Hamming, Jaccard) within
    $10^{-4}$
    relative numerical tolerance and self-match rankings.
  - **No-Panic Fuzz & Soak Hardening**: Includes a randomized adversarial fuzz
    suite
    (`tests/test_fuzz_no_panic.rs`) verifying total ordering and graceful error
    handling
    under non-finite/adversarial floats (NaN, Inf, zeroes, dimension mismatches,
    sparse
    filters), alongside a sustained high-throughput soak harness
    (`tests/test_soak.rs`)
    verifying zero memory or VRAM handle leaks.
  - **Product Integration**: Wired directly into root `Cargo.toml` with feature
    forwarding
    (`cuda`, `wgpu`), with `ComputeContext::to_gpu_ann_context()` and
    `VectorMetric`
    bi-directional conversions.

## [0.11.0] - 2026-09-28

### Added
- **Index preload (`Table::preload_indexes_async`)** — warms the read-path
  caches (HNSW/IVF vector, inverted/BM25, CSR graph, manifest/PARQUET metadata)
  at table-open time so the first query is served from memory instead of the
  object store — the Dgraph-style "in-memory posting list" residency, but on
  demand. `PreloadOptions` takes an explicit `max_memory_bytes` budget: indexes
  are warmed into RAM until the budget is exhausted, then the remainder spills
  into the mmap disk cache (`BENOSTREAM_DISK_CACHE_DIR`, `MADV_RANDOM`) for
  out-of-core serving. The bounded moka caches evict by LRU/TinyLFU + TTI, so
  overflow simply evicts the least-recently-used entries. Type filters
  (`include_vector` / `include_inverted` / `include_graph`) and idempotent
  re-warm are supported. Exposed via `Table.preload_indexes(...)` in Python.
  The `benostream-search` gateway calls it on open, controlled by
  `BENOSEARCH_PRELOAD` (default on) and `BENOSEARCH_PRELOAD_GB` (default 4 GiB;
  overflow spills to disk). 5 tests cover budget, disk spill, correctness,
  idempotency, and type filters.
- **Wikipedia section extraction (`ingest_wikipedia --sections`)** — extracts
  every `== Section ==` header with byte ranges so a downstream build can carve
  full wikitext into a 3-tier dataset (article → section → section text). The
  regex is now multiline and the extraction is per-page, fixing two bugs: the
  non-multiline anchor only matched a page's leading header, and the running
  `end_byte` state accumulated across pages, corrupting every section's range.
- **3-tier Wikipedia build pipeline (`scripts/build_three_tier.py`)** — a single
  end-to-end, idempotent script with resumable stages (`download`, `parse`,
  `merge`, `tier`, `embed`, `load`): fetch the enwiki XML chunks, run
  `ingest_wikipedia --sections --full-text` in parallel, merge the per-worker
  parquets, build the articles / sections / section_texts tiers, compute GPU
  SentenceTransformer embeddings over section text, then load into BenoStreamDB
  with an edge table (CSR) + HNSW and preload the index caches. It replaces the
  earlier separate `embed_sections.py` and `load_three_tier.py` helpers.
- **CSR-backed `subgraph()` / `graph_neighbors()` fast path.** `Table.subgraph`
  and `Table.graph_neighbors` now accept an optional `graph_column` kwarg. When
  a memory-mapped CSR graph index exists on that column, the multi-hop BFS runs
  in Rust over `MultiSegmentCsrGraph::get_neighbors` (bounded, visited-set)
  instead of the SQL `bfs_visited` path, which re-materialized the entire
  deduplicated symmetric adjacency (`SELECT DISTINCT src,dst ... UNION ALL
  SELECT DISTINCT dst,src ...`) to parquet on every call. On the 383M-edge wiki
  demo this turns a ~100 s-class 1-hop `subgraph()` into a sub-second call.
  `subgraph(..., graph_column=...)` returns the induced edges with payload
  columns preserved. The CSR is forward-directed; for `directed=False` a reverse
  CSR is used when the table has a graph index on the other endpoint column,
  otherwise the call transparently falls back to the SQL path so undirected
  results are unchanged. The default (`graph_column=None`) is byte-for-byte the
  old behaviour.
- **`Table.has_graph_index(column)`** — reports whether a CSR graph index exists
  on a column, so callers can opt into the fast path.
- **Shared CSR loader** `python::helpers::load_multi_csr` — one implementation
  of the manifest → mmap → `MultiSegmentCsrGraph` block, now used by
  `GraphAPI`, `Table.subgraph`, `Table.graph_neighbors` and `Table.drift_search`
  (previously copy-pasted in three places).
- **Graph RAG uses the CSR fast path.** `graph_rag_search` (local mode) now
  routes its induced-subgraph extraction through the CSR when the edge table has
  a graph index on `source` and no relation/time filters are requested, with a
  fallback to the original call for tables without a CSR.
- **Max-degree truncation for CSR traversal.** `Table.subgraph`,
  `Table.graph_neighbors` and `Table.graph_rag_search` accept an optional
  `max_degree` kwarg. Nodes whose degree exceeds the cap (Wikipedia
  "super-nodes" such as `United States`) are reported but not expanded, bounding
  the frontier explosion at each hop. On the wiki demo an unconstrained 2-hop
  BFS from 5 seeds reached ~44M rows; truncation keeps it bounded.
- **Token-budget cap for CSR traversal.** `Table.subgraph` and
  `Table.graph_neighbors` accept an optional `max_nodes` kwarg, and
  `Table.graph_rag_search` accepts `traversal_max_nodes` (distinct from its
  existing output `max_nodes`). It is a hard cap on the total visited set (seeds
  included): once reached, expansion stops mid-hop. Unlike `max_degree` it
  bounds memory regardless of graph density.
- **Leiden community detection.** `Table.leiden_communities(resolution)` and the
  `leiden_communities` SQL UDAF add a connectivity-refinement phase on top of
  Louvain's greedy modularity local move, so every returned community is
  internally connected. `Table.communities(resolution, algorithm=...)` selects
  between `'louvain'` (default) and `'leiden'`; `summarize_communities` and
  `graph_rag_search` (global mode) accept the same selector.
- **LLM-generated community reports.** `summarize_communities` accepts optional
  `llm` and `embed` callables. `llm(prompt) -> str` generates a model-authored
  report per community (stored in `report_column`); `embed(texts) -> vectors`
  embeds the reports and writes an `embedding` column with an HNSW index, making
  community reports first-class vector-retrieval units.
- **Dynamic community selection via an LLM router.** `graph_rag_search` (global
  mode) accepts `llm_router(query, community_summary) -> float` and
  `relevance_threshold`. Communities scoring below the threshold are pruned
  before descending, so irrelevant subtrees are never expanded. Without a
  router, the existing `seed_overlap`/`member_count` heuristic is used.
- **`Table.extract_graph()`.** Builds a knowledge graph from a document table
  using an LLM: `llm(prompt) -> json` returns entities, relationships, and
  claims, which are materialized as an edge table (with a graph index on
  `source`) plus an optional claims table (`subject`, `object`, `claim`,
  `source_doc`, `confidence`). Entity names map to stable uint64 node IDs via
  SHA-256, so the graph is reproducible across runs.
- **Bounded, configurable DataFusion memory.** `BenoStreamSession::new(None)`
  and `Table.execute_sql` (the path the graph UDFs run on) now apply a query
  memory limit instead of leaving it unbounded, so SQL sorts/joins/aggregations
  spill to disk rather than OOMing. The limit is derived from the effective
  (cgroup-aware) memory at `DATAFUSION_MEMORY_FRACTION` of it, and is
  independently configurable via `BSDB_DATAFUSION_MEMORY_GB`; an explicit limit
  still wins. Deliberately its own knob, not part of a fixed split — see
  `docs/RESOURCE_LIMITS.md`.
- **CSR-backed community detection.** `Table.communities()` now runs Louvain or
  Leiden over the memory-mapped CSR when the table has a graph index, keeping
  only **O(V)** state resident (`community`, `degree`, `comm_tot`) instead of
  buffering every edge in RAM. On a 6M-node graph that is ~72 MB versus ~7.7 GB
  for the 383M-edge accumulator path. The SQL UDAF remains the fallback for
  tables without a CSR.
- **`Table.subgraph_nodes()` and bounded Graph RAG materialization.** New
  `subgraph_nodes(seeds, hops, ...)` returns the BFS visited node IDs without
  the edge join. `graph_rag_search` (local mode) now uses it to compute the
  neighbourhood set, and materializes induced edges only for the ranked
  `max_nodes` nodes — so peak memory scales with `max_nodes`, not the graph
  size. `summarize_communities` fetches only `(id, content, title)` instead of
  `SELECT *` over the whole document table.
- **Incremental communities with stable IDs.** `Table.communities()` and the
  new `Table.update_communities(previous=...)` now return a `community_id`
  column alongside `community`. `update_communities` warm-starts the CSR
  algorithm from a previous partition, so a graph that grows by a few edges
  does not renumber every community; unchanged communities keep their IDs.
  Without a CSR it falls back to a full recompute.
- **`build_profile()` / `is_debug_build()`** — report whether the loaded
  extension was compiled with optimisations, so benchmarks can refuse to report
  timings from a `maturin develop` (debug) build.

### Fixed
- **Binary literal parser panicked on a bare `B'` prefix (fuzz-found).**
  `parse_binary_vector` sliced `&trimmed[2..trimmed.len() - 1]` after only
  checking that the input started with `B'`/`b'` and ended with `'`. For the
  two-character input `B'` (or `b'`, including with surrounding whitespace) the
  opening prefix and the closing quote overlap, so the slice range was reversed
  (`2..1`) and panicked with *"byte range starts at 2 but ends at 1"*. A length
  guard now rejects such inputs with a `Plan` error; `B''` (a well-formed empty
  literal) still parses. The crash input is added to
  `fuzz/corpus/parse_binary/prefix_only` as a permanent regression seed.
- **CSR graph index direction mismatch.** `add_index` registered a graph index
  under both its `src_column` and the original `column` argument, so two graph
  indexes (forward + reverse) collided in `index_configs`, clobbering each other
  and mislabeling the physical CSR files. The CSR fast path could then follow
  the wrong direction, returning a different induced subgraph than the SQL
  `bfs_visited` path. Graph indexes are now keyed solely by `src_column`. The
  on-disk format is bumped to `graph_v2`; legacy v1 files are ignored (the table
  falls back to the correct SQL path) and rebuilt in the background on open.
- **`graph_rag_search` text query against a vector column.** Passing a raw
  string query when `vector_column` is a configured embedding column previously
  routed to `search(column=<vector col>, query=<str>)` and failed with
  `No keyword/inverted index found for column 'embedding'`. It now falls back to
  BM25/keyword search on a real text column (`title`/`summary`/`content`/...),
  or raises a clear `ValueError` telling the caller to pass an embedding vector
  when no text column exists.
- **OpenSearch/Elasticsearch aggregations in `bsdb-search`.** `POST
  /{index}/_search` now accepts an `aggs` (or `aggregations`) object, compiled
  to SQL and executed with DataFusion: `terms`, `histogram`, `date_histogram`,
  `range`, `filter`, `missing`, `avg`, `sum`, `min`, `max`, `value_count`,
  `cardinality`, `stats`, `extended_stats`, and nested `aggs` on bucket
  aggregations. Aggregations run over the top-level `filter`. See
  `docs/OPENSEARCH_COMPATIBILITY.md`.
- **Complete Qdrant v1.x REST API in `bsdb-search` (port 6333).** The
  Qdrant-compatible listener now implements the full surface instead of the
  three-endpoint subset: collection list/exists/get/create/update/delete and
  payload-index create/delete; point upsert, retrieve (`GET` **and** `POST`),
  get-by-id, `search`, `query` (universal query API), `scroll`, `count`,
  `recommend`, `discover`, `batch`, and delete-by-ids/filter; payload
  set/overwrite/delete/clear; vector updates; collection aliases
  (list/create/delete/rename); and the service endpoints `GET /`, `/healthz`,
  `/livez`, `/readyz`, `/telemetry`. See `docs/QDRANT_COMPATIBILITY.md` for the
  support matrix.
- **Qdrant collection metadata sidecar.** The distance metric and HNSW config
  are persisted in `_qdrant_collection.json` under the collection root, so
  `GET /collections/:name` reports the configured `distance` and `size`.
- **`benostreamdb-search/tests/qdrant_api.rs`** — an in-process conformance
  suite (axum `Router` + `tower::ServiceExt::oneshot`) covering every endpoint.

### Fixed
- **Qdrant `points_count` is now the real row count.** `GET /collections/:name`
  previously read stale manifest statistics and reported `0` immediately after
  an upsert; it now counts the effective rows (including the write buffer).
- **Qdrant search scores follow the collection's `distance`.** The engine's
  trailing distance column (squared L2 / `1-cos` / `-dot` / L1) is converted to
  the Qdrant score for the configured metric — Euclidean distance for `Euclid`,
  cosine similarity for `Cosine`, dot product for `Dot` — and results are
  ordered best-first. Previously every metric returned squared-L2.
- **Qdrant upserts overwrite by id.** Upserts are now *flush → delete-by-id →
  append* (merge-on-read via Iceberg position deletes) instead of a plain
  append, so re-upserting an id replaces the point.
- **Qdrant `GET /` returns service info** (`{title, version, commit}`) instead
  of `404`, and error/success envelopes carry the real elapsed `time`.
- **Qdrant collections index only the `vector` column.** The shared
  `open_or_create` path indexes every column (needed for OpenSearch BM25); the
  new `open_or_create_qdrant` path avoids building inverted indexes over payload
  columns, which made each point write rebuild indexes for columns that are
  never lexically searched.

### Changed
- **Qdrant docs are now honest.** `README.md`, `docs/INSTALLATION.md`,
  `docs/architecture.md`, `docs/index.md`, `docs/ROADMAP.md`, and
  `docs/PYTHON_VECTOR_API.md` no longer claim blanket "Qdrant-compatible"
  support; they link to the new `docs/QDRANT_COMPATIBILITY.md` support matrix.

## [0.10.0] - 2026-09-24

### Changed
- **Resource limits are now cgroup-aware and always active (serverless-safe).**
  The ingest RAM high-water mark previously defaulted to 80% of the *host's*
  physical memory (`/proc/meminfo`), so a 4 GB container on a 32 GB host got a
  ~25 GB ceiling and OOM-killed instead of throttling. `core::resources` now
  resolves the memory actually available to the process — cgroup v2
  `memory.max`, then cgroup v1 `memory.limit_in_bytes`, then host RAM, then a
  conservative 4 GiB fallback — and every memory limit is derived from it:
  `BSDB_MAX_INGEST_RAM_GB` and `BSDB_INGEST_MEMORY_BUDGET_GB` default to 80% of
  it,
  and `BSDB_INDEX_BUILD_CONCURRENCY` scales as one build per ~8 GiB (capped at
  the
  CPU count, at least 1). The "off unless configured" semantics are gone: an
  unset or non-positive variable now means "use the derived default", never
  "disable the guard". `BSDB_MIN_FREE_DISK_GB` defaults to 1 GiB, and
  `BSDB_INDEX_BUILD_CONCURRENCY=0` no longer re-enables unbounded fan-out.
  `HeapTrimPolicy::from_budget_or_env` always returns a policy. See
  `RESOURCE_LIMITS.md` for how the knobs interact, with worked 4 GB → 128 GB
  examples.
- **`bincode` 1.3 → 2.x.** bincode 1.3.3 is unmaintained (RUSTSEC-2025-0141).
  The vendored HNSW dump/load path now uses `bincode::serde` with
  `bincode::config::legacy()`, which is byte-compatible with 1.3, so existing
  index dumps still load. The yanked `chacha20` (0.10.1 → 0.10.2) and `spin`
  (0.9.8 → 0.9.9) transitive crates were updated. The unmaintained `paste`
  (RUSTSEC-2024-0436) remains, pulled in transitively by `datafusion-common`
  52.x; it is a compile-time proc-macro with no runtime surface and is
  documented in `deny.toml`.
- **`cargo fmt --all` applied** across the workspace.
- **Memory guards now size from *available* memory, not total.** The value that
  derives `BSDB_MAX_INGEST_RAM_GB`, `BSDB_INGEST_MEMORY_BUDGET_GB`, and
  `BSDB_INDEX_BUILD_CONCURRENCY` previously used the host's `MemTotal`, so on a
  shared machine the guards were sized for RAM the process could not actually
  get — the demo load was OOM-killed at ~74 GB RSS on a 121 GiB host because the
  derived high-water mark was ~102 GB. `effective_memory_bytes()` now prefers
  `MemAvailable` (Linux) for the host-RAM fallback; a cgroup limit or
  `RLIMIT_AS` ceiling is still used as-is. New `usable_memory_bytes()` exposes
  the value.

### Added
- **Disk-headroom admission limit + a single resource-limits reference.** New
  `core::resources` module: `BSDB_MIN_FREE_DISK_GB` refuses a flush when a local
  table's filesystem is below the threshold, so the failure is a clear error
  *before* any bytes are written rather than a partial segment, and
  `benostreamdb_free_disk_bytes` exposes the sampled headroom. Fail-open when
  the threshold is unset or the free space cannot be queried. Every
  resource/back-pressure knob (`BSDB_MAX_INGEST_RAM_GB`,
  `BSDB_INGEST_MEMORY_BUDGET_GB`, `BSDB_INDEX_BUILD_CONCURRENCY`,
  `BSDB_MIN_FREE_DISK_GB`, and the CPU/concurrency bounds) is now documented in
  one place: `RESOURCE_LIMITS.md`.
- **No-panic gate enforced for the library.** The production `unwrap()`/
  `expect()`/`panic!` count across both crates went **289 → 0** (247 core
  library
  + 42 search library remediated). With the baseline at zero the staged
  `no-panic` cargo feature is now live: `scripts/no_panic_check.sh` runs the
  ratchet *and* `cargo clippy --features python,no-panic`, which turns the three
  restriction lints into hard errors for non-test code. The only exemptions are
  the vendored `hnsw_rs` subtree (graph invariants in the inner search loop) and
  `src/telemetry/metrics.rs` (static `prometheus` definitions with no infallible
  constructor), both module-scoped and documented in `NO_PANIC_POLICY.md`.
- **Fuzzing workspace (`fuzz/`).** Coverage-guided fuzzing (cargo-fuzz /
  libFuzzer) for the untrusted-input parsers: the dense, sparse, and binary
  vector literal parsers, the SQL string rewriters (`strip_partitioned_by`,
  `rewrite_sql_string`), and the Qdrant-compatible request bodies (whose
  `#[serde(untagged)]` enums are a classic pathological-input surface). Seed
  corpora are committed; the `Fuzz` workflow runs each target for a 5-minute
  budget on push and uploads crash artifacts. `fuzz/` is its own workspace
  root so `libfuzzer-sys` never enters the main build, and it shares the root
  lockfile to stay on the pinned toolchain. See `fuzz/README.md`.
- **Soak / stress suite (`tests/stress/`).** A maintained replacement for the
  one-off `test_oom.py` script, asserting graceful degradation under saturation:
  a low ingest RAM high-water mark must pause and resume rather than OOM
  (`test_memory_backpressure.py`), gated index builds must not starve queries
  (`test_cpu_saturation.py`), many commit cycles must stay durable
  (`test_disk_io.py`), and a mixed write/search/read churn must keep memory
  bounded over a wall-clock budget (`test_mixed_soak.py`). Skipped unless
  `BSDB_STRESS=1`; run on push by the `Soak` workflow. A Rust long-running soak
  harness (`tests/soak.rs`, `#[ignore]`) provides the same coverage in-process.
- **Event-driven ingest back-pressure.** The ingest RAM high-water mark
  (`BSDB_MAX_INGEST_RAM_GB`) previously blocked writers with a flat 500 ms sleep
  loop that re-read `/proc/self/status` on every iteration — wasting time when
  memory had already been freed and delaying resumption when it had not. Writers
  now wait on a `tokio::sync::Notify` signalled by the tasks that release memory
  (a segment index build finishing, or the ingest loop trimming the heap) with a
  250 ms bounded fallback poll, so they resume the instant memory is reclaimed.
  New Prometheus metrics expose the sampled RSS gauge, the back-pressure pause
  count and duration distribution, and index-build gate wait time
  (`benostreamdb_ingest_rss_bytes`,
  `benostreamdb_ingest_backpressure_pauses_total`,
  `benostreamdb_ingest_backpressure_pause_seconds`,
  `benostreamdb_index_build_gate_wait_seconds`). Covered by
  `memory_reclaimed_notification_wakes_blocked_writer`.
- **No-panic ratchet for production code paths.** `scripts/no_panic_check.sh`
  counts `unwrap()`/`expect()`/`panic!` sites in production (non-test) code via
  `cargo clippy --lib` with the restriction lints forced on, and fails if the
  count grows beyond `scripts/no_panic_baseline.txt`. Test code (`#[cfg(test)]`
  blocks and the `tests/` crates) is excluded automatically. Wired into CI as
  the `no-panic` job. A staged `no-panic` cargo feature already carries the
  `#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used,
  clippy::panic))]` attributes for when remediation completes. See
  `NO_PANIC_POLICY.md`.

### Fixed
- **The `bsdb-search` server binary could not be built.**
  `benostreamdb-search`'s
  `main.rs` installed `dhat::Alloc` as the global allocator while the
  `benostreamdb` library installs jemalloc on Linux, so linking failed with
  "the `#[global_allocator]` in this crate conflicts with global allocator in:
  benostreamdb". Heap profiling is now opt-in behind a `dhat-heap` feature and
  the default build works. CI now runs `cargo check --bins` (and the search
  crate's bin), which would have caught this: clippy does not lint bin targets
  when a same-package lib exists, so nothing was compiling them.
- **Panics removed from the network-server binaries.** `src/bin/gateway.rs`
  unwrapped a user-supplied filter (`QueryFilter::parse(...).unwrap()` in the
  `POST /query` handler) — a malformed filter now returns 400. Also fixed: the
  gateway's store creation, mock batch construction and listener bind; the
  iceberg_rest server's bind, graceful-shutdown and metrics response;
  `bsdb.rs`'s `print_batches`; and the search server's bind/parse/signal-handler
  `expect`s. The four production bins and the search bin now carry the
  `cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used,
  clippy::panic))` gate.
- **Panics removed from the search request path.** `benostreamdb-search` had 42
  production `unwrap()` sites: `handlers/search.rs` unwrapped JSON map lookups
  and every Arrow→JSON downcast, and `handlers/{bulk,qdrant,docs,metrics}.rs`,
  `infer.rs` and `state.rs` unwrapped request data. JSON lookups now use
  `ok_or_else`/`if let`, and the Arrow downcasts (guaranteed by the
  `data_type()` match) are total — a violated invariant yields `null` instead of
  crashing the handler. The crate is now clean.
- **Panics removed from the graph UDF compute path.** All 14
  `src/core/sql/graph_udf/*.rs` accumulators unwrapped `states[i]` downcasts in
  `merge_batch`/`update_batch`; they now return
  `DataFusionError::Execution`. These run inside SQL `GROUP BY` for graph
  analytics, so a malformed state previously aborted the query with a panic.
- **Panics removed from the manifest decode, reader filter, and segment paths.**
  `ManifestValue::from_array` (Arrow type-invariant downcasts), the
  inverted-index
  filter's key downcasts, and the segment writer's path handling no longer
  `unwrap()`; the filter's `NaiveDate::from_ymd_opt(1970, 1, 1).unwrap()` sites
  became `NaiveDate::default()`.
- **Panics removed from the write/index-build path.** `table/write.rs` schema
  `index_of` lookups, `index/csr_graph.rs` fixed-size reads and id
  binary-searches, and `index/build_graph.rs` edge downcasts are now total.
- **`build.rs` returns `Result`** instead of unwrapping `OUT_DIR`/file writes.
- **Unbounded concurrent index builds OOM-killed large loads.** The write path
  spawns one segment index build per flush onto the tokio runtime, whose worker
  count is the CPU count. On a 32-core workstation that meant up to 32 builds in
  flight, each holding its segment's vectors plus the HNSW/IVF/quantizer
  structures (several GB at the demo's 1 GB flush size) — the Wikipedia load hit
  **105 GB RSS** and was OOM-killed, and before that thrashed so badly that
  per-chunk time exploded 43 min → 8.65 h. Builds now go through a shared
  semaphore (`Table::index_build_gate`, default 2,
  `BSDB_INDEX_BUILD_CONCURRENCY`)
  that back-pressures the writer; the bounded working set also removes the
  page-cache eviction that made the unbounded case slow, not just fatal.
- **`add_index` re-indexed every prior segment on each call (O(n²) demo-load
  OOM).** `add_index` on a non-empty table unconditionally triggered
  `backfill_indexes_async`, which rebuilt the indexes of *every* manifest entry
  via an unbounded `futures::future::join_all` — with no "already indexed" skip
  and no per-segment build gate (the single permit gated the whole task, not the
  fan-out). The demo's chunked load calls `add_index` in every fresh process, so
  each chunk re-indexed all previously loaded data: per-chunk time grew
  293s → 885s → 2071s and peak RSS climbed until the process was killed.
  Backfill
  now skips segments that already carry the required `(column, index_type)` pair
  and bounds in-flight builds with `buffer_unordered(index_build_concurrency)`,
  taking the shared build gate per segment. `prepare_demo.py` also sets the
  default device once and only applies `add_index` for columns not already
  configured (the config is restored from the manifest on open). A full
  51.8M-row node load now completes in **~45 min** (est.) at a flat **~46–56 s
  per M rows (~20k rows/s)**, versus the escalating 293s → 885s → 2071s that
  preceded the kill — see `examples/web_ui/README.md`.
- **Heap not returned to the OS at flush/build boundaries.** glibc keeps freed
  memory in per-thread arenas, so the HNSW/TQ builders' millions of small
  allocations ratcheted RSS toward the sum of every arena's high-water mark.
  Both
  the flush path and the end of each background index build now call
  `memory::trim_if_over_budget` (budget via `BSDB_INGEST_MEMORY_BUDGET_GB`,
  default 8 GiB) so freed arena pages are released before the next build starts.
- **`simple_kmeans` allocated a `Vec` per vector** in the capacity-capped final
  assignment. Over millions of vectors that churned and fragmented the heap. It
  now reuses a thread-local scratch buffer, keeping the strict capacity cap.

### Security
- **PR security checks + a local AI diff review.** A new `PR Security Checks`
  workflow runs `cargo-deny` (advisories, bans, licenses, sources) and gitleaks
  on every pull request — free and secret-free. For deeper review,
  `scripts/ai_pr_review.py` is a local tool that reads the standard `OPENAI_*`
  environment variables (OpenRouter by default) and asks an LLM to look for
  *injected* vulnerabilities in a diff (backdoors, weakened checks, credential
  exfiltration, unsafe deserialization, injection, disabled security controls,
  supply-chain changes), printing findings and optionally posting a PR comment.
  It is intentionally not run in CI, so no secrets are stored.
- **`SECURITY.md` right-sized for a one-person open-source project.** The bug
  bounty program was removed, the response SLA is now explicitly best-effort,
  and only the latest release is supported (fast-moving project).

## [0.9.1] - 2026-09-23

### Fixed
- **Metal distance kernels read past the end of the vector buffer.** The
  dispatch covers `ceil(n/256)*256` threads, but the kernels indexed rows by
  `thread_position_in_grid` with no bounds guard, so up to 255 threads read out
  of bounds. Added an `n_vectors` argument + guard to all six kernels, and
  aligned the Hamming (`!=`) and Jaccard (`> 0.0`) comparisons with the CPU/CUDA
  definition. Covered by the new macOS CI job (Apple Silicon).
- **rustdoc: `VEC_TMP_MAGIC` linked to a private item**, breaking
  `cargo doc --features python,wgpu,java`. Now plain text.
- **CUDA distance kernels launched with the wrong grid and no shared memory.**
  `CudaBackend::compute_distance` used `LaunchConfig::for_num_elems`, which
  packs
  rows into 1024-thread blocks and sets `shared_mem_bytes: 0`, but the kernels
  use one block per row with an `extern __shared__` reduction — an illegal
  memory access, not a wrong answer. Now launches `n_vectors` blocks of 256
  threads with the shared memory sized for the block. Caught by the new
  cross-backend harness.
- **CUDA JIT now works with pip's CUDA 13 wheels.** cudarc 0.13's nvrtc loader
  probes a fixed candidate list (`libnvrtc.so`, `libnvrtc64*.so`,
  `libnvrtc.so.{12,11,10,1}`) that predates CUDA 13, so `nvidia-*-cu13` wheels
  (`libnvrtc.so.13`) were never found: the probe panicked and the GPU silently
  fell back to CPU. `core::index::nvrtc` now resolves *whatever* `libnvrtc.so*`
  is installed — any version, any layout (`BSDB_NVRTC_PATH`, the interpreter's
  `site-packages`, `CUDA_HOME`/`CUDA_PATH`, `PYTHONPATH`, `LD_LIBRARY_PATH`,
  system paths) — preloads the `libnvrtc-builtins` companion with `RTLD_GLOBAL`,
  and compiles the embedded `.cu` sources itself, handing the PTX to cudarc. No
  re-exec, no env-var prefix, no shim script (`scripts/create_cuda_shims.sh`
  removed).
- **Demo load resume no longer duplicates rows.** `scripts/prepare_demo.py`
  resumed from the rounded-down chunk boundary on the assumption that chunks
  commit atomically. They don't: the write path spills to a real commit whenever
  the buffer exceeds `BENOSTREAM_CACHE_GB` (default 1 GB), so a killed chunk
  leaves partial rows committed and the round-down re-wrote them. Resume now
  continues from the exact committed row count (`_resume_offset`), guarded by a
  regression test.

### Added
- **GPU acceleration for sparse vectors (dense-conversion path)**: batched
  `bsdb.sparse_l2_batch` / `bsdb.sparse_cosine_batch` /
  `bsdb.sparse_inner_product_batch` convert a sparse query plus N sparse vectors
  to dense and dispatch to the existing dense GPU kernels. Backend-agnostic, so
  it works on every backend; it pays off once the batch clears
  `GPU_DISPATCH_THRESHOLD`. Equivalence with the sparse CPU reference is covered
  by `sparse_dense_equivalence_tests`.
- **GPU-accelerated packed-binary distance (CUDA + WGPU)**:
  `GpuBackend::compute_binary_distance`
  plus packed-u8 Hamming/Jaccard kernels for CUDA (`hamming_packed.cu`,
  `jaccard_packed.cu`) and WGPU (`wgpu_binary_kernel.wgsl`, covering AMD/Intel
  via Vulkan). New batched Python API `bsdb.hamming_distance_batch` /
  `bsdb.jaccard_distance_batch` (one packed query vs N packed vectors), plus
  Metal (`mps/hamming_packed.metal`, `mps/jaccard_packed.metal`) — verified by
  the new `metal` CI job on Apple Silicon. Cross-backend harness passes on an
  RTX 3090 (`["cpu", "cuda", "wgpu"]`) and on macOS CI (`["cpu", "mps"]`).
- **Cross-backend GPU correctness harness**:
  `cross_backend_matches_cpu_all_metrics`
  runs the same vectors through every *available* backend (CUDA, Metal, WGPU)
  and asserts agreement with the **CPU reference (the gold source)** within
  tolerance. Backends absent from the machine are skipped, so the same test runs
  everywhere. Verified on an RTX 3090 with `["cpu", "cuda", "wgpu"]` across
  L2/Cosine/IP/L1/Hamming/Jaccard.
- **Native ingest orchestrator (A4)**: `Table::ingest_async` /
  `table.ingest(paths, chunk_rows, parallelism, index_all, resume)` plans
  parquet
  inputs into row-range work units, runs a bounded
  `buffer_unordered(parallelism)`
  pool where each worker builds a *private* segment (data + indexes)
  concurrently,
  and commits completed segments through the OCC manifest CAS. Completed units
  are
  recorded in a `_ingest_state.json` sidecar so an interrupted load resumes at
  the
  unit boundary. Returns a report (`units_total/skipped/committed`,
  `rows_ingested`, `segments`). `IngestOptions::compact_after` runs
  `rewrite_data_files` at the end so segment/manifest counts stay bounded.
  CLI: `bsdb table ingest --uri … --input … [--plan] [--chunk-rows N]
  [--parallelism N] [--index-all] [--compact]`, with `--row-start/--row-end` as
  the serverless thin-runner mode (`Table::ingest_range_async`).
  **Multi-format inputs**: `.parquet` (row-range units) plus `.csv`, `.json` /
  `.ndjson`, and `.arrow` / `.ipc` / `.feather` (one unit per file, streamed
  whole-file with schema inference) — all via Arrow readers already in the tree.
- **Ingest memory discipline**: `core::memory::HeapTrimPolicy` returns freed
  heap pages to the OS (`malloc_trim`) at work-unit boundaries once RSS exceeds
  a budget, so long-lived in-process loads no longer ratchet toward the sum of
  every glibc arena's high-water mark. Wired through
  `IngestOptions::memory_budget_bytes`, the `BSDB_INGEST_MEMORY_BUDGET_GB` env
  var, `bsdb table ingest --memory-budget-gb`, and
  `table.ingest(..., memory_budget_gb=…)`. Allocator evaluation recorded in
  `core::memory`: glibc + `malloc_trim` chosen; jemalloc deferred; mimalloc
  rejected (static-TLS under pyo3).
- **Multi-machine ingest (A4)**: `WorkCoordinator` trait +
  `ObjectStoreCoordinator`
  backend (`core::table::coordinator`) — lease-based work stealing over the
  object store, reusing `FileBasedLock` (CAS claim + heartbeat + expiry-steal).
  `Table::ingest_coordinated_async` claims units dynamically (build in parallel,
  commit serially via the OCC CAS, release-on-failure so another node retries);
  a dead node's lease expires and its unit is stolen. Surfaces:
  `bsdb table ingest --coordinate [--lease-ttl-secs N]` and
  `table.ingest(..., coordinate=True, lease_ttl_secs=300)`. No broker, no etcd,
  no Raft cluster.

### Removed
- **Dead OpenCL kernels** (`src/core/index/opencl/*.cl`) — no `OpenClBackend`
  was ever wired into `ComputeBackend`.

### Performance
- **IVF clustering is now balanced (k-means++ seeding + capacity cap).** The
  index build slowed down over successive ingest chunks because `simple_kmeans`
  seeded centroids at random, leaving dense regions uncovered: one 750k-row
  segment produced 173 clusters where the largest held ~25x the mean (604 KB vs
  443 B of row-id mappings). Since the per-bucket HNSW build is superlinear in
  cluster size, those few oversized clusters dominated Pass 3. Centroids are now
  seeded with **k-means++ (D² sampling)** and the final assignment is
  **capacity-capped** at 1.5x the mean, spilling overflow points to their
  next-nearest cluster. Regression test
  `test_kmeans_clusters_are_balanced_on_skewed_data` asserts no cluster exceeds
  the cap on a deliberately skewed dataset.
- **Out-of-core HNSW-IVF build is now parallel and skips a full file re-scan.**
  Two changes to `HnswIvfIndex::build_from_file`:
  1. The per-bucket HNSW graphs (Pass 3) were built in a **sequential** loop —
     the dominant cost of a large index build. Buckets are independent and
     bounded in size, so they now build with `rayon` (`into_par_iter`), using
     all cores.
  2. The temp vector file written by `build_vector_index` now carries a
     self-describing header (`magic` + `dim`); the builder derives the vector
     count from the file length instead of re-reading the whole file just to
     count vectors (a multi-GB read per segment). Legacy header-less files still
     work via the old full-scan path.
  Tuning knobs (no rebuild): `BSDB_HNSW_N_LISTS`, `BSDB_HNSW_M`,
  `BSDB_HNSW_EF_CONSTRUCTION`.

## [0.9.0] - 2026-09-23

### Fixed
- **Statistics pruning was inert — four independent faults, all fixed.**
  Per-column
  min/max never reached the planner, so only partition pruning did any work:
  1. Parquet statistics were **disabled** (`EnabledStatistics::None`) → now
  `Chunk`.
  2. The Avro manifest writer hardcoded `lower_bounds` / `upper_bounds` /
     `null_value_counts` to `Null`, and the reader rebuilds `column_stats` from
     exactly those three → added `encode_iceberg_value` (mirror of the existing
     `decode_iceberg_value`) and `bounds_avro_values`.
  3. The reader never unwrapped Avro's nullable-union wrapper:
  `parse_map_int_long`
     / `parse_map_int_bytes` matched a bare `Array`, but a `["null", T]` field
     decodes as `Union(1, Box(Array(..)))`, so every stats map silently parsed
     as
     `None` → `unwrap_nullable_union`.
  4. `BETWEEN` produced no `QueryFilter` at all (DataFusion does not lower it to
     `>= AND <=`) → added an `Expr::Between` arm.
  Result: `id >= 100` prunes 5/5 segments ("column max < filter min"),
  `id = 15` prunes 4/5, `id >= 40` keeps exactly the matching segment; query
  results unchanged.
- **Binary column statistics pruned matching rows** — the manifest writer
  encodes bounds as raw bytes (`encode_iceberg_value`), but the reader decoded
  `binary`/`fixed` bounds unconditionally to **base64**, so a `blob` column's
  stats came back as `"YmFy"`/`"cXV4"` while the filter literal was `"foo"`.
  `"cXV4" < "foo"` therefore held and the segment was pruned with
  `StatsBelowMin`, dropping the matching row (surfaced by
  `tests/all_types_index_test.rs`). `decode_iceberg_value` now treats
  `binary`/`fixed` symmetrically with `string` — UTF-8 when valid, base64 only
  as a fallback for non-UTF-8 bytes.
- **`Device.auto_detect()` raised instead of falling back** when
  `torch.cuda.is_available()` reported `True` but no usable device existed (no
  GPU, or a missing `libnvrtc`): constructing the CUDA/ROCm/Intel device now
  falls through to native probing and finally CPU rather than propagating the
  error. Previously this failed every GPU-context test on a GPU-less host with
  a CUDA-enabled torch build.
- **AWS Glue `metadata_location` is now authoritative**: Glue is the only
  catalog whose commit API cannot return a metadata location (REST/Nessie
  return it; Hive/JDBC set it directly), so the client must supply one. The
  code reconstructed it from the snapshot's `sequence-number`, which only
  matches the metadata version because `write.rs` happens to set those two
  counters equal. The writer now captures the path
  `TableMetadata::save_to_store`
  returns and sends an explicit `set-metadata-location` update; Glue prefers it,
  falls back to the old derivation with a warning, and warns rather than
  silently no-op'ing when neither is available.

### Added
- **`vector_search_scored` — ids and scores with no Parquet I/O**: new Python
  method returning `(segment_id, row_id, score)` straight from the HNSW/BM25
  index, for seed discovery, RRF fusion, and candidate reranking. Fetch rows
  only for the winners. (`Table::execute_vector_search_as_scored` and
  `HybridReader::vector_search_index_raw` already existed underneath.)
- **`EXPLAIN` now says *why* segments were pruned**:
  `QueryPlanner::might_match_condition`
  gained `classify_condition(entry, filter, emit_metrics)`, returning a
  `PruneReason` (partition below-min / above-max / not-in-IN-list, stats
  all-null / below-min / above-max / not-in-IN-list). `explain()` prints a
  ranked breakdown instead of a bare count, and suppresses metric emission so
  diagnostic EXPLAINs don't inflate the operational pruning counters.
- **Sparse query vectors as SQL maps**: the vector-search sort-expr parser now
  accepts a sparse query as a `Map<key, f32>` literal
  (`{'1': 0.5, '10': 0.3}`), in addition to the existing `Struct` form
  (`indices`/`values`/`dim`). Keys may be integer or numeric-string typed;
  entries are sorted by index and de-duplicated. A map cannot carry the vector
  dimension, so it is inferred as `max(index) + 1` — use the Struct form to
  state the true dimension.
- **OR-over-ranges pushdown (A1.6)**: `TableProvider::scan` recognises a
  same-column disjunction of ranges — `(id BETWEEN 1 AND 5) OR (id BETWEEN 50
  AND 55)`, including the lowered `a >= x AND a <= y` form — and unions the
  per-range index bitmaps to skip segments that cannot match before reading
  them. Mixed-column disjunctions are left to DataFusion, and pruning only
  happens when the column is indexed and every range yields a bitmap.
- **Row-value IN-list pushdown on the read path (A1.7)**: new
  `Table::read_pk_filter_async`, plus a `TableProvider::scan` hook, push
  `(c1, c2) IN ((..), (..))` down as a single expression and use the per-column
  inverted indexes to prune non-matching segments. The guard requires *every*
  PK column to be indexed, since a partial index would under-count matches.
- **Sparse vector Arrow IPC serialization (A1.9)**: the `ArrowType` trait was a
  zero-copy `as_bytes(&[Self]) -> &[u8]`, which a variable-length `SparseVector`
  cannot satisfy (its elements live in separate allocations). It is now an owned
  `to_bytes`/`from_bytes` pair with a self-describing little-endian encoding for
  sparse vectors; fixed-width `f32`/`u8` still `bytemuck`-cast.
  `ArrowHnsw::get_vector` returns an owned `Vec<T>`.

### Fixed
- **Score-only vector search failed / read the whole row**: projecting only the
  synthesised `distance` column built an *empty* Parquet projection and failed
  with "must either specify a row count or at least one column". An empty
  projection is now the signal to skip Parquet entirely — the reader emits the
  score straight from the index search using an explicit row count, and
  `fetch_results_by_id` short-circuits the same way. Removed a stray `println!`
  from the HNSW chunk-search hot path.
- **Vector-search column projection was ignored**:
  `HybridReader::read_rows_by_id`
  took a `columns` parameter but never used it (`_columns`), so a vector search
  projecting two columns still read *every* column of the row from Parquet. It
  now builds a projected schema and reads only what was asked for, falling back
  to the full schema when a requested name is unknown.

### Removed
- Dead code: `src/core/planner/filter.rs` and
  `src/core/planner/vector_search.rs`.
  Neither was declared as a module (`planner.rs` has no `mod filter;` /
  `mod vector_search;`) and nothing referenced them, so they were orphaned
  duplicates of the live `QueryFilter` / `VectorSearchParams` in `planner.rs`.
  Removing them is a no-op for behaviour.
- **MVCC manifest commits (lock-free)**: the global `commit.lock` is gone from
  `update_schema` — schema/index-spec evolution is now pure optimistic
  concurrency (`PutMode::Create` + rebase).
  `CommitMetadata::skip_missing_remove_paths`
  lets a writer rebase onto a newer snapshot when a candidate file was
  concurrently removed (compaction uses it, so racing compactions no longer
  abort). New `Table::snapshot_version()` exposes the monotonic snapshot id
  (also on the Python `Table`). Metrics:
  `benostreamdb_manifest_commit_rebases_total`,
  `benostreamdb_manifest_commit_skipped_removals_total`.
- **Cross-partition compaction**: `PartitionSpec::partition_batch` now applies
  the declared Iceberg transform (`identity`/`void`/`bucket`/`truncate`/`year`/
  `month`/`day`/`hour`) via the canonical `IcebergTransform`, so a merged bin
  re-partitions deterministically. `CompactionOptions::allow_cross_partition`
  (default `true`) controls whether bins may span partitions.

### Fixed
- **`add_primary_key` / `drop_primary_key` panicked from Python** with
  "Cannot start a runtime from within a runtime": the async
  `_validate_pk_uniqueness` called the sync `read_with_columns`, which
  re-enters the Tokio runtime. It now uses the async read path.
- **Index joins returned no rows for non-int/string keys**:
  `extract_distinct_values` only handled `Int32`/`Int64`/`Utf8`, so a join on
  a date, timestamp, float, boolean, or `large_string` key produced an empty
  filter. It now derives keys via `ManifestValue::from_array` (all scalar
  types).
- **`SparseVector(indices, values, dim)` rejected plain Python lists** despite
  documenting them; it now accepts lists and NumPy arrays
  (`PyArrayLike1` + `AllowTypeChange`).
- **Partition values were the raw source value, not the transform result**:
  `partition_batch` ignored `PartitionField::transform`, so `bucket`/`truncate`/
  time partitions stored the untransformed value. Now routed through
  `IcebergTransform` (which also gained `bucket(N)`/`truncate(W)` parenthesis
  syntax and `large_string`/`Utf8View` support).
- **Manifest Avro partition type was derived from the source column**: a
  `bucket` transform yields an `int`, but the writer declared the source type
  (e.g. `string`), so the commit failed with an Avro type mismatch. The type is
  now derived from the transform.
- **Compaction dropped the Hive partition directory**: compacted files were
  written to the table root instead of `col=value/`, silently losing the
  physical partitioning. Compaction now preserves the partition path.
- **`ManifestValue::from_array` ignored `LargeUtf8`**: `large_string` partition
  columns (PyArrow's default) silently became `null`. Also added `Utf8View`,
  `Date32`, `Date64`, and `Timestamp` handling.

### Performance
- **Chunked-ingest memory is now a knob, not a wall**: demo load defaults to
  fresh-process 2M-row chunks (measured 9.5 GB peak per chunk, 2M rows/118 s);
  `--load-chunk-rows 500000` fits serverless tasks (~3-4 GB). ROADMAP gained
  the scale-out items this implies (serverless chunk orchestrator on the OCC
  manifest CAS; long-lived-process allocator discipline).
- **Out-of-core vector serving**: `DiskCache::get_mmap` now applies
  `MADV_RANDOM` to serving mmaps (HNSW graph/vectors, CSR offsets/edges/dict).
  Combined with the existing `use_mmap` default and TQ4 quantization, whole-site
  dense search runs on memory-constrained hosts with a bounded resident set.
- **Demo prep is now laptop-feasible on RAM**: `prepare_demo.py` embed streams
  per row-group (was materializing the whole slice, ~9 GB/worker), resolve joins
  per edge-chunk against one broadcast title map (was ~47 GB), and load deletes
  each embedding shard after ingesting it (bounds peak disk). Embed runs on GPU;
  default model pivoted to `all-MiniLM-L6-v2` (384-d seed-index centroids,
  ~6,000 sent/s on an RTX 3090 → ~2.5 h for 51.8M pages; bge-large-1024 measured
  279 sent/s = 50 h, impractical for a whole-site seed index).

### Fixed
- **Allocator-stranded memory during index-heavy loads**: the whole-site node
  load (32 in-process HNSW-IVF/TQ8 builders) ratcheted RSS to 82 GB — builders
  free everything, but glibc strands freed small allocations in per-thread
  arenas (430 observed) *and* in the unreturnable interior of the main heap
  (~2.6 GB per million rows even with `MALLOC_ARENA_MAX=2`). Two-part fix:
  the Python extension now calls `mallopt(M_ARENA_MAX, 2)` at import (respects
  an explicit user value; no-op on musl/macOS), and the demo load writes the
  nodes table in **fresh-process 10M-row chunks** so the allocator high-water
  mark resets per chunk — peak RSS ~25 GB regardless of dataset size.
- **Embedding column type**: the demo pipeline wrote `LargeList` embeddings,
  which
  the vector index silently ignores (it matches `List`/`FixedSizeList` only) —
  switched to `FixedSizeList`, so the HNSW index actually builds.
- **Demo load OOM + data-loss hazards** (`prepare_demo.py`): stage `load` read
  each embedding shard with a whole-shard `pq.read_table` — a 37 GB single-shard
  run was OOM-killed by the kernel; it now streams shards in 250k-row batches
  (verified position-aligned across multi-shard/coprime-batch boundaries).
  Shard deletion moved to strictly AFTER table commit (an earlier delete-on-
  advance variant destroyed the only embedding copy). Stage `embed` now rotates
  5M-row shards (`BSDB_EMBED_SHARD_ROWS` override) so load can reclaim disk
  incrementally. Covered by `tests/python/test_prepare_demo.py`.

### Added
- **Two-level Graph RAG in the Wikipedia demo** (`examples/web_ui/app.py`): the
  local-search tab now reranks with a bitmap-filtered vector search — the PPR
  neighborhood becomes an `id IN (...)` RoaringBitmap predicate on the HNSW
  index (topology prunes, semantics orders), mirroring the seed-index → CSR
  expansion → filtered-rerank architecture validated at whole-site scale.
- **Roadmap**: LangChain + LlamaIndex connector detail (Active Roadmap §9,
  Phase 12): vector stores, Graph-RAG retrievers wrapping
  `graph_rag_search`/`drift_search`, and an edge-table property-graph store.
- **Wikipedia demo dataset pipeline** (`scripts/build_demo_dataset.py`):
  consumes
  the full dump parquets, resolves mixed curid/title edge endpoints to int64
  curids (parallel, memory-bounded), prunes to the largest connected component
  and
  a dense hub-centered subgraph **inside BenoStreamDB** (`connected_components`,
  `degree_centrality`, `subgraph`), and emits `data/demo_nodes.parquet` /
  `data/demo_edges.parquet` with integer node ids the CSR index requires.
  Optional `sentence-transformers` embeddings (default `BAAI/bge-large-en-v1.5`,
  1024-d) for the UI's semantic entity resolution.

### Fixed
- **Merge/upsert nested-runtime panic**: `Table::merge` no longer drives the
  synchronous
  `MergePlanner` entry points inside a running tokio runtime (fixes
  `Cannot start a runtime from within a runtime` in `test_compound_pk.py`);
  `runtime_block_on` additionally offloads to a dedicated thread when invoked
  from
  within a runtime context.
- **Merge-path performance**: one reused current-thread runtime per
  `MergePlanner`
  (was: a new runtime per async call), object-store client hoisted out of
  per-segment
  loops, streams/bloom checks drained in a single runtime turn, and key matching
  changed from O(source × segment) linear scan to a hash index.
- **Graph UDAFs were stubs**: implemented real BFS in `shortest_path` and
  `graph_neighbors`, union-find in the `connected_components` UDAF, and
  neighborhood
  Jaccard similarity in `jaccard_coefficient` (previously returned hardcoded
  values).
- **Missing `connecting_paths` UDF**: implemented and registered
  `ConnectingPathsUDF` (pairwise BFS paths between seed nodes).
- **Scalar-arg overwrite in multi-partition merges**: `hops`/`directed` in the
  `subgraph`/`graph_neighbors`/`connecting_paths` accumulators are now optional
  so
  empty partitions no longer clobber captured values during `merge_batch`.
- **Python graph API drift**: `subgraph`, `connecting_paths`, `shortest_path`,
  `graph_neighbors` wrappers now match the Rust bindings (CSR-index route when
  `graph_column` is given, SQL route otherwise; `hops`/`directed`/`max_depth`
  accepted); `drift_search` now passes keyword arguments to the binding.
- `subgraph()` now returns full edge rows (payload columns such as `weight`
  preserved) by joining the extracted edge set back against the table.
- Added `jinja2` to the `dev` extra so `test_dbt_macro_file_syntax` runs in CI.

### Performance
- **Frontier-based out-of-core graph traversal**: new
  `core::algorithms::frontier`
  executes BFS (level-synchronous, one hash join per hop over a deduplicated
  symmetric adjacency, parquet-materialized visited/frontier state with early
  exit) inside DataFusion. The `subgraph()` and `graph_neighbors()` bindings now
  route through it instead of the UDAF accumulator that buffered the entire edge
  set plus a `HashMap` adjacency in RAM (tens of GB at 100M+ edge scale). The
  UDAFs remain available for raw SQL use.
- **Connected components at scale**: rewrote `compute_connected_components` from
  textbook per-hop label propagation (O(diameter) rounds, two full edge joins
  per
  round) to **pointer jumping + edge contraction** with a single symmetric edge
  join per round and a cheap checksum convergence probe. Validated on the
  English-Wikipedia link graph (160.5M raw / 91.7M clean edges):
  `connected_components()` completes in **~137s** (was effectively intractable
  at this scale).
- **Label propagation**: pre-materializes a symmetric edge set so each round
  does
  one join instead of two; per-run unique temp directory; state-sized
  convergence.

### Fixed
- **Intermittent vector-index panic**: `ArrowHnsw::get_vector` casts an Arrow
  binary value slice to `&[f32]`; when the IPC buffer landed on a misaligned
  byte
  offset, `bytemuck::cast_slice` panicked with
  `TargetAlignmentGreaterAndInputNotAligned` (debug-only, order-dependent). The
  binary buffer is now repacked into an aligned buffer once at load when needed,
  keeping the search hot path zero-copy.

### Notes
- Evaluated `mimalloc` as a `#[global_allocator]` for the Python extension; it
  fails to import under glibc (`cannot allocate memory in static TLS block`) and
  was reverted. A future option is `ld_preload`/`GLIBC_TUNABLES` or a non-TLS
  allocator.

---

## [0.8.1] - 2026-09-19

### Added
- **Graph RAG (CSR Graph + Shared Algorithms)**:
  - Extracted shared graph algorithms into `src/core/algorithms/` (connected
    components, label propagation, topological sort).
  - Added CSR graph index (`src/core/index/csr_graph.rs`) and builder
    (`build_graph.rs`); registered `IndexAlgorithm::CsrGraph` across manifest,
    index_config, segment, and Python helpers.
  - Refactored graph UDFs (pagerank, personalized_pagerank, shortest_path,
    strongly_connected_components, connected_components, label_propagation,
    jaccard_coefficient) to use shared algorithms + CSR graph.
  - Added `drift_search` UDF and Python GraphAPI + drift_search bindings.
- **Vector Index Joins**:
  - Reworked the index-join optimizer and physical plan; added a vector search
    sort-expression parser; updated HNSW-IVF, GPU, and HNSW paths.
- **Raw Vector Search API**:
  - Added `execute_vector_search_raw_with_config` for raw vector search
    returning `ScoredResult` with segment and row IDs.
- **PrimaryKeyFilter Row-Value IN-List Pushdown**:
  - Added `PrimaryKeyFilter` type representing a set of candidate PK rows
    (row-value IN list).
  - `from_expr` supports `Expr::InList`, equality, AND of disjoint columns, and
    OR (row-value IN lists).
  - `from_batch` builds from a RecordBatch; `to_expr` converts back to a
    DataFusion Expr; `to_query_filter` yields a single-column IN-list
    QueryFilter for inverted-index pushdown.
  - Reworked `check_primary_key_uniqueness_async` to push the whole batch as a
    single expression.
- **Time32/Time64 Datatype Support**:
  - Safely ignore Time datatypes for in-memory vector indexing.

### Changed
- Reworked compaction, reader (filter/scan), segment, manifest, query, and
  cache; renamed `MergeMode::MergeOnWrite` to `CopyOnWrite`.
- Extended PyTable bindings and the Python package with graph and drift_search
  APIs.
- DRY up release workflow, add deps, and document tech debt.

### Fixed
- Fixed FFI `load_async_with_cache_key` call missing `use_mmap` argument
  (compile error with `java` feature).
- Fixed clippy lints: `manual_map`, `needless_borrow`, `new_without_default`,
  `len_zero`, `needless_borrows_for_generic_args`.

---

## [0.8.0] - 2026-09-16

### Added
- **Graph RAG & Graph Analytics**:
  - Added a comprehensive suite of graph UDFs: pagerank, personalized_pagerank,
    shortest_path, strongly_connected_components, connected_components,
    label_propagation, jaccard_coefficient, degree_centrality,
    preferential_attachment, subgraph, louvain_communities, modularity,
    clustering_coefficient, adamic_adar, connecting_paths, neighbors,
    resource_allocation, to_graphviz, and topological_sort.
  - Added a graph search handler to the `benostreamdb-search` gateway.
  - Added the Python Graph API and graph RAG pipeline bindings.
  - Added dbt graph macros and graph RAG edge-table documentation.
- **Arrow IPC Vector Index & Micro-Batch Streaming Ingest Buffer**:
  - Implemented the Arrow IPC vector index and a micro-batch streaming ingest
    buffer.
  - Reworked HNSW-IVF index construction and search.
- **Zero-Copy Arrow IPC vectorSearch FFI Bridge**:
  - Added a zero-copy Arrow IPC `vectorSearch` FFI bridge for Spark and Trino.
  - Added `vectorSearch` JNI bindings to the Java connectors.
- **Python CLI for Background Services**:
  - Added Python CLI commands for installing and uninstalling background
    services.
  - Added a universal installer/uninstaller and a centralized configuration file
    for background services.
  - Added native background service configurations for `benostream-search`.

### Changed
- Reworked vector search for correctness and SQL aggregate consistency.
- Updated the Trino and Spark connectors.

### Fixed
- Updated rustls to 0.23.45 to fix RUSTSEC-2026-0285.
- Various connector and CI/CD hotfixes.

### Build / CI
- DRY up the CI pipeline to dynamically install `[dev]` extras directly from the
  built wheel.
- Added networkx, cffi, and scikit-learn to CI test environments.
- Upgraded `actions/checkout` from v4 to v5 for Node.js 24 support.
- Added the Renovate workflow; removed Dependabot.

---

## [0.7.0] - 2026-09-10

### Added
- **Multi-Vector Search & Reciprocal Rank Fusion (RRF)**:
  - Coordinate multi-vector search queries across multiple vector columns (e.g.,
    `title_vec` and `body_vec`) using reciprocal rank fusion ($1 / (k +
    \text{rank} + 1)$).
  - DataFusion SQL physical plan optimization and optimizer pushdown for
    multi-vector expressions (`dist_l2(col1, ...) as dist1, dist_l2(col2, ...)
    as dist2`).
  - Added programmatic and SQL end-to-end integration tests in
    `tests/test_multi_vector_search.rs`.
- **Composite Scalar Roaring Bitmap Indexes
  (`IndexAlgorithm::CompositeBitmap`)**:
  - Multi-column point and range query acceleration using combined composite
    inverted bitmap indexes.
  - Virtual composite column naming and tokenization using exact `"identity"`
    tokenization to preserve multi-column key terms (`val1\0val2`).
  - Fluent table API `table.add_composite_index(name, columns)` and filter
    rewriting.
  - Added end-to-end integration tests in `tests/test_composite_index.rs`.
- **Apache Polaris & Lakekeeper Iceberg REST Catalog OAuth2 Client
  Credentials**:
  - Implemented standard `/v1/oauth/tokens` client credentials grant flow for
    Iceberg REST catalogs.
  - Added token caching with automatic expiry tracking and refresh within
    60-second window.
  - Injected `Authorization: Bearer <token>` across all REST catalog requests
    (`load_table`, `create_table`, `commit_table`).
- **Core Community TurboQuant™ (TQ4 / TQ8) Scalar Quantization**:
  - Polar Quantization (PQ) and TurboQuant 4-bit / 8-bit quantized HNSW vector
    index algorithms integrated into open-source core engine.
- **Production Hardening & Correctness Enhancements**:
  - **Vector Metric Propagation**: Dynamic metric parsing
    (`VectorMetric::from_str`) and propagation from `IndexAlgorithm` (`L2`,
    `Cosine`, `InnerProduct`, `L1`, `Hamming`, `Jaccard`) through index
    construction and Puffin/Parquet serialization.
  - **Global KNN Ordering**: Refactored `merge_and_rerank_vector_results` to
    guarantee monotonic ascending distance order across all returned batches
    without unordered `HashMap` bucketing.
  - **Metric Parity Test Suite**: Added `tests/test_vector_metrics_parity.rs`
    establishing 100% nearest-neighbor accuracy against exact brute force ground
    truth across all 6 metrics.
  - **Strict Query Execution Semantics**: Segment vector search failures now
    fail the query immediately with actionable diagnostics rather than silently
    omitting rows.
  - **Zero-Warning Standard**: Eliminated diagnostic `println!` statements in
    favor of structured `tracing::debug!`, resolved all clippy compiler warnings
    with `#![deny(warnings)]`.

---

## [0.6.0] - 2026-09-09

### Added
- **Multi-Protocol Gateway Ecosystem**:
  - **`benostreamdb-search` Service (`bsdb-search` binary)**: Dual Elasticsearch
    7.10 (Port 9200) and Qdrant (Port 6333) compatible REST APIs over
    BenoStreamDB tables.
    - Elasticsearch 7.10 API: Full document CRUD (`POST /{index}/_doc`), hybrid
      search (`POST /{index}/_search` with BM25 + HNSW kNN + Reciprocal Rank
      Fusion), index management (`PUT /{index}`, `POST /{index}/_refresh`), and
      cluster health (`GET /_cluster/health`).
    - Qdrant REST API: Collection management (`/collections/{name}`), point
      upsert/retrieval (`/collections/{name}/points`), and vector search
      (`/collections/{name}/points/search`).
    - Prometheus metrics exporter on `/metrics` (Port 9090).
  - **`benostreamdb-flight` Gateway Service**: Native Arrow Flight SQL gRPC
    gateway (Port 50051) enabling zero-copy analytics for DuckDB, Polars, Apache
    Spark, and JDBC/ODBC BI tools via standard Flight SQL/ADBC.
- **Multi-Flavor GPU Acceleration & Hardware Auto-Detection**:
  - Optional GPU acceleration exposed across `benostreamdb-search` and
    `benostreamdb-flight` via `cuda`, `wgpu`, `rocm`, `intel`, and `all-gpu`
    feature flags.
  - Runtime device selection via
    `BENOSEARCH_DEVICE=auto|cuda[:N]|rocm[:N]|intel[:N]|mps|cpu`.
  - Active compute backend (`compute` block) exposed in `GET /` cluster info and
    `GET /_cluster/stats`.
  - `docker/Dockerfile.gpu`: Multi-flavor GPU container image with CUDA 12
    runtime, NVRTC JIT compilation, and Vulkan/Mesa drivers for AMD Radeon and
    Intel Arc.
  - `docker/docker-compose.gpu.yml`: Compose GPU override with hardware
    reservations and device pass-through.
- **Iceberg Compaction Resilience & Index Recovery**:
  - `Table::recover_indexes_async(&self)` / `recover_indexes(&self)`: Re-indexes
    data files that are missing overlay index sidecars, recovering fast vector
    (HNSW) and keyword (BM25) search after external Iceberg tools (Spark
    `rewriteDataFiles`, Trino `OPTIMIZE`, PyIceberg) compact table data files.
- **Docker Container Infrastructure**:
  - `benostreamdb/quickstart:latest`: Single all-in-one developer container
    running ES 7.10 (9200), Qdrant (6333), and Flight SQL (50051).
  - `benostreamdb/search:latest`: Standalone production search microservice.
  - `benostreamdb/flight:latest`: Standalone production Arrow Flight SQL
    microservice.
  - `docker/docker-compose.quickstart.yml`: Single-command full stack with
    RustFS (S3), Project Nessie catalog, and BenoStreamDB.
  - `docker-compose.production.yml`: Production multi-container configuration
    with health checks and resource limits.
- Okapi BM25 keyword scoring (tunable `k1`/`b`) with an English analyzer in the
  core engine; public `keyword_search_index` API.
- Smart hybrid trigger fusing keyword (BM25) and vector (HNSW) results via
  reciprocal rank fusion (RRF, k=60).
- Background segment index builds with `wait_for_background_tasks_async` for
  deterministic refresh semantics.

## [0.5.3] - 2026-06-21

### Added
- Unified catalog table loading logic (`load_from_catalog`) across Glue, Hive,
  Nessie, and REST catalog wrappers in python bindings.
- Explicit commit-time flushing and background synchronization in integration
  tests (`test_turboquant_integration.py` and `test_wikipedia.py`).

### Changed
- Fixed remote path routing in `flush_async` to correctly construct relative
  remote paths for staged files instead of using local staging paths.
- Upload data files synchronously relative to commit operation to avoid
  read-after-write race conditions in remote catalogs.
- Fixed a missing file upload call in the background indexing task for vector
  indexes.

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
- `WriteAheadLog::append_fire_and_forget` — non-blocking WAL append that hands
  off batches to the WAL worker without blocking on `fdatasync`, eliminating
  ~800 ms write latency per call

### Changed
- CUDA is now an optional feature (no longer required for CI builds)
- FFI bounds hardened with panic safety guarantees
- Public APIs documented with rustdoc
- **`Table(index_all=False)` is now the default** (previously `True`). Automatic
  HNSW/BM25 index building on commit is now opt-in. This eliminates a silent
  15–18 s background build that previously fired on every `commit()` for any
  table containing a vector column. To restore the old behaviour: `Table(uri,
  index_all=True)` or `table.index_all = True`.
- **`Table(autocommit=False)` is now the default** (previously `True`). Writes
  accumulate in an in-memory buffer and must be explicitly committed with
  `table.commit()`. This eliminates unexpected auto-flush overhead during
  ingestion loops.
- `write_async` no longer auto-detects columns named `"embedding"` as implicit
  HNSW index targets. Only columns explicitly registered via `add_index()` or
  `set_index_columns()` are indexed.
- `manifest_manager.load_latest_full` replaced with `load_latest` in the
  `flush_async` hot path, eliminating two unnecessary full manifest scans per
  commit.
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
- **High-Density Storage Milestone**: TQ4 (4-bit) and TQ8 (8-bit) TurboQuant
  quantization
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
- Extracted read, write, builder, schema, and fluent APIs from monolithic table
  module
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
- Fully internalized `hnsw_rs` to `src/core/index/hnsw_rs` for crates.io
  compatibility
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
- Initial release of BenoStreamDB
- Serverless index-streaming database with overlay indexing
- Apache Iceberg V2/V3 compliance
- Persistent scalar (RoaringBitmap) and vector (HNSW) indexes
- Native SQL support via DataFusion
- Python bindings via PyO3
- Multi-backend GPU acceleration (CUDA, ROCm, Metal, Intel XPU)
- Fluent query API with method chaining
- pgvector-compatible SQL operators

---

[Unreleased]: https://github.com/benolabsai/benostreamdb/compare/v0.11.1...HEAD
[0.11.1]: https://github.com/benolabsai/benostreamdb/compare/v0.11.0...v0.11.1
[0.11.0]: https://github.com/benolabsai/benostreamdb/compare/v0.9.0...v0.11.0
[0.9.0]: https://github.com/benolabsai/benostreamdb/compare/v0.8.1...v0.9.0
[0.5.3]: https://github.com/benolabsai/benostreamdb/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/benolabsai/benostreamdb/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/benolabsai/benostreamdb/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/benolabsai/benostreamdb/compare/v0.4.1...v0.5.0
[0.4.0]: https://github.com/benolabsai/benostreamdb/compare/v0.3.3...v0.4.0
[0.3.3]: https://github.com/benolabsai/benostreamdb/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/benolabsai/benostreamdb/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/benolabsai/benostreamdb/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/benolabsai/benostreamdb/compare/v0.2.6...v0.3.0
[0.2.6]: https://github.com/benolabsai/benostreamdb/compare/v0.2.3...v0.2.6
[0.2.3]: https://github.com/benolabsai/benostreamdb/compare/v0.2.1...v0.2.3
[0.2.1]: https://github.com/benolabsai/benostreamdb/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/benolabsai/benostreamdb/compare/v0.1.12...v0.2.0
[0.1.12]: https://github.com/benolabsai/benostreamdb/compare/v0.1.9...v0.1.12
[0.1.9]: https://github.com/benolabsai/benostreamdb/compare/v0.1.8...v0.1.9
[0.1.8]: https://github.com/benolabsai/benostreamdb/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/benolabsai/benostreamdb/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/benolabsai/benostreamdb/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/benolabsai/benostreamdb/compare/v0.1.3...v0.1.5
[0.1.3]: https://github.com/benolabsai/benostreamdb/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/benolabsai/benostreamdb/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/benolabsai/benostreamdb/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/benolabsai/benostreamdb/releases/tag/v0.1.0
