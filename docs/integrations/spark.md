# Apache Spark Connector

The BenoStreamDB Spark connector is a **fully native** Apache Spark DataSource V2 connector: reads, writes, DDL, catalog metadata, and row-level operations (DELETE / UPDATE / MERGE) all execute inside the BenoStreamDB Rust engine via JNI and the Arrow C Data Interface. It provides universal vector search, secondary index pushdown, stored procedures, and streaming capabilities.

There is **no Iceberg runtime dependency** for Spark 4.x builds. (Spark 3.5 predates the native DSv2 `ProcedureCatalog`/`CALL` API, so the optional `spark-3.5` build links `iceberg-spark-runtime` in `provided` scope solely to expose stored procedures; tables, reads, writes, and row-level ops on 3.5 are native too.)

---

## Requirements

* **Apache Spark**: 3.5.x, 4.0.x, 4.1.x, or 4.2.x
* **Java**: 17 (required for Spark 3.5; supported on 4.x)
* **Scala**: 2.12 (Spark 3.5) / 2.13 (Spark 4.x) — one artifact per Scala binary covers every Spark minor in that line
* **BenoStreamDB Native Library**: `libbenostreamdb.so` (bundled in the JAR resources, or on `java.library.path`)

---

## Building the Connector

The connector source resides in `spark-benostreamdb`. Use Maven to build the profile corresponding to your Spark version:

```bash
cd spark-benostreamdb

# Default (Spark 4.0)
mvn clean package -Pspark-4.0

# Spark 3.5 (Scala 2.12)
mvn clean package -Pspark-3.5

# Spark 4.1
mvn clean package -Pspark-4.1

# Spark 4.2
mvn clean package -Pspark-4.2
```

---

## Configuration & Catalog Setup

Configure SparkSession with the native catalog. On Spark 4.x use `BenoStreamProcedureCatalog` (the native table catalog **plus** `system.*` stored procedures); on Spark 3.5 use `BenoStreamCatalog` for tables/DDL:

```scala
val spark = SparkSession.builder()
  .appName("BenoStreamSparkApp")
  // Spark 4.x: tables + namespaces + CALL in one catalog
  .config("spark.sql.catalog.benostream", "com.benostreamdb.spark.BenoStreamProcedureCatalog")
  .config("spark.sql.catalog.benostream.warehouse", "s3://my-lakehouse/warehouse")
  .getOrCreate()

// Register BenoStreamDB vector search and similarity UDFs
import com.benostreamdb.spark.functions.BenoStreamFunctions
BenoStreamFunctions.register(spark)
```

---

## Core Capabilities

### 1. Vector Search & Similarity Functions

BenoStreamDB provides native SQL functions and DataFrame extensions for high-dimensional vector search.

#### Catalog Functions (no registration required)

The functions below are exposed through the Spark DSv2 `FunctionCatalog` and resolve with their fully-qualified names from Spark SQL, DataFrames, and PySpark on both Spark 3.5 and 4.x — no `register()` call needed:

```sql
SHOW FUNCTIONS IN benostream.system;

SELECT id, title, benostream.system.cosine_distance(embedding, array(0.12, 0.45, -0.23)) AS dist
FROM benostream.default.documents
ORDER BY dist ASC
LIMIT 10;
```

Registering the session-level aliases (`BenoStreamFunctions.register(spark)`, shown below) lets you use the unqualified names. Note that pgvector operators (`<->`) are intentionally **not** part of Spark SQL — Spark's grammar rejects them and they would be inconsistent with other catalogs in the same session; use these function names instead (identical to the Trino/DataFusion spellings).

#### Spark SQL Functions

```sql
-- Cosine Distance (0.0 = identical, 1.0 = orthogonal)
SELECT id, title, cosine_distance(embedding, array(0.12, 0.45, -0.23)) as dist
FROM spark_catalog.default.documents
ORDER BY dist ASC
LIMIT 10;

-- Euclidean (L2) Distance
SELECT id, l2_distance(embedding, array(0.12, 0.45, -0.23)) as dist
FROM spark_catalog.default.documents;

-- Dot Product (Higher is more similar)
SELECT id, dot_product(embedding, array(0.12, 0.45, -0.23)) as score
FROM spark_catalog.default.documents;

-- Generic Vector Distance
SELECT id, vector_distance(embedding, array(0.12, 0.45, -0.23), 'cosine') as dist
FROM spark_catalog.default.documents;
```

#### Fluent DataFrame API

```scala
import com.benostreamdb.spark.implicits._

val query = Array(0.12, 0.45, -0.23)

// Retrieve Top-K nearest neighbors
val top10 = df.vectorSearch(
  vectorCol = "embedding",
  query = query,
  k = 10,
  metric = "cosine"
)

// Enrich DataFrame with distance column
val withDist = df.withVectorDistance(
  outputCol = "_distance",
  vectorCol = "embedding",
  query = query,
  metric = "cosine"
)
```

---

### 2. Sparse Vectors & Hybrid Search (SPLADE / BM25)

BenoStreamDB supports learned sparse vectors (such as SPLADE or BM25 term weights) represented as Arrow `Struct<indices: List<Int64>, values: List<Float32>>`:

```sql
-- Compute sparse dot product across token weights
SELECT id, sparse_dot_product(query_indices, query_values, doc_indices, doc_values) as score
FROM spark_catalog.default.search_corpus;

-- Score Blending: Combine dense embedding similarity with sparse lexical score
SELECT id, hybrid_score(dense_score, sparse_score, 0.7) as final_score
FROM ranked_candidates;

-- Reciprocal Rank Fusion (RRF) across multi-retrieval candidate lists
SELECT id, reciprocal_rank_fusion(array(dense_rank, sparse_rank), 60) as rrf_score
FROM candidates;
```

---

### 2b. Graph Traversal (table functions)

Graph walks are available directly in `FROM` clauses via the engine's DataFusion
table functions, reached through the pass-through reader
(`option("query", ...)`). Each follows the same in-memory / out-of-core
`GraphMode` pattern as the graph UDAFs (`auto` | `in_memory` | `out_of_core` |
`cached`). The pass-through reader registers the table as `t`:

```scala
spark.read.format("benostream")
  .option("path", "s3://my-lakehouse/tables/edges")
  .option("query", "SELECT * FROM graph_neighbors('t', '101', 2, 'auto')")
  .load()
```

```sql
-- Nodes within 2 hops of node 101 (node, hop, seed)
SELECT * FROM graph_neighbors('edges', '101', 2, 'auto');
-- Shortest path between two nodes (node, hop)
SELECT * FROM graph_shortest_path('edges', 101, 205, 'auto');
-- Every shortest path (path)
SELECT * FROM graph_all_shortest_paths('edges', 101, 205, 'auto');
-- Induced subgraph edges within 2 hops (source, target)
SELECT * FROM graph_subgraph('edges', '101,102', 2, 'auto');
-- Union of pairwise shortest paths between seeds (source, target)
SELECT * FROM graph_connecting_paths('edges', '101,205,309', 'auto');
```

Endpoint columns are auto-detected (`source`/`src`/`src_id`/`from`/`u` and
`target`/`dst`/`dst_id`/`to`/`v`); pass trailing `source`, `target` string
arguments to name them explicitly.

Declaring an edge table (`SET TBLPROPERTIES ('table_type'='edge', 'src_col'=…, 'dst_col'=…)`)
automatically configures the forward (source) and reverse (target) CSR graph indexes.

---

### 2c. Live Subscriptions (change feed)

`subscribe_events` is a DataFusion table function, so it is reachable from Spark
through the pass-through reader. It drains the next committed-change events for a
table (a bounded live tail) and returns one row per event:

```sql
-- event_type ('batch' | 'commit'), rows
SELECT * FROM subscribe_events('edges', 'weight > 0.5', 100, 1000);
```

Arguments: `table`, optional `filter` (SQL predicate), optional `max_events`
(default 100), optional `timeout_ms` (default 1000). The change feed is
**in-process**: a Spark job only observes commits made by writers in the same
JVM. For cross-process streaming use the Flight SQL subscription ticket.

---

### 3. Secondary & Bitmap Index Filter Pushdown

`BenoStreamScanBuilder` implements `SupportsPushDownFilters` and `SupportsPushDownRequiredColumns`.

Supported predicates (`=`, `<`, `<=`, `>`, `>=`, `IN (...)`, `IS [NOT] NULL`, `LIKE 'prefix%'`, `AND`/`OR`/`NOT`) are translated to SQL and pushed into the engine's DataFusion planner through `openQuery`, which applies the scalar/Bitmap, Bloom, and lexical indexes during planning — pruning data files and row splits before any Parquet is decoded by Spark executors:

```scala
// Prunes unindexed Parquet splits using BenoStreamDB bitmap indexes
val activeUsers = spark.read
  .format("benostream")
  .load("s3://my-lakehouse/tables/users")
  .filter("user_id IN (1001, 1002, 1003)")
```

---

### 4. Index Lifecycle & Catalog Procedures

On Spark 4.x, manage BenoStreamDB indexes and primary keys directly through native Spark stored procedures under `benostream.system` (the catalog above implements the DSv2 `ProcedureCatalog` API). Every procedure routes straight into the engine over JNI, so it applies to native BenoStreamDB tables as well as externally mapped ones. (`list_indexes`, `optimize`, and `compact` are aliases.)

```sql
-- Add an index to a column: algorithm is one of
--   hnsw | hnsw_pq | hnsw_tq4 | hnsw_tq8 | bm25 | bloom | bitmap |
--   composite_bitmap | csr_graph | json_path
-- (defaults to the TurboQuant-8 vector index)
CALL benostream.system.add_index('spark_catalog.default.users', 'embedding', 'hnsw');

-- Build offline index files for all segments
CALL benostream.system.build_index('spark_catalog.default.users');

-- Rebuild a specific index column (drops and re-creates it)
CALL benostream.system.rebuild_index('spark_catalog.default.users', 'embedding');

-- List all configured and built indexes (JSON)
CALL benostream.system.show_indexes('spark_catalog.default.users');

-- Drop an index
CALL benostream.system.drop_index('spark_catalog.default.users', 'embedding', 'hnsw');

-- Compact segments and consolidate index metadata
CALL benostream.system.compact('spark_catalog.default.users');

-- Set / drop the primary key used for Merge-on-Read row identity
CALL benostream.system.set_primary_key('spark_catalog.default.users', 'id');
CALL benostream.system.drop_primary_key('spark_catalog.default.users', 'id');

-- Regional DRIFT graph search (top-k node ids)
CALL benostream.system.regional_drift_search(
  'spark_catalog.default.edges', 'vector query text', array(101L, 205L), 5, 2, 2, 3, 'auto');
```

On Spark 3.5 the same procedures are reachable through the Iceberg `ProcedureCatalog` shim. On Spark 4.2+ the catalog also implements `listProcedures`, so `SHOW PROCEDURES IN benostream.system` lists them.

---

### 5. Row-Level Operations: DELETE / UPDATE / MERGE

The connector implements `SupportsRowLevelOperations` natively (`BenoStreamRowLevelOperation` extends `SupportsDelta`). Row identity is the table's **primary key** columns (`rowId()` / `requiredMetadataAttributes()`), which are ordinary table columns — so Spark resolves them against the target relation without metadata-column plumbing.

Execution maps directly onto the engine's primitives:

* matched `DELETE` / `UPDATE` (old image) → predicate deletes via JNI (`deleteRows`)
* `INSERT` / updated new rows → Arrow record batches appended via JNI (`appendBatch`)
* plain `DELETE FROM ... WHERE ...` (no subquery) → pushed down as a single `deleteRows` predicate through `SupportsDelete`

```sql
MERGE INTO benostream.default.users target
USING new_updates source
ON target.id = source.id
WHEN MATCHED AND source.age < 0 THEN DELETE
WHEN MATCHED THEN UPDATE SET age = source.age
WHEN NOT MATCHED THEN INSERT (id, age) VALUES (source.id, source.age);

UPDATE benostream.default.users SET age = 31 WHERE id = 7;
DELETE FROM benostream.default.users WHERE age < 0;
```

The full MERGE matrix (UPDATE / DELETE / INSERT clauses, plus standalone UPDATE/DELETE) is verified end-to-end on Spark 3.5.9 and 4.2.0 with whole-stage codegen enabled. Declare the primary key with `CALL benostream.system.set_primary_key('db.tbl', 'id')` (or the `primary_key` table property); when unset, the first column is used.

> Note: the engine does not currently honor Iceberg position-delete files (`commitPositionDeletes` is a placeholder), so deletes are applied as predicate removals at commit time rather than as delete-file overlays.

---

### 6. Dynamic GPU Task Binding

For distributed GPU clusters (NVIDIA CUDA, AMD ROCm, Apple Metal, Intel Arc/XPU), `GpuContextResolver` automatically resolves and binds the native thread GPU context per task:

1. Dynamic discovery from Spark `TaskContext.get().resources().get("gpu")`.
2. `CUDA_VISIBLE_DEVICES` environment variable mapping.
3. Fallback to `spark.benostream.gpu.device` or table properties.
