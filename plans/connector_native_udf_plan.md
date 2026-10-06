# BenoStreamDB Connector Architecture: Native Engine + UDF Exposure + Version Matrix

## Problem statement

Three interlocking requirements emerged while testing the Spark connector:

1. **Remove the Iceberg Java dependency.** The Spark connector currently wraps
   `IcebergSource`/`SparkTable`/`SparkCatalog`. This (a) couples the connector to
   a per-Spark-minor `iceberg-spark-runtime` jar, and (b) is a dead end: the
   row-level writer (`BenoStreamPositionDeltaWriter`) has **stubbed
   `insert()`/`update()`**, so MERGE INSERT/UPDATE silently drops rows. Reads
   delegate to Iceberg and use JNI only as a discarded hint.
2. **Spark version matrix.** Environment has Spark 3.5.9 (Scala 2.12) and 4.2.0
   (Scala 2.13). Support full 3.5 and 4 series.
3. **Expose the engine's UDFs** in both Trino and Spark — at minimum the vector
   distance UDFs and the graph UDAFs (and ideally the full DataFusion surface).

## Key facts

- The native engine already reads/writes Iceberg-format tables itself (Rust),
  with no Java. The **Trino connector proves the native model**: it reads via
  JNI sessions + Arrow C Data, writes via `appendBatch`/`deleteRows`/`mergeRows`,
  and runs **arbitrary DataFusion SQL** via `Table::sql()` (JNI `openQuery`).
- The engine's DataFusion session registers: `all_vector_udfs()` (dist_* +
  `<->`/`<=>` operators + KNN optimizer rule), `all_json_udfs()`, BM25 pushdown,
  and `all_graph_aggregates()` (~18 graph UDAFs).
- Spark JNI natives already exist (`listDataFiles`, `PartitionReader
  openSession/readBatch`); Spark-named mirrors for the Trino surface were just
  added in `src/core/ffi.rs` (getTableSchema/appendBatch/mergeRows/deleteRows/
  createTable/openQuery/readQueryBatch/closeQuery/list*/dropTable/closeSession).
- **Trino has no generic pass-through query mechanism** and connectors cannot
  register arbitrary scalar UDFs; it *does* support connector **table
  functions** (`ConnectorTableFunction`, SPI 468).
- **Spark 4.x supports DSv2 `TableFunction`** (`FunctionCatalog`); Spark 3.5
  does not, but supports relation options + registered Spark UDFs.

## Recommended architecture

### A. Native Spark connector (Iceberg-free)

Rewrite the Spark connector to mirror the Trino connector's native model:

- `BenoStreamTable`: holds `tableUri` + `StructType` (from `getTableSchema`);
  implements `SupportsRead`, `SupportsWrite`, `SupportsRowLevelOperations`.
- **Read**: `BenoStreamScanBuilder` translates required columns + Spark filters →
  a DataFusion SQL string; `BenoStreamScan` → one `InputPartition`;
  `BenoStreamPartitionReader` calls `openQuery`/`readQueryBatch` and decodes
  Arrow→catalyst (new `BenoStreamArrowUtils`, already scaffolded).
- **Write**: `BenoStreamWriteBuilder` → `BatchWrite` → `DataWriter` buffers rows,
  converts to Arrow, calls `appendBatch` (commit already done engine-side per batch).
- **Row-level ops**: DELETE → `deleteRows(filter)`; UPDATE/MERGE → `mergeRows`
  (engine upsert-by-PK). Replaces the stubbed DeltaWrite.
- `BenoStreamCatalog` implements `TableCatalog`/`SupportsNamespaces` via
  `listSchemas`/`listTables`/`getTableSchema`/`createTable`/`dropTable` (no
  `SparkCatalog`). `DefaultSource` returns the native table.
- Drop `iceberg-spark-runtime` from the pom; remove all `org.apache.iceberg.*`
  imports from the connector.

### B. UDF exposure

Two complementary mechanisms:

1. **Pass-through native SQL (full DataFusion surface) — Spark first.**
   `spark.read.format("benostream").option("query", "SELECT id, cosine_distance(...)
   FROM t ORDER BY embedding <-> [..] LIMIT 10").load()` runs engine SQL and
   returns rows. This gives *all* UDFs + operators + KNN/graph optimizer rules
   for free. Expose the same via a Spark 4 DSv2 `TableFunction` `bsdb_query(sql)`.
2. **Typed table functions (both engines) — the named UDFs.** Define fixed-schema
   table functions so users get IntelliSense-style calls without raw SQL:
   - `vector_search(table, column, query_vector, k, metric)` → `(row_id, score)`
   - `graph_neighbors(table, node, hops, graph_column)` → `(node, hop, ...)`
   - `drift_search` / `regional_drift`, `keyword_search` (BM25), `hybrid_search`
   - Trino: `ConnectorTableFunction` registered on the connector.
     Spark 4: DSv2 `TableFunction`; Spark 3.5: DataFrame API + registered scalar
     UDFs (`BenoStreamFunctions`) as the fallback.

Trino cannot do generic pass-through, so its UDF story is the typed table
functions (+ scalar distance operators where feasible). Spark gets both.

### C. Version matrix (answer to the question)

Once Iceberg is gone, the only real axis is **Scala**:
- **Spark 3.5.x** (incl. 3.5.9): one Scala 2.12 artifact (`spark-3.5` profile).
  Patch releases are binary-compatible; no per-patch builds.
- **Spark 4.0 / 4.1 / 4.2**: one Scala 2.13 artifact covers all of them (DSv2
  APIs are stable across 4.x; we no longer touch version-specific Iceberg).
  Add the `spark-4.2` profile (done) for compile/CI targeting.
- **Trino**: single SPI 468 build.
Net: 2 Spark artifacts (2.12, 2.13) + 1 Trino, not per-minor.

## Sequenced milestones

1. [Native read] Scan builder + partition reader + Arrow→catalyst; verify reads
   work with **no Iceberg on the classpath**.
2. [Native write] DataWriter → Arrow → appendBatch; verify INSERT/CTAS/overwrite.
3. [Native row-level] DELETE→deleteRows; UPDATE/MERGE→mergeRows; verify with
   value assertions (the thing the Iceberg wrapper silently broke).
4. [Native catalog/DDL] BenoStreamCatalog + DefaultSource; drop Iceberg dep.
5. [Spark UDFs] Pass-through `query` option + Spark 4 `bsdb_query` table function
   + typed table functions (vector_search, graph_neighbors, drift, keyword, hybrid).
6. [Trino UDFs] `ConnectorTableFunction` set (vector_search, graph_neighbors, …).
7. [Matrix/CI] Build 2.12 + 2.13 artifacts; add spark-4.2 to build matrix; docs.
8. [Regression] Re-run full surface suites for both connectors.

## Risks / open questions

- Native write must preserve Iceberg-format correctness (snapshots/manifests) —
  the engine already does this (Trino writes prove it), but Spark's partitioning/
  bucketing semantics need parity work.
- Trino table functions have fixed return schemas; vector/graph outputs must be
  modeled accordingly (row_id + score / node columns).
- `libbenostreamdb.so` must be on `java.library.path` on every Spark executor /
  Trino worker (already true for Trino).
- Scope is large; ordering below is a recommendation, not a commitment.
