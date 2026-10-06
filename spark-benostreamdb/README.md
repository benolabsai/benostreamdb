# BenoStreamDB Spark Connector

Universal Vector & Metadata Streaming Connector for Apache Spark.

The BenoStreamDB Spark connector is a **fully native** DataSource V2 connector: reads, writes, DDL, catalog metadata, row-level operations (DELETE / UPDATE / MERGE), and vector functions all execute through the BenoStreamDB Rust engine over JNI + the Arrow C Data Interface. Spark 4.x builds have **no Iceberg runtime dependency**; the Spark 3.5 build links `iceberg-spark-runtime` in `provided` scope only to expose stored procedures (Spark 3.5 predates the native DSv2 `ProcedureCatalog`/`CALL` API).

## Requirements

- Apache Spark 3.5.x, 4.0.x, 4.1.x, or 4.2.x
- Java 17 (required for Spark 3.5; supported on 4.x)
- Scala 2.12 (Spark 3.5) / 2.13 (Spark 4.0–4.2) — one artifact per Scala binary covers every Spark minor
- BenoStreamDB core native library (`libbenostreamdb.so`) bundled in the JAR resources or on `java.library.path`

## Building

```bash
cd spark-benostreamdb
mvn clean package -DskipTests
```

To build against specific Spark profiles:
```bash
# Spark 4.0 (Default)
mvn clean package -Pspark-4.0

# Spark 3.5 (Scala 2.12)
mvn clean package -Pspark-3.5

# Spark 4.1
mvn clean package -Pspark-4.1

# Spark 4.2
mvn clean package -Pspark-4.2
```

Always `clean` when switching profiles (Scala binary cross-contamination).

## Configuration & Usage

Add the JAR to Spark's classpath and configure the catalog:

```bash
spark-shell \
  --jars target/spark-benostream-0.12.0.jar \
  --conf spark.sql.catalog.benostream=com.benostreamdb.spark.BenoStreamCatalog \
  --conf spark.sql.catalog.benostream.warehouse=s3://my-bucket/warehouse
```

On Spark 4.x use `com.benostreamdb.spark.BenoStreamProcedureCatalog` to get tables, namespaces, `system.*` procedures, and catalog functions from a single catalog.

### Batch Read
```scala
val df = spark.read
  .format("benostream")
  .load("s3://my-bucket/tables/events")

df.filter("severity = 'ERROR'").show()
```

### Structured Streaming
```scala
val stream = spark.readStream
  .format("benostream")
  .load("s3://my-bucket/tables/events")

stream.writeStream
  .format("console")
  .start()
```

### Filter & Predicate Pushdown
The connector implements `SupportsPushDownFilters` and `SupportsPushDownRequiredColumns`. Filters matching indexed columns and primary keys are pushed directly down to the BenoStreamDB native engine via JNI, evaluating secondary indexes and pruning unneeded partitions and row groups before data is transferred back to Spark executors.

### Vector Search & Distance Functions

The distance/hybrid functions are exposed as **DSv2 catalog functions** — no registration needed, usable from SQL, DataFrames, and PySpark:

```sql
-- Compute Cosine Distance
SELECT id, benostream.system.cosine_distance(embedding, array(0.1, 0.2, 0.3)) as dist
FROM benostream.default.my_table
ORDER BY dist ASC
LIMIT 10;

-- Compute L2 Distance
SELECT id, benostream.system.l2_distance(embedding, array(0.1, 0.2, 0.3)) as dist FROM benostream.default.my_table;

-- Compute Dot Product
SELECT id, benostream.system.dot_product(embedding, array(0.1, 0.2, 0.3)) as score FROM benostream.default.my_table;

-- Sparse Dot Product (SPLADE / BM25)
SELECT benostream.system.sparse_dot_product(query_indices, query_values, doc_indices, doc_values) FROM t;

-- Hybrid Score & Reciprocal Rank Fusion (RRF)
SELECT benostream.system.hybrid_score(dense_score, sparse_score, 0.7) as final_score FROM t;
SELECT benostream.system.reciprocal_rank_fusion(array(dense_rank, sparse_rank), 60) as rrf;
```

`SHOW FUNCTIONS IN benostream.system` lists them. Scala/Java users can also register session-level aliases (`BenoStreamFunctions.register(spark)`) to use the unqualified names. pgvector operators (`<->`) are intentionally **not** supported in Spark SQL — Spark's grammar rejects them and they would confuse mixed-catalog sessions; use the function spelling above (same names as Trino/DataFusion).

Fluent DataFrame API:
```scala
import com.benostreamdb.spark.implicits._

// Top-K Vector Search
val top10 = df.vectorSearch(vectorCol = "embedding", query = Array(0.1, 0.2, 0.3), k = 10, metric = "cosine")

// Add distance column
val withDist = df.withVectorDistance(outputCol = "_dist", vectorCol = "embedding", query = Array(0.1, 0.2, 0.3))
```

### Row-Level Operations (DELETE / UPDATE / MERGE)

Native `SupportsRowLevelOperations` (`SupportsDelta`): row identity is the table's primary key (declare with `CALL benostream.system.set_primary_key(...)` or the `primary_key` property; defaults to the first column). Matched deletes/updates become predicate deletes in the engine; inserts and updated rows are appended as Arrow batches. The full MERGE matrix is verified on Spark 3.5.9 and 4.2.0 with whole-stage codegen enabled.

```sql
MERGE INTO benostream.default.users t
USING updates s ON t.id = s.id
WHEN MATCHED AND s.age < 0 THEN DELETE
WHEN MATCHED THEN UPDATE SET age = s.age
WHEN NOT MATCHED THEN INSERT (id, age) VALUES (s.id, s.age);
```

### Stored Procedures (`system.*`)

Spark 4.x: configure `spark.sql.catalog.benostream=com.benostreamdb.spark.BenoStreamProcedureCatalog` (native DSv2 procedures). Spark 3.5: procedures require `iceberg-spark-runtime` on the classpath plus `spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions` (Spark 3.5 has no native `CALL` support); table/row-level features remain fully native there.

```sql
-- Index Management
CALL benostream.system.add_index('my_table', 'embedding', 'vector');
CALL benostream.system.build_index('my_table');
CALL benostream.system.rebuild_index('my_table', 'embedding');
CALL benostream.system.drop_index('my_table', 'embedding', 'vector');
CALL benostream.system.show_indexes('my_table');

-- Table Maintenance
CALL benostream.system.compact('my_table');
CALL benostream.system.set_primary_key('my_table', 'id');
```

### Dynamic GPU Task Binding
When running on GPU-enabled Spark clusters, executor tasks automatically discover their assigned GPU device through Spark's `TaskContext` resource allocation or `CUDA_VISIBLE_DEVICES`, dynamically binding native GPU contexts per task.
