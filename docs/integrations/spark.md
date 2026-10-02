# Apache Spark Connector

The BenoStreamDB Spark connector provides universal vector search, secondary index pushdown, position delete writes (Merge-on-Read), and streaming capabilities for Apache Spark.

The connector wraps the Apache Iceberg Spark Table SPI while intercepting read scans and row-level write operations to leverage BenoStreamDB's native Rust index engine via JNI.

---

## Requirements

* **Apache Spark**: 3.5.x, 4.0.x, or 4.1.x
* **Java**: 17+
* **Scala**: 2.12 (Spark 3.5) / 2.13 (Spark 4.x)
* **BenoStreamDB Native Library**: `libbenostreamdb.so` (available on `java.library.path` or packaged in JAR)

---

## Building the Connector

The connector source resides in `spark-benostreamdb`. Use Maven to build the profile corresponding to your Spark version:

```bash
cd spark-benostreamdb

# Default (Spark 4.0)
mvn clean package -Pspark-4.0

# Spark 3.5
mvn clean package -Pspark-3.5

# Spark 4.1
mvn clean package -Pspark-4.1
```

---

## Configuration & Catalog Setup

Configure SparkSession to use BenoStreamDB's procedure catalog:

```scala
val spark = SparkSession.builder()
  .appName("BenoStreamSparkApp")
  .config("spark.sql.catalog.spark_catalog", "org.apache.iceberg.spark.SparkCatalog")
  .config("spark.sql.catalog.spark_catalog.type", "hadoop")
  .config("spark.sql.catalog.spark_catalog.warehouse", "s3://my-lakehouse/warehouse")
  .config("spark.sql.catalog.benostream", "com.benostreamdb.spark.BenoStreamProcedureCatalog")
  .getOrCreate()

// Register BenoStreamDB vector search and similarity UDFs
import com.benostreamdb.spark.functions.BenoStreamFunctions
BenoStreamFunctions.register(spark)
```

---

## Core Capabilities

### 1. Vector Search & Similarity Functions

BenoStreamDB provides native SQL functions and DataFrame extensions for high-dimensional vector search.

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

### 3. Secondary & Bitmap Index Filter Pushdown

`BenoStreamScanBuilder` implements `SupportsPushDownFilters` and `SupportsPushDownRequiredColumns`.

When queries contain equality (`=`) or membership (`IN (...)`) predicates on indexed columns, filters are intercepted and evaluated against BenoStreamDB's native RoaringBitmap and secondary indexes via JNI (`queryIndexIn`), pruning data files and row splits before Parquet data is read by Spark executors:

```scala
// Prunes unindexed Parquet splits using BenoStreamDB bitmap indexes
val activeUsers = spark.read
  .format("benostream")
  .load("s3://my-lakehouse/tables/users")
  .filter("user_id IN (1001, 1002, 1003)")
```

---

### 4. Index Lifecycle & Catalog Procedures

Manage BenoStreamDB indexes directly through Spark SQL stored procedures under `benostream.system`:

```sql
-- Add an index to an Iceberg column
CALL benostream.system.add_index('spark_catalog.default.users', 'embedding', 'vector');

-- Build offline index files for all segments
CALL benostream.system.build_index('spark_catalog.default.users');

-- Rebuild a specific index column
CALL benostream.system.rebuild_index('spark_catalog.default.users', 'embedding');

-- List all configured and built indexes
CALL benostream.system.show_indexes('spark_catalog.default.users');

-- Drop an index
CALL benostream.system.drop_index('spark_catalog.default.users', 'embedding', 'vector');

-- Compact segments and consolidate index metadata
CALL benostream.system.compact('spark_catalog.default.users');

-- Set Primary Key constraint for Merge-on-Read
CALL benostream.system.set_primary_key('spark_catalog.default.users', 'id');
```

---

### 5. Merge-on-Read & Position Delete Writes

The connector implements `SupportsRowLevelOperations` (`BenoStreamMergeBuilder` and `BenoStreamPositionDeltaWrite`). When executing `MERGE INTO`, `UPDATE`, or `DELETE`, Spark uses BenoStreamDB's primary key indexes for accelerated candidate identification, writing Iceberg position delete files with zero table duplication.

```sql
MERGE INTO spark_catalog.default.users target
USING new_updates source
ON target.id = source.id
WHEN MATCHED THEN UPDATE SET target.age = source.age
WHEN NOT MATCHED THEN INSERT *;
```

---

### 6. Dynamic GPU Task Binding

For distributed GPU clusters (NVIDIA CUDA, AMD ROCm, Apple Metal, Intel Arc/XPU), `GpuContextResolver` automatically resolves and binds the native thread GPU context per task:

1. Dynamic discovery from Spark `TaskContext.get().resources().get("gpu")`.
2. `CUDA_VISIBLE_DEVICES` environment variable mapping.
3. Fallback to `spark.benostream.gpu.device` or table properties.
