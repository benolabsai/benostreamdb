# BenoStreamDB Spark Connector

Universal Vector & Metadata Streaming Connector for Apache Spark.

The BenoStreamDB Spark connector enables you to use Apache Spark for high-throughput batch processing, ETL, vector search, and structured streaming over BenoStreamDB tables.

## Requirements

- Apache Spark 3.5.x, 4.0.x, or 4.1.x
- Java 17+ (or Java 21)
- Scala 2.12 / 2.13
- BenoStreamDB core native library (`libbenostreamdb.so`) on `java.library.path`

## Building

```bash
cd spark-benostreamdb
mvn clean install -DskipTests
```

To build against specific Spark profiles:
```bash
# Spark 4.0 (Default)
mvn clean package -Pspark-4.0

# Spark 3.5
mvn clean package -Pspark-3.5

# Spark 4.1
mvn clean package -Pspark-4.1
```

## Configuration & Usage

Add the JAR to Spark's classpath and configure the catalog:

```bash
spark-shell \
  --jars target/spark-hyperstream-0.1.0-SNAPSHOT.jar \
  --conf spark.sql.catalog.benostream=com.benostreamdb.spark.BenoStreamCatalog
```

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
Register the BenoStreamDB SQL functions:
```scala
import com.benostreamdb.spark.functions.BenoStreamFunctions
BenoStreamFunctions.register(spark)
```

Use in Spark SQL:
```sql
-- Compute Cosine Distance
SELECT id, cosine_distance(embedding, array(0.1, 0.2, 0.3)) as dist 
FROM my_table 
ORDER BY dist ASC 
LIMIT 10;

-- Compute L2 Distance
SELECT id, l2_distance(embedding, array(0.1, 0.2, 0.3)) as dist FROM my_table;

-- Compute Dot Product
SELECT id, dot_product(embedding, array(0.1, 0.2, 0.3)) as score FROM my_table;

-- Sparse Dot Product (SPLADE / BM25)
SELECT id, sparse_dot_product(query_indices, query_values, doc_indices, doc_values) FROM my_table;

-- Hybrid Score & Reciprocal Rank Fusion (RRF)
SELECT id, hybrid_score(dense_score, sparse_score, 0.7) as final_score FROM my_table;
SELECT id, reciprocal_rank_fusion(array(dense_rank, sparse_rank), 60) as rrf FROM my_table;
```

Fluent DataFrame API:
```scala
import com.benostreamdb.spark.implicits._

// Top-K Vector Search
val top10 = df.vectorSearch(vectorCol = "embedding", query = Array(0.1, 0.2, 0.3), k = 10, metric = "cosine")

// Add distance column
val withDist = df.withVectorDistance(outputCol = "_dist", vectorCol = "embedding", query = Array(0.1, 0.2, 0.3))
```

### Stored Procedures (`system.*`)
Configure `spark.sql.catalog.benostream=com.benostreamdb.spark.BenoStreamProcedureCatalog`:

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
