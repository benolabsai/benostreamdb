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
The connector implements `SupportsPushDownFilters`. Filters are pushed directly down to the BenoStreamDB Rust reader via JNI, pruning unneeded partitions and row groups before data is transferred back to Spark executors.
