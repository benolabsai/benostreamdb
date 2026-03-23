# Spark Connector

The HyperStreamDB Spark connector enables you to use Apache Spark for Batch Processing, ETL, and Structured Streaming.

## Building

The connector is a Maven project located in `spark-hyperstream`.

```bash
cd spark-hyperstream
mvn clean install -DskipTests
```

## Usage

### Batch Read

```scala
val df = spark.read
  .format("hyperstreamdb")
  .load("s3://my-data/tables/user_logs")

df.filter("user_id = 12345").show()
```

### Structured Streaming

HyperStreamDB supports streaming reads, picking up new segments as they are committed to the manifest.

```scala
val stream = spark.readStream
  .format("hyperstreamdb")
  .load("s3://my-data/tables/logs")

stream.writeStream
  .format("console")
  .start()
```

### Filter Pushdown

Similar to Trino, the Spark connector implements `SupportsPushDownFilters`. Spark SQL filters are translated and passed to the Rust Reader JNI, ensuring that only data matching the index is transferred back to the Spark Executor.
