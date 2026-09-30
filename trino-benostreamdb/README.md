# BenoStreamDB Trino Connector

Universal Vector & Metadata Streaming Connector for Trino.

The BenoStreamDB Trino connector enables distributed SQL analytics and vector search over BenoStreamDB tables using Trino. It implements the Trino SPI (Service Provider Interface) and delegates IO and indexing operations to the native Rust engine via JNI.

## Requirements

- Trino 435+
- Java 17+ (or Java 21)
- BenoStreamDB core native library (`libbenostreamdb.so`) on `java.library.path`

## Building

```bash
cd trino-benostreamdb
mvn clean install -DskipTests
```

This generates the plugin ZIP archive in `target/`.

## Installation & Configuration

1. **Extract Plugin**:
   ```bash
   mkdir -p /usr/lib/trino/plugin/benostream
   unzip target/trino-*-SNAPSHOT.zip -d /usr/lib/trino/plugin/benostream
   ```

2. **Configure Catalog**:
   Sample catalog configurations are provided in [`etc/catalog/`](etc/catalog/):
   ```properties
   connector.name=benostreamdb
   benostream.base-uri=s3://my-bucket/
   benostream.s3.endpoint=http://rustfs:9000
   benostream.s3.access-key=${TRINO_S3_ACCESS_KEY}
   benostream.s3.secret-key=${TRINO_S3_SECRET_KEY}
   ```

3. **Local Docker Environment**:
   The [`etc/`](etc/) directory contains the complete Trino configuration (`config.properties`, `jvm.config`, `node.properties`, `catalog/`) for containerized testing.

## Querying

```sql
SELECT * FROM benostreamdb.default.events
WHERE severity = 'ERROR' AND timestamp > NOW() - INTERVAL '1' DAY;
```

### Predicate & Vector Pushdown
The connector translates Trino domain constraints directly into BenoStreamDB index queries (inverted scalar indexes and vector indexes), pruning data before decoding Arrow batches.
