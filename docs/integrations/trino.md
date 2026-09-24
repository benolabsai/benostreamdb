# Trino Connector

The BenoStreamDB Trino connector allows you to query your datasets using distributed SQL. It implements the Trino SPI (Service Provider Interface) and delegates IO operations to the Rust core via JNI.

## Building

The connector is a standard Maven project.

```bash
cd trino-benostream
mvn clean install -DskipTests
```

This produces a plugin archive (ZIP) in `target/`.

## Installation

1.  **Extract Plugin**: Unzip the artifact into the Trino plugin directory on all nodes.
    ```bash
    mkdir -p /usr/lib/trino/plugin/benostream
    unzip trino-benostream-*-plugin.zip -d /usr/lib/trino/plugin/benostream
    ```

2.  **Configure Catalog**: Create a catalog properties file `etc/catalog/benostreamdb.properties`.
    ```properties
    connector.name=benostreamdb
    benostream.base-uri=s3://my-bucket/
    ```

## Usage

Once configured, you can query BenoStreamDB tables just like any other SQL table.

```sql
SELECT * FROM benostreamdb.default.logs
WHERE severity = 'ERROR' AND timestamp > NOW() - INTERVAL '1' DAY
```

### Predicate Pushdown

The connector supports aggressive predicate pushdown. The query engine passes the `WHERE` clause to the Rust core, which uses:
*   **Inverted Indexes**: To prune segments and rows based on scalar columns (e.g., `severity`).
*   **Vector Indexes**: (Planned) To optimize `ORDER BY similarity(...)` queries.
