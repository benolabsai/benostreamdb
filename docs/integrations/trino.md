# Trino Connector

The BenoStreamDB Trino connector allows you to query your datasets using distributed SQL. It implements the Trino SPI (Service Provider Interface) and delegates IO operations to the Rust core via JNI.

## Building

The connector is a standard Maven project.

```bash
cd trino-benostreamdb
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

### Graph Traversal

The connector forwards each scan to the engine's DataFusion session, so the
engine's graph traversal **table functions** are callable directly in `FROM`
clauses. Each follows the in-memory / out-of-core `GraphMode` pattern
(`auto` | `in_memory` | `out_of_core` | `cached`):

```sql
-- Nodes within 2 hops of node 101 (node, hop, seed)
SELECT * FROM graph_neighbors('edges', '101', 2, 'auto');
-- Shortest path between two nodes (node, hop)
SELECT * FROM graph_shortest_path('edges', 101, 205, 'auto');
-- Induced subgraph edges within 2 hops (source, target)
SELECT * FROM graph_subgraph('edges', '101,102', 2, 'auto');
```

Endpoint columns are auto-detected (`source`/`src`/… , `target`/`dst`/…); pass
trailing `source`, `target` string arguments to name them explicitly.

### Live Subscriptions (change feed)

The engine's `subscribe_events` table function is reachable through the same
pass-through path. It drains the next committed-change events for a table and
returns one row per event (`event_type` = `batch` | `commit`, `rows`):

```sql
SELECT * FROM subscribe_events('edges', 'weight > 0.5', 100, 1000);
```

The change feed is in-process, so a Trino worker only observes commits made by
writers in the same JVM.

### GPU Device

Set `benostream.gpu-device` in the catalog properties (`auto` | `cpu` |
`cuda[:N]` | `mps` | `intel` | `rocm`). The connector resolves it through the
same core mapping as every other surface. Note: Trino uses a **single
connector-wide device** — it does not distribute splits across multiple GPUs.

### Predicate Pushdown

The connector supports aggressive predicate pushdown. The query engine passes the `WHERE` clause to the Rust core, which uses:
*   **Inverted Indexes**: To prune segments and rows based on scalar columns (e.g., `severity`).
*   **Vector Indexes**: (Planned) To optimize `ORDER BY similarity(...)` queries.
