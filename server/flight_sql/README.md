# BenoStreamDB Arrow Flight & Flight SQL Gateway

`benostreamdb-flight` provides a high-performance native Apache Arrow Flight and Flight SQL interface over BenoStreamDB.

> **Architecture Note:** BenoStreamDB is **serverless-first**. Arrow Flight SQL is an optional server interface for network-accessible database access. Flight SQL currently provides **single-process server access**; distributed or clustered execution across multiple nodes is not currently supported or implied.
>
> **⚠️ Security:** This server has **no authentication, no authorization, and no TLS**. Run it on a **trusted internal network**, bound to `127.0.0.1` or an internal interface, behind a gateway/reverse proxy that terminates TLS and enforces authentication. Do **not** expose it directly to the public internet. See [SECURITY.md](../../SECURITY.md).

## Overview

Arrow Flight SQL is the standard, language-agnostic data transport and database protocol for analytical engines. It enables zero-copy, highly parallelized stream transfers of Arrow RecordBatches between clients and BenoStreamDB over gRPC/HTTP/2.

### Architecture

```
Client Application (JDBC / ODBC / ADBC / Go / C++ / BI tools)
                    │
                    ▼
         Arrow Flight SQL (gRPC / HTTP2)
                    │
                    ▼
         BenoStreamFlightSqlService
                    │
                    ▼
           BenoStreamSession
                    │
                    ▼
       DataFusion Query Engine + BenoStreamDB Core
```

> Applications with a native path — Rust (the `benostreamdb` crate), Python (the
> `benostreamdb` package), and JVM (the Spark/Trino connectors) — normally use
> those directly rather than going through the gateway. Flight SQL is for
> polyglot clients and BI tools that speak the protocol (JDBC/ODBC/ADBC, Go,
> C++, DBeaver, …).

## Features

- **Standard Flight SQL Protocol**: Supports `CommandStatementQuery`, `CommandPreparedStatementQuery`, and metadata introspection (`CommandGetTables`, `CommandGetCatalogs`, `CommandGetDbSchemas`, etc.).
- **Zero-Copy Arrow Transport**: Streams record batches directly into memory without row-wise serialization or deserialization penalties.
- **DataFusion Integration**: Direct query execution backed by DataFusion with vector search and indexing pushdowns.
- **Full DDL / maintenance surface**: `CREATE DATABASE`/`SCHEMA`/`TABLE`, index and primary-key DDL, schema evolution, compaction, vacuum, and `ALTER TABLE ... EXECUTE` procedures — all over the wire.
- **Ecosystem Compatibility**: Works out-of-the-box with any Flight SQL compliant client (DBeaver, JDBC/ODBC/ADBC drivers, DuckDB, Go, C++, BI tools). Applications with a native path (Python, Rust, Spark/Trino) should prefer those.

## Running the Server

```bash
cargo run -p benostreamdb-flight
```

### Configuration

| Variable | Purpose |
|---|---|
| `BSDB_WAREHOUSE` | Base location for `CREATE TABLE` when no `LOCATION` is given. Tables land at `<warehouse>/<schema>/<table>`. |
| `BSDB_CATALOG_NAME` | DataFusion catalog name to bind the external catalog to (default: the catalog type, e.g. `rest`). |
| `BSDB_CONFIG` | Path to a catalog config TOML (Nessie / REST / Glue / Hive / Unity / JDBC). Falls back to `./benostream.toml` then `~/.benostream/config.toml`. |
| `BSDB_METRICS_BIND` | Bind address for the observability HTTP listener. | `127.0.0.1` |
| `BSDB_METRICS_PORT` | Port for the observability HTTP listener. | `9090` |

Without a warehouse or catalog config, the server still serves queries against tables registered in-process, but `CREATE TABLE` requires either `BSDB_WAREHOUSE` or an explicit `LOCATION`.

### Observability

The server exposes an HTTP observability surface on
`BSDB_METRICS_BIND:BSDB_METRICS_PORT` (default `127.0.0.1:9090`):

| Endpoint | Purpose |
|---|---|
| `GET /metrics` | Prometheus text format — the engine's `benostreamdb_*` metrics. |
| `GET /health` | Liveness (`200 ok`). |
| `GET /readyz` | Readiness (`200 ok`). |

See [docs/monitoring.md](../../docs/monitoring.md) for the full metric catalog.

## Connecting with Python (ADBC Flight SQL)

```python
import adbc_driver_flightsql.dbapi as flight_sql

with flight_sql.connect(uri="grpc://localhost:50051") as conn:
    with conn.cursor() as cur:
        cur.execute("CREATE DATABASE IF NOT EXISTS mydb;")
        cur.execute("CREATE SCHEMA IF NOT EXISTS mydb.myschema;")
        cur.execute("CREATE TABLE mydb.myschema.t (id INT, name VARCHAR);")
        cur.execute("INSERT INTO mydb.myschema.t VALUES (1, 'Alice'), (2, 'Bob');")
        cur.execute("SELECT * FROM mydb.myschema.t;")
        print(cur.fetchall())
```

---

# SQL Language Guide

BenoStreamDB speaks a PostgreSQL-flavoured SQL dialect (DataFusion's parser with
`datafusion.sql_parser.dialect = PostgreSQL`), extended with Iceberg-style DDL
and maintenance. This section is the reference for everything the gateway
accepts.

## Naming model

The namespace hierarchy maps directly onto PostgreSQL:

| PostgreSQL | Iceberg | DataFusion |
|---|---|---|
| database | catalog | catalog |
| schema | namespace | schema |
| table | table | table |

So `mydb.myschema.t` means catalog `mydb`, namespace `myschema`, table `t`.
Unqualified names resolve against the session's default catalog/schema
(`datafusion.public`).

## Catalog and namespace

```sql
CREATE DATABASE mydb;
CREATE DATABASE IF NOT EXISTS mydb;
CREATE SCHEMA mydb.myschema;
CREATE SCHEMA IF NOT EXISTS mydb.myschema;

DROP SCHEMA mydb.myschema;
DROP DATABASE mydb;
```

- `CREATE DATABASE` registers a DataFusion catalog and binds the configured
  external catalog (if any) under that name.
- `CREATE SCHEMA` registers the DataFusion schema and calls the external
  catalog's `create_namespace`.
- `DROP DATABASE` is a best-effort no-op (DataFusion exposes no catalog
  deregistration).

## Tables

```sql
CREATE TABLE mydb.myschema.t (id INT, name VARCHAR);
CREATE TABLE mydb.myschema.t (id INT, name VARCHAR) LOCATION 's3://bucket/t';
CREATE TABLE mydb.myschema.t (id INT) WITH (format_version = 3);
CREATE TABLE mydb.myschema.t (id INT) WITH (sort_order = 'id');

DROP TABLE mydb.myschema.t;
DROP TABLE IF EXISTS mydb.myschema.t;
TRUNCATE TABLE mydb.myschema.t;
```

- The table is created as a real BenoStreamDB/Iceberg table (genesis manifest +
  metadata), not a DataFusion in-memory table.
- Location resolution order: explicit `LOCATION` → `BSDB_WAREHOUSE`-derived
  (`<warehouse>/<schema>/<table>`) → error.
- `WITH (format_version = N)` sets the Iceberg format version.
- `WITH (sort_order = 'col1, col2')` sets the write sort order.
- `DROP TABLE` deregisters from DataFusion **and** drops from the external
  catalog.
- `CREATE TABLE AS SELECT` is not supported.

Supported column types: `BOOLEAN`, `TINYINT`, `SMALLINT`, `INT`/`INTEGER`,
`BIGINT`, `FLOAT`/`REAL`, `DOUBLE`, `VARCHAR`/`TEXT`/`STRING`/`CHAR`, `BINARY`,
`DATE`, `TIMESTAMP`. Unknown types fall back to `VARCHAR`.

## Indexes

```sql
CREATE INDEX ON mydb.myschema.t (id);
CREATE INDEX ON mydb.myschema.t (id, name);          -- composite
ALTER TABLE mydb.myschema.t ADD INDEX (name);
ALTER TABLE mydb.myschema.t DROP INDEX id;
```

- A single-column index defaults to a **Bitmap** index (the closest analogue to
  a B-tree for scalar columns).
- A multi-column `CREATE INDEX` builds a composite roaring-bitmap index.
- `DROP INDEX <column>` removes the index for that column.

## Primary keys

```sql
ALTER TABLE mydb.myschema.t ADD PRIMARY KEY (id);
ALTER TABLE mydb.myschema.t ADD PRIMARY KEY (id, name);   -- composite
ALTER TABLE mydb.myschema.t DROP PRIMARY KEY;
```

Adding a primary key marks the columns `NOT NULL` and commits a new manifest
version. `DROP PRIMARY KEY` clears the key.

## Schema evolution

```sql
ALTER TABLE mydb.myschema.t ADD COLUMN age INT;
ALTER TABLE mydb.myschema.t DROP COLUMN age;
ALTER TABLE mydb.myschema.t RENAME COLUMN name TO full_name;
ALTER TABLE mydb.myschema.t ALTER COLUMN id TYPE BIGINT;
```

## Data

```sql
INSERT INTO mydb.myschema.t VALUES (1, 'Alice'), (2, 'Bob');
INSERT INTO mydb.myschema.t SELECT * FROM staging;

SELECT * FROM mydb.myschema.t WHERE id = 1;
DELETE FROM mydb.myschema.t WHERE id = 1;

MERGE INTO mydb.myschema.t AS t
USING staging AS s ON t.id = s.id
WHEN MATCHED THEN UPDATE SET name = s.name
WHEN NOT MATCHED THEN INSERT (id, name) VALUES (s.id, s.name);
```

- `INSERT` writes and commits the rows, so they are immediately visible.
- `DELETE FROM` is intercepted and routed to the table's delete primitive.
- `MERGE INTO` is intercepted and executed via the key-based merge primitive.

## Maintenance

```sql
OPTIMIZE TABLE mydb.myschema.t;   -- compaction (rewrite data files)
COMPACT mydb.myschema.t;          -- alias for OPTIMIZE
VACUUM mydb.myschema.t;           -- reclaim unreferenced files
MSCK REPAIR TABLE mydb.myschema.t; -- rebuild sidecars after an external engine
```

`OPTIMIZE`/`COMPACT` merge small files; `VACUUM` reclaims unreferenced data and
manifest files; `MSCK REPAIR TABLE` recovers indexes for segments rewritten by
an external engine (Spark/Trino).

## `ALTER TABLE ... EXECUTE` procedures

Iceberg/Spark-style maintenance procedures:

```sql
ALTER TABLE mydb.myschema.t EXECUTE verify_integrity;
ALTER TABLE mydb.myschema.t EXECUTE recover_indexes;
ALTER TABLE mydb.myschema.t EXECUTE checkpoint;
ALTER TABLE mydb.myschema.t EXECUTE rewrite_data_files;
ALTER TABLE mydb.myschema.t EXECUTE remove_orphan_files(older_than_ms => 0);
ALTER TABLE mydb.myschema.t EXECUTE expire_snapshots(retention => 2);
ALTER TABLE mydb.myschema.t EXECUTE rollback(snapshot_id => 3);
ALTER TABLE mydb.myschema.t EXECUTE preload_indexes;
```

| Action | Effect |
|---|---|
| `verify_integrity` | Verify file checksums against the manifest |
| `recover_indexes` | Rebuild sidecars for externally-rewritten segments |
| `checkpoint` | Compact the write-ahead log |
| `rewrite_data_files` / `compact` | Compaction |
| `remove_orphan_files(older_than_ms => N)` | Delete unreferenced files |
| `expire_snapshots(retention => N)` | Vacuum with a retention window |
| `rollback(snapshot_id => N)` | Roll back to a snapshot |
| `preload_indexes` | Warm the index cache |

## Session settings

```sql
SET benostream.warehouse = 's3://bucket/warehouse';
SHOW benostream.warehouse;
```

Session-scoped and allowlisted (`warehouse`, `default_index_algorithm`). SQL
`SET` **never** writes process environment variables — settings live on the
session only, so one client cannot affect another. Unknown keys are rejected.

## Vector search and UDFs

Every UDF, aggregate, and operator registered on the session is available over
Flight SQL. Vector literals use `ARRAY[...]` (or `'[...]'::vector`).

### pgvector operators

Rewritten to the corresponding UDF before planning. The left-hand side must be
a column or function call (not an `ARRAY` literal).

| Operator | UDF | Meaning |
|---|---|---|
| `<->` | `dist_l2` | L2 / Euclidean distance |
| `<=>` | `dist_cosine` | Cosine distance |
| `<#>` | `dist_ip` | Inner-product distance (negative dot product) |
| `<+>` | `dist_l1` | L1 / Manhattan distance |
| `<~>` | `dist_hamming` | Hamming distance |
| `<%>` | `dist_jaccard` | Jaccard distance |

```sql
SELECT * FROM docs WHERE embedding <-> ARRAY[1.0, 0.0, 0.0] < 2.0;
SELECT * FROM docs ORDER BY embedding <=> ARRAY[1.0, 0.0, 0.0] LIMIT 10;
```

### Vector scalar functions

| Function | Purpose |
|---|---|
| `dist_l2(a, b)` | L2 / Euclidean distance |
| `dist_cosine(a, b)` | Cosine distance |
| `dist_ip(a, b)` | Inner-product distance |
| `dist_l1(a, b)` | L1 / Manhattan distance |
| `dist_hamming(a, b)` | Hamming distance |
| `dist_jaccard(a, b)` | Jaccard distance |
| `vector_add(a, b)` | Element-wise addition |
| `vector_sub(a, b)` | Element-wise subtraction |
| `vector_mul(a, b)` | Element-wise multiplication |
| `vector_concat(a, b)` | Concatenate two vectors |
| `vector_dims(v)` | Number of dimensions |
| `vector_norm(v)` | L2 norm |
| `l2_normalize(v)` | Unit-normalize a vector |
| `binary_quantize(v)` | Quantize to a binary vector |
| `subvector(v, start, len)` | Slice a sub-vector |
| `vector_to_sparse(v)` | Dense → sparse representation |
| `sparse_to_vector(s)` | Sparse → dense representation |
| `vector_to_binary(v)` | Dense → bit-packed binary |

```sql
SELECT dist_l2(embedding, ARRAY[1.0, 0.0, 0.0]) AS distance FROM docs;
SELECT vector_dims(embedding), vector_norm(embedding) FROM docs;
SELECT l2_normalize(embedding) FROM docs;
```

### Vector aggregates

| Aggregate | Purpose |
|---|---|
| `vector_sum(v)` | Element-wise sum over a group |
| `vector_avg(v)` | Element-wise mean over a group |

```sql
SELECT vector_avg(embedding) FROM docs GROUP BY category;
```

### Graph aggregates

| Aggregate | Purpose |
|---|---|
| `label_propagation(...)` | Community detection by label propagation |
| `graph_neighbors(...)` | Neighbors within N hops |
| `subgraph(...)` | Induced subgraph around seeds |
| `connecting_paths(...)` | Pairwise shortest connecting paths |
| `shortest_path(...)` | Shortest path between two nodes |
| `drift_search(...)` | DRIFT graph search |
| `regional_drift(...)` | Regional DRIFT search |
| `connected_components(...)` | Connected components |
| `strongly_connected_components(...)` | Strongly connected components |
| `degree_centrality(...)` | Degree centrality |
| `jaccard_coefficient(...)` | Jaccard similarity of neighborhoods |
| `louvain_communities(...)` | Louvain community detection |
| `leiden_communities(...)` | Leiden community detection |
| `modularity(...)` | Modularity of a partition |
| `pagerank(...)` | PageRank |
| `personalized_pagerank(...)` | Personalized PageRank |
| `preferential_attachment(...)` | Preferential-attachment link score |

```sql
SELECT pagerank(source, target) FROM edges;
SELECT louvain_communities(source, target) FROM edges;
```

`drift_search` and `regional_drift` take two trailing optional arguments —
`graph_uri` and `mode` — that select how the graph is sourced:

```sql
-- In-memory graph built from the aggregated edge rows (default).
SELECT regional_drift(source, target, 'query', [1, 2], 2, 2) FROM edges;

-- Out-of-core: load the table's CSR index and walk it directly.
SELECT regional_drift(source, target, 'query', [1, 2], 2, 2,
                      'file:///data/edges', 'out_of_core') FROM edges;
```

`mode` is one of `in_memory`, `out_of_core`, `cached`, or `auto` (the default
when `graph_uri` is supplied). Every mode is restricted to the same regional
induced subgraph, so the mode is a memory strategy, not a different answer.

## Introspection

Standard Flight SQL metadata RPCs (`GetCatalogs`, `GetDbSchemas`, `GetTables`,
`GetPrimaryKeys`, `GetSqlInfo`, `GetXdbcTypeInfo`) and DataFusion's
`information_schema` are all populated from the registered catalogs/schemas.

## Notes and limitations

- `CREATE TABLE AS SELECT`, `CREATE TABLE LIKE`, and `CREATE OR REPLACE TABLE`
  are not supported.
- `DROP DATABASE` is a no-op (no catalog deregistration in DataFusion).
- `OPTIMIZE ... ZORDER BY` is not supported (no true Z-order clustering).
- `GRANT`/`REVOKE`, roles, and `LISTEN`/`NOTIFY` are out of scope.
- DML (`INSERT`/`UPDATE`/`DELETE`) is executed during `GetFlightInfo` because
  ADBC cancels the follow-up `DoGet` for DML statements.
