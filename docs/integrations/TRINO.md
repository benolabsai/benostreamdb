# Trino Connector

The BenoStreamDB Trino connector queries your datasets with distributed SQL. It
implements the Trino SPI and delegates IO to the Rust core over JNI, so vectors,
keyword indexes, and graph traversals run in the engine rather than in Trino.

## Supported Trino versions

Trino publishes no LTS and does not keep the connector SPI source- or
binary-compatible between releases, so the connector is built and tested per
**band**:

| Band | Trino versions | Status |
|------|----------------|--------|
| `e` | 484 and newer | current, default |
| `d` | 482-483 | current-minus |

A plugin built for one band does not load on another. See
[`plans/trino_version_bands.md`](../../plans/trino_version_bands.md) for the
measured SPI differences and the refresh procedure when a new release lands.

## Building

Prerequisites:

- **JDK 23 or newer.** The 482+ SPI is compiled for Java 23 (class file 67); a
  JDK 21 compiler cannot read it.
- **The native library, built with the `java` Cargo feature.** The JNI entry
  points are feature-gated, so a plain `cargo build` produces a library with
  zero `Java_com_benostreamdb_*` symbols and every query fails with
  `UnsatisfiedLinkError`.

```bash
cargo build --release --features java

cd trino-benostreamdb
# Band e (484+), the default
mvn clean package -DskipTests
# Band d (482-483)
mvn clean package -DskipTests -Dtrino.version=482 -Dtrino.band=d
```

Each build produces a plugin ZIP in `trino-benostreamdb/target/` plus the main
JAR.

## Installation

Trino's plugin loader only scans JARs **directly** inside the plugin directory;
it does not recurse. The `trino-plugin` packaging nests everything under a
versioned folder, so flatten it before installing:

```bash
mkdir -p /usr/lib/trino/plugin/benostream
unzip -q trino-benostream-<version>.zip -d /tmp/bsdb-plugin
cp /tmp/bsdb-plugin/trino-benostream-<version>/*.jar /usr/lib/trino/plugin/benostream/
```

The native library is **not** bundled in the plugin JAR: Trino loads it from
`java.library.path`. Point it at a host or manylinux build of
`libbenostreamdb.so`.

### Required JVM flags

Trino 484's `io.trino.simd.BlockEncodingSimdSupport` references
`jdk.incubator.vector.VectorShape`. That class exists on JDK 25 but only inside
the incubator module, which is not resolved by default; without the flag the
engine dies during startup with
`NoClassDefFoundError: jdk/incubator/vector/VectorShape`. Trino's own launcher
adds it, but any embedder must match:

```
--add-modules=jdk.incubator.vector
--add-opens=java.base/java.nio=ALL-UNNAMED     # Arrow Java memory access
-Djava.library.path=/path/to/libbenostreamdb
```

## Configuration

`etc/catalog/benostreamdb.properties`:

```properties
connector.name=benostreamdb
# Warehouse root: a local path or an object-store URI.
benostream.warehouse=s3://my-bucket/
# auto | cpu | cuda[:N] | mps | intel | rocm; comma-separated for a device pool.
benostream.gpu-device=auto
```

## Querying tables

```sql
SELECT * FROM benostreamdb.default.logs
WHERE severity = 'ERROR' AND ts > NOW() - INTERVAL '1' DAY
```

Scalar predicates are pushed into the engine as a SQL `WHERE` clause, so
inverted indexes prune segments before Parquet is decoded.

## Stored procedures

All procedures live under `system` and mirror the Spark connector's catalog
name for name, so a migration needs no rewrite. Index and primary-key changes
are engine-side operations, not Trino DDL.

| Procedure | Arguments | Purpose |
|-----------|-----------|---------|
| `add_index` | `TABLE`, `COLUMN`, `ALGORITHM` (default `vector`) | Add a secondary index |
| `build_index` | `TABLE`, `SEGMENT_ID` (default `all`) | Build a pending index |
| `rebuild_index` | `TABLE`, `COLUMN` | Rebuild an index segment |
| `drop_index` | `TABLE`, `COLUMN` | Drop a secondary index |
| `compact` | `TABLE` | Rewrite data files |
| `optimize` | `TABLE` | Alias of `compact` |
| `set_primary_key` | `TABLE`, `COLUMNS` (comma-separated) | Declare the row key |
| `drop_primary_key` | `TABLE`, `COLUMNS` (comma-separated) | Remove key columns |

```sql
CALL benostreamdb.system.add_index('default.docs', 'embedding', 'hnsw');
CALL benostreamdb.system.build_index('default.docs');
CALL benostreamdb.system.set_primary_key('default.docs', 'id');
CALL benostreamdb.system.compact('default.docs');
```

## Table functions

Vector, hybrid, and graph retrieval are exposed as table functions in the
`system` schema. Arguments are named, and names are uppercase. `TABLE_NAME`
(rather than `TABLE`) is used because `TABLE => ...` is reserved syntax in
Trino for table arguments.

### Vector search

```sql
SELECT id, title, distance
FROM TABLE(system.vector_search(
    table_name => 'default.docs',
    column     => 'embedding',
    query      => '[1.0, 0.0, 0.0]',
    k          => 10,
    metric     => 'cosine'));          -- optional, default 'cosine'
```

The query vector is a JSON-style string literal. Trino's planner rewrites that
literal into a `FixedSizeList`, which is what the KNN optimizer recognises —
`ARRAY[1.0, 0.0, 0.0]` is not usable here. The engine runs the KNN search and
adds a `distance` column.

`METRIC` selects the distance UDF used for both the projected `distance` and the
`ORDER BY`:

| `metric` | Engine UDF |
|----------|------------|
| `cosine` (default), `cosine_distance` | `dist_cosine` |
| `l2`, `euclidean`, `l2_distance` | `dist_l2` |
| `dot`, `ip`, `inner_product`, `dot_product` | `dist_ip` |
| `l1`, `manhattan`, `l1_distance` | `dist_l1` |
| `hamming`, `hamming_distance` | `dist_hamming` |
| `jaccard`, `jaccard_distance` | `dist_jaccard` |

An unknown metric is rejected at analysis time (user error), never interpolated
into the SQL string.

### Hybrid search (vector + BM25, fused with RRF)

```sql
SELECT id, title, rrf_score
FROM TABLE(system.hybrid_search(
    table_name    => 'default.docs',
    text_column   => 'title',
    query_text    => 'graph',
    vector_column => 'embedding',
    query_vector  => '[1.0, 0.0, 0.0]',
    k             => 10,
    rrf_k         => 60,
    filter        => 'category = ''science'''));
```

`filter` is a SQL `WHERE` clause applied to **both** retrieval paths before
fusion: the engine resolves it to a row-id bitmap for indexed columns and
evaluates it against the hydrated rows otherwise, so a filtered hybrid query
never silently returns unfiltered neighbours. A malformed predicate is reported
as a user error; the parser is fuzzed (`fuzz/fuzz_targets/connector_inputs.rs`)
so it can never bring the coordinator down.

### Writes and MERGE

Deletions produced by `MERGE` are applied as batched predicates
(`key IN (v1, ..., v1000)`) instead of one `deleteRows` JNI call per row, and the
rendered predicate is capped at 64 KiB. A 5,000-row `MERGE` therefore costs a
handful of JNI transitions instead of 5,000 engine commits.

### Graph traversal

```sql
-- Nodes within N hops of the seed nodes: (node, hop, seed)
SELECT * FROM TABLE(system.graph_neighbors(
    table_name => 'default.edges', seeds => '101', hops => 2));

-- Shortest path between two nodes: (node, hop)
SELECT * FROM TABLE(system.graph_shortest_path(
    table_name => 'default.edges', source => '101', target => '205'));

-- Induced subgraph edges within N hops: (source, target)
SELECT * FROM TABLE(system.graph_subgraph(
    table_name => 'default.edges', seeds => '101,102', hops => 2));
```

All three accept an optional `mode` (`auto` | `in_memory` | `out_of_core` |
`cached`). Endpoint columns are auto-detected (`source`/`src`/… and
`target`/`dst`/…).

### Graph-expanded retrieval (`graph_rag_search`)

Vector ranking over a document table, joined to a traversal of a separate edge
table — two tables in one statement:

```sql
SELECT id, title, hop
FROM TABLE(system.graph_rag_search(
    table_name    => 'default.docs',
    edge_table    => 'default.edges',
    vector_column => 'embedding',
    query_vector  => '1.0, 0.0',
    seeds         => '101',
    id_column     => 'id',       -- default: id
    hops          => 1,          -- default: 1
    k             => 10,         -- default: 10
    mode          => 'auto'));
```

Returns the document columns plus `hop`. Internally the engine registers both
tables in one session (documents as `t`, edges as `e`) and evaluates:

```sql
SELECT d.*, g.hop FROM t d
JOIN graph_neighbors('e', '<seeds>', <hops>, '<mode>') g ON d.<id> = g.node
ORDER BY dist_cosine(d.<vector>, '[<query>]') LIMIT <k>
```

### DRIFT graph search

```sql
SELECT node_id FROM TABLE(system.regional_drift_search(
    table_name   => 'default.edges',
    query        => 'related incidents',
    seeds        => '101,102',
    top_k        => 5,
    hops         => 2,
    n_depth      => 2,
    k_followups  => 3,
    mode         => 'auto'));
```

Returns the discovered node ids as a single `node_id` column. This is a table
function rather than a procedure on purpose: Spark registers it as a procedure
whose `call` runs the search and then returns an empty iterator, so
`CALL regional_drift_search(...)` computes an answer and discards it. Both
surfaces run the same engine code and read the same Arrow `node_id` column.

### Index listing

```sql
SELECT column_name, index_category, algorithm, index_id, format_version
FROM TABLE(system.list_indexes(table_name => 'default.docs'));
```

Backed by the engine's manifest, so it reports what actually exists rather than
what was requested. Spark's equivalent `show_indexes` calls the same JNI entry
point but discards the rows.

Note that indexing is two-phase: `add_index` registers an index and builds it on
a detached background task. Call `build_index` when a subsequent query must
observe the index immediately.

## Parallel scans

`benostream.split.file-granularity=true` (catalog property, default `false`)
makes a plain, unfiltered table scan emit **one split per Parquet file**, so the
coordinator can spread the read across workers instead of running one
table-wide query:

```properties
# etc/catalog/benostream.properties
connector.name=benostream
benostream.warehouse=/data/warehouse
benostream.split.file-granularity=true
```

Each split is read through a file-scoped engine session. The connector only fans
out when it is safe: the scan must carry no pushed predicate, no
`pushdownSql`/table function, no multi-table statement, and no file may have
merge-on-read delete files (a raw per-file read would resurrect deleted rows).
Otherwise the single-split SQL path is used, which is why the default is off.

## Joining with other catalogs

Retrieval results join foreign catalogs directly in one statement; nothing is
exported or copied. This is the same code path exercised by
`TestCrossCatalogJoin`. An Iceberg catalog registered against a BenoStreamDB
table directory works the same way; see [ICEBERG.md](ICEBERG.md).

```sql
SELECT v.title, o.orderstatus
FROM TABLE(system.vector_search(
        table_name => 'default.docs', column => 'embedding',
        query => '[1.0, 0.0, 0.0]', k => 5)) v
JOIN tpch.tiny.orders o ON v.id = o.orderkey;
```

## Limitations

These are deliberate and documented rather than hidden:

- **No live change feed.** The engine's in-process `subscribe_events` feed is
  not exposed as a Trino table function. It is JVM-local by design, so a worker
  would only ever observe commits from its own process.
- **Iceberg reads are zero-copy.** Table metadata, manifests, Parquet, and
  delete files are spec-compliant, so Trino's Iceberg connector can
  `CALL iceberg.system.register_table(...)` against a BenoStreamDB table
  directory and read the same files without copying — including position
  deletes. Equality deletes are Avro, as the spec allows, and Trino reads them.
  See [ICEBERG.md](ICEBERG.md) for the contract and the exact commands.
- **Built-ins cannot be overridden per catalog.** Trino registers scalar
  functions globally, so `cosine_distance` and `dot_product` are Trino's
  implementations (the plugin cannot shadow them, and there is no
  schema-scoped overload). For engine-side scoring on our own tables, use
  `system.vector_search` / `system.hybrid_search`, which compute `distance` and
  `rrf_score` inside the Rust session.

## Scalar functions

Distance, fusion, and vector-transformation functions are available SQL-wide,
matching the Spark connector's UDFs:

| Function | Source | Signature |
|----------|--------|-----------|
| `cosine_distance` | Trino built-in | `(array(double), array(double)) -> double` |
| `dot_product` | Trino built-in | `(array(double), array(double)) -> double` |
| `euclidean_distance` | Trino built-in | `(array(double), array(double)) -> double` |
| `l2_distance` | this plugin | `(array(double), array(double)) -> double` |
| `vector_distance` | this plugin | `(array(double), array(double), varchar) -> double` |
| `sparse_dot_product` | this plugin | `(array(bigint), array(double), array(bigint), array(double)) -> double` |
| `hybrid_score` | this plugin | `(double, double, double) -> double` |
| `reciprocal_rank_fusion` | this plugin | `(array(integer), integer) -> double` |
| `vector_dims` | this plugin | `(array(double)) -> bigint` |
| `vector_norm` | this plugin | `(array(double)) -> double` |
| `l2_normalize` | this plugin | `(array(double)) -> array(double)` |
| `subvector` | this plugin | `(array(double), bigint, bigint) -> array(double)` |
| `vector_add` | this plugin | `(array(double), array(double)) -> array(double)` |
| `vector_sub` | this plugin | `(array(double), array(double)) -> array(double)` |

```sql
SELECT l2_distance(embedding, CAST(ARRAY[1.0, 0.0, 0.0] AS array(double))) AS d,
       vector_distance(embedding, CAST(ARRAY[1.0, 0.0, 0.0] AS array(double)), 'dot') AS dot_d,
       reciprocal_rank_fusion(CAST(ARRAY[1, 3] AS array(integer)), 60) AS rrf
FROM benostreamdb.default.docs;
```

`vector_distance` accepts `cosine`, `l2`/`euclidean`, and `dot`; `dot` is
negated so that lower is closer, matching the engine.

The vector transformations are **total functions** — connector- or
query-supplied input can never make a query fail at execution time:

- `l2_normalize` returns zeros for the zero vector instead of dividing by zero.
- `subvector(vector, offset, length)` uses 1-based `offset` with SQL `slice`
  semantics; negative or out-of-range bounds are clamped and an empty vector is
  returned rather than an error.
- `vector_add` / `vector_sub` stop at the shorter input.
- `vector_norm`/`l2_normalize` treat `NULL` elements as `0.0`.

## Testing

```bash
# Both supported bands
mvn test -Dtrino.version=482 -Dtrino.band=d
mvn test -Dtrino.version=484 -Dtrino.band=e
```

The suite runs a real `DistributedQueryRunner` with the native library loaded,
covering DDL/DML, every table function, the procedure surface, the type matrix,
error paths, and cross-catalog joins against a second first-party catalog.
