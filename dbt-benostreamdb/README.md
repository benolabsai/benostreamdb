# dbt Adapter for BenoStreamDB (`dbt-benostreamdb`)

`dbt-benostreamdb` is the official dbt adapter enabling analytics engineering, transformation models, and data lineage directly against BenoStreamDB.

## Features

- **Table & View Materializations**: Build standard and optimized Iceberg-format tables.
- **Incremental Materializations**: Efficient append and merge updates leveraging BenoStreamDB's primary key upserts and delete vector acceleration.
- **Vector & Graph Extensions**: Built-in macros for vector distance scoring, hybrid search, and graph relationship traversals (`benostreamdb.vector_search(...)`, `benostreamdb.graph_neighbors(...)`).

## Installation

```bash
pip install dbt-benostreamdb
```

## Configuration (`profiles.yml`)

```yaml
benostreamdb_profile:
  target: dev
  outputs:
    dev:
      type: benostreamdb
      database: default
      schema: public
      threads: 4
      uri: "file:///path/to/benostreamdb/warehouse"
```

## Remote / containerised runs: use `dbt-trino`

This adapter is an **embedded** adapter: it opens the warehouse through the
engine's JNI/`benostreamdb` Python module, so the warehouse path has to be
mounted in the process running dbt.

For Airflow, Cloud Run, CI, or any topology where dbt should not mount the
warehouse, point `dbt-trino` at a Trino coordinator that has the
`trino-benostreamdb` plugin installed. This is the supported remote path and
needs no extra transport in this adapter:

```yaml
# profiles.yml — remote access through the BenoStreamDB Trino connector
benostreamdb_remote:
  target: prod
  outputs:
    prod:
      type: trino
      method: none            # or ldap / jwt, per your coordinator
      host: trino.internal
      port: 8080
      catalog: benostream     # the catalog backed by trino-benostreamdb
      schema: default
      threads: 8
```

Notes:

- The catalog name must match the entry in the coordinator's
  `etc/catalog/benostream.properties` (`connector.name=benostream`).
- Table functions work remotely too: `system.vector_search`,
  `system.hybrid_search`, `system.list_indexes`, `system.regional_drift_search`.
  `hybrid_search` now honours its `filter` argument, so a filtered hybrid query
  is pushed to the engine instead of being rejected.
- `vector_search` accepts a sixth `metric` argument (`cosine` by default, plus
  `l2`, `dot`, `l1`, `hamming`, `jaccard`).
- Keep the embedded adapter for local development and for features that need the
  Python API directly (index building, graph traversal from Python).
