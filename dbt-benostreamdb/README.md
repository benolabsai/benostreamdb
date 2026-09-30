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
