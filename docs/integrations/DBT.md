# dbt Adapter (`dbt-benostreamdb`)

The `dbt-benostreamdb` adapter enables analytics engineering, automated data transformations, and data lineage directly over BenoStreamDB and Apache Iceberg tables.

---

## Overview

With `dbt-benostreamdb`, you can define models using familiar dbt SQL workflows while taking full advantage of BenoStreamDB's native capabilities:
* **Iceberg Table & View Materializations**: Build standard Parquet tables and metadata views without moving data out of your lakehouse.
* **Incremental Materializations**: Efficient append and merge models leveraging primary keys and deletion vectors.
* **Vector & Hybrid Search Macros**: Embedded macros for vector distance scoring and hybrid retrieval (`benostreamdb.vector_search(...)`).
* **Graph Traversal Macros**: Native relationship walks (`benostreamdb.graph_neighbors(...)`, `benostreamdb.graph_shortest_path(...)`) over edge tables.

---

## Installation

Install the adapter via pip:

```bash
pip install dbt-benostreamdb
```

Or install from source within the repository:

```bash
cd dbt-benostreamdb
pip install -e .
```

---

## Connection Profile (`profiles.yml`)

Configure your target connection in `~/.dbt/profiles.yml`:

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

For cloud object storage targets (S3 / GCS / Azure), specify the storage URI:

```yaml
benostreamdb_profile:
  target: prod
  outputs:
    prod:
      type: benostreamdb
      database: default
      schema: analytics
      threads: 8
      uri: "s3://my-lakehouse-bucket/warehouse"
```

---

## Supported Materializations

### 1. Table Materialization

Creates an authoritative Apache Iceberg table:

```sql+jinja
{{ config(
    materialized='table',
    format_version=2
) }}

SELECT
    id,
    title,
    category,
    embedding
FROM {{ ref('stg_documents') }}
WHERE is_active = true
```

### 2. Incremental Materialization

Processes only new or updated rows since the last dbt execution:

```sql+jinja
{{ config(
    materialized='incremental',
    unique_key='id'
) }}

SELECT
    id,
    user_id,
    event_name,
    event_timestamp
FROM {{ ref('stg_events') }}

{% if is_incremental() %}
    WHERE event_timestamp > (SELECT max(event_timestamp) FROM {{ this }})
{% endif %}
```

---

## Vector & Search Macros

BenoStreamDB exposes helper macros for similarity queries and vector analytics:

```sql+jinja
{{ config(materialized='table') }}

SELECT
    id,
    title,
    embedding <-> '[0.12, 0.45, -0.23]'::vector AS l2_dist,
    embedding <=> '[0.12, 0.45, -0.23]'::vector AS cosine_dist
FROM {{ ref('catalog_items') }}
ORDER BY cosine_dist ASC
LIMIT 50
```

---

## Graph Traversal Macros

Query relationship models and edge tables directly in your dbt models:

```sql+jinja
{{ config(materialized='table') }}

-- Traverse 2 hops from seed entity 101
SELECT *
FROM graph_neighbors('{{ ref("stg_edges") }}', '101', 2, 'auto')
```

---

## Testing & Verification

Run standard dbt commands to validate and execute models:

```bash
# Test connection
dbt debug

# Run models
dbt run

# Run tests
dbt test
```
