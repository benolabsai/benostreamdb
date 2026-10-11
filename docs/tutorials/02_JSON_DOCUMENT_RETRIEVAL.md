# Tutorial 2 — JSON / document retrieval with a JSON-path index

> The runnable Python version is [`examples/tutorials/02_json_document_retrieval.ipynb`](../../examples/tutorials/02_json_document_retrieval.ipynb).
> This page is the SQL / connector form.

Store semi-structured documents in a `Utf8` JSON column, add a **JSON-path
inverted index** over the paths you filter on, and query with the
PostgreSQL-style `json_*` functions. The index is a rebuildable overlay — the
Parquet data is never rewritten.

## 1. Create the table

```sql
CREATE TABLE docs (id BIGINT, doc VARCHAR);
INSERT INTO docs VALUES
  (1, '{"user": {"id": "u-42"}, "level": "gold"}'),
  (2, '{"user": {"id": "u-7"},  "level": "silver"}'),
  (3, '{"user": {"id": "u-42"}, "level": "silver"}');
```

## 2. Add the JSON-path index

```sql
-- The index *type* is named in the USING clause. (A bare
-- `ALTER TABLE docs ADD INDEX (doc)` creates a scalar Bitmap, not a JSON index.)
CREATE INDEX idx_doc ON docs (doc) USING json_path;
```

The JSON paths to index are supplied through the connector/Python config form
(the SQL `USING` clause names the algorithm but cannot carry the path list):

```python
table.add_index("doc", {"type": "json_path", "paths": ["$.user.id", "$.level"]})
```

Each configured path becomes a `(path, value) -> row_ids` overlay packed into
the segment's Puffin bundle under the `json_path` category.

## 3. Query

```sql
-- Equality on an indexed path -> index lookup. The path is *variadic*
-- (one argument per level), not a single `$.a.b` string.
SELECT id FROM docs WHERE json_extract_path_text(doc, 'user', 'id') = 'u-42';

-- Containment (recursive @>).
SELECT id FROM docs WHERE json_contains(doc, '{"level": "silver"}');

-- Extract a value.
SELECT id, json_extract_path_text(doc, 'level') AS level FROM docs;
```

The planner rewrites `json_contains` / `json_exists` / `json_path_exists` /
`json_extract_path_text(...) = 'value'` into index lookups, falling back to a
full scan for unindexed paths. The index records a *superset* of the true
matches; the filter is re-applied above the scan, so results are exact.

## 4. From each connector

The `json_*` functions and the `json_path` index are engine features, so they
work identically from **Spark** (pass-through), **Trino**, **dbt**, **Flight
SQL**, and **MCP** (`execute_sql`). Only the index DDL spelling differs:

| Surface | Add the index |
| :--- | :--- |
| SQL (engine/Trino/Flight) | `CREATE INDEX idx_doc ON docs (doc) USING json_path;` |
| Spark | `CALL benostream.system.add_index('docs', 'doc', 'json_path');` |
| dbt | `{{ benostreamdb__create_index('docs', 'doc', 'json_path') }}` |
| Python | `table.add_index("doc", {"type": "json_path", "paths": ["$.user.id"]})` |

## Why this matters

The index is **advisory**: if it is dropped or lost, the same query still
returns the correct rows via a full scan, and the overlay rebuilds. That is the
story — *"add an index to an existing Iceberg table without rewriting the
table."*
