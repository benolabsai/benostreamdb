# Tutorial 1 — AI / semantic retrieval via MCP

> The runnable Python version is [`examples/tutorials/01_ai_retrieval_mcp.ipynb`](../../examples/tutorials/01_ai_retrieval_mcp.ipynb).
> This page is the SQL / connector form for Spark, Trino, dbt, Flight SQL, MCP, and Rust.

Build a document corpus with embeddings, index it with HNSW, and run the
six-step agent workflow: **discover → schema → SQL → vector → graph → combine**.

## 1. Create the table and index (SQL)

```sql
CREATE TABLE docs (id BIGINT, title VARCHAR, embedding FLOAT[4]);
INSERT INTO docs VALUES
  (1, 'Iceberg overlays',      [1.0, 0.0, 0.0, 0.0]),
  (2, 'Vector search',         [0.9, 0.1, 0.0, 0.0]),
  (3, 'Graph on the lakehouse',[0.0, 1.0, 0.0, 0.0]);

-- HNSW overlay (rebuildable; the Parquet data is never rewritten).
-- The index *type* is named in the USING clause; a bare
-- `ALTER TABLE docs ADD INDEX (embedding)` would create a scalar Bitmap.
CREATE INDEX idx_emb ON docs (embedding) USING hnsw;
```

## 2. Vector retrieval (SQL)

```sql
SELECT id, title
FROM docs
ORDER BY vector_distance(embedding, [1.0, 0.0, 0.0, 0.0]) ASC
LIMIT 2;
```

## 3. The agent workflow via MCP

Build the server and register it with your MCP client (Claude Desktop, Cursor,
Antigravity):

```bash
cargo build --release --manifest-path contrib/benostreamdb-mcp/Cargo.toml
```

```json
{
  "mcpServers": {
    "benostreamdb": {
      "command": "/path/to/benostreamdb-mcp",
      "env": { "BSDB_WAREHOUSE": "/tmp/bsdb_warehouse", "BSDB_GPU_DEVICE": "auto" }
    }
  }
}
```

The agent then calls these tools:

| Step | Tool | Arguments |
| :--- | :--- | :--- |
| 1. discover | `list_graph_tables` | `{}` |
| 2. schema | `resources/read` | `{ "uri": "benostreamdb://tables/default/docs" }` |
| 3. SQL | `execute_sql` | `{ "query": "SELECT id, title FROM docs ORDER BY id" }` |
| 4. vector | `execute_sql` | `{ "query": "SELECT id FROM docs ORDER BY vector_distance(embedding, [1,0,0,0]) LIMIT 2" }` |
| 5. graph | `execute_sql` | `{ "query": "SELECT * FROM graph_neighbors('edges', '101', 2, 'auto')" }` |
| 6. combine | `execute_sql` | join the above |
| live tail | `subscribe_events` | `{ "table": "docs", "max_events": 5, "timeout_ms": 100 }` |

The MCP server is **read-all / write-scratch-only**: it can query any schema but
only writes to the local `scratch` schema.

## 4. The same workflow from each connector

**Spark** (pass-through reader):

```scala
spark.read.format("benostream")
  .option("path", "s3://my-lakehouse/docs")
  .option("query", "SELECT id, title FROM t ORDER BY vector_distance(embedding, array(1.0,0.0,0.0,0.0)) LIMIT 2")
  .load()
```

**Trino**:

```sql
SELECT id, title FROM benostreamdb.default.docs
ORDER BY vector_distance(embedding, ARRAY[1.0, 0.0, 0.0, 0.0]) LIMIT 2;
```

**dbt** (a model):

```sql+jinja
{{ config(materialized='table') }}
SELECT id, title FROM {{ ref('docs') }}
ORDER BY vector_distance(embedding, [1.0, 0.0, 0.0, 0.0]) LIMIT 2
```

**Flight SQL** (ADBC / pyarrow.flight): send the same SQL as a
`CommandStatementQuery`.

**Rust** (core API):

```rust
let table = Table::new_async(uri).await?;
let hits = table.vector_search("embedding", &[1.0, 0.0, 0.0, 0.0], 2).await?;
```

## GPU

Pin the device with `BSDB_GPU_DEVICE` (Flight/MCP), `spark.benostream.gpu.device`
(Spark), `benostream.gpu-device` (Trino), the dbt profile `gpu_device`, or
`benostreamdb.set_gpu_device(...)` (Python).
