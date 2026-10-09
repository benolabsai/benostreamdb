# Tutorial 3 — Graph traversal via `graph_neighbors(...)`

> The runnable Python version is [`examples/tutorials/03_graph_traversal.ipynb`](../../examples/tutorials/03_graph_traversal.ipynb).
> This page is the SQL / connector form.

Model a graph as an ordinary edge table, then traverse it directly in a `FROM`
clause. The graph is an **overlay on your lakehouse data**, not a separate graph
database.

## 1. Create the edge table

```sql
CREATE TABLE edges (source BIGINT, target BIGINT, relation VARCHAR, weight DOUBLE);
INSERT INTO edges VALUES
  (101, 205, 'knows', 1.0),
  (101, 309, 'knows', 1.0),
  (205, 410, 'works_with', 0.5),
  (309, 410, 'works_with', 0.5);

-- Declare it as an edge table. This auto-configures the forward (source) and
-- reverse (target) CSR graph indexes.
ALTER TABLE edges SET TBLPROPERTIES ('table_type' = 'edge', 'src_col' = 'source', 'dst_col' = 'target');
```

## 2. Traverse in a `FROM` clause

```sql
-- Nodes within 2 hops of node 101 (node, hop, seed).
SELECT * FROM graph_neighbors('edges', '101', 2, 'auto');

-- Shortest path between two nodes (node, hop).
SELECT * FROM graph_shortest_path('edges', 101, 410, 'auto');

-- Every shortest path (path).
SELECT * FROM graph_all_shortest_paths('edges', 101, 410, 'auto');

-- Induced subgraph edges within 2 hops (source, target).
SELECT * FROM graph_subgraph('edges', '101', 2, 'auto');

-- Union of pairwise shortest paths between seeds (source, target).
SELECT * FROM graph_connecting_paths('edges', '101,410', 'auto');
```

The `mode` argument follows the in-memory / out-of-core `GraphMode` pattern
(`auto` | `in_memory` | `out_of_core` | `cached`). Endpoint columns are resolved
**metadata-first** (`src_col`/`dst_col`), then by the standard name candidates
(`source`/`src`/… , `target`/`dst`/…); pass trailing `source`, `target` string
arguments to name them explicitly.

## 3. From each connector

`graph_neighbors` is a DataFusion table function, so it is reachable from every
SQL surface:

- **Python**: `session.sql("SELECT * FROM graph_neighbors('edges', '101', 2, 'auto')")`
- **Spark**: pass-through reader — `.option("query", "SELECT * FROM graph_neighbors('t', '101', 2, 'auto')")`
- **Trino**: `SELECT * FROM graph_neighbors('edges', '101', 2, 'auto');`
- **dbt**: `{{ benostreamdb__graph_neighbors_table('edges', '101', 2) }}`
- **Flight SQL / MCP**: send the same SQL.

## 4. Discovery

`Session.list_graph_tables()` (Python) / the MCP `list_graph_tables` tool
enumerate node/edge tables and their endpoint columns — the primitive an agent
uses to find the graph without being told the column names.

## Notes

- Declaring `table_type = 'edge'` (via `SET TBLPROPERTIES` or `set_property`)
  auto-configures the forward/reverse CSR indexes (`Table::ensure_edge_indexes_async`).
- The graph UDAFs (`graph_pagerank`, `graph_louvain_communities`, …) are also
  available over a `JOIN` of the edge table.
