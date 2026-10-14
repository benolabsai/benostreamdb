# Python Binding Ergonomics: Hiding the `arrow_cast(...)` Tax

Status: proposal
Scope: embedded Python session (`bsdb.Session` / `bsdb.Table`) and the SQL surface it exposes
Motivation: the RAG + knowledge-graph how-to notebook only runs because of dozens of hand-written
type casts in SQL. Users should not have to know that graph UDFs want `UInt64`/`UInt32`, that
`louvain` wants `Float32`, or that a vector literal needs `::vector`. The bindings should absorb
all of that.

---

## 1. What we actually had to cast in the notebook / tests

Every one of these is friction the user pays for:

| Where | Cast written | Why it was needed |
|---|---|---|
| Graph traversal SQL (edge `source`/`target`) | `arrow_cast(source, 'UInt64')`, `arrow_cast(target, 'UInt64')` | Edge columns arrive as `Int64`/`String`; graph UDFs are declared `Exact([UInt64, UInt64, ...])` and DataFusion will not implicitly coerce. |
| Graph traversal SQL (seed / endpoints) | `arrow_cast(133, 'UInt64')`, `arrow_cast(src, 'UInt64')` | Integer literals parse as `Int64`; the UDF wants `UInt64`. |
| Graph traversal SQL (hops) | `arrow_cast(2, 'UInt32')` | Literal `2` is `Int64`; the UDF wants `UInt32`. |
| PageRank / PPR weights | `arrow_cast(0.85, 'Float64')` | Literal `0.85` is `Float64` here, but PPR accepts a specific `Float64` slot and fails without it in some signatures. |
| Louvain / Leiden resolution | `arrow_cast(1.0, 'Float32')` | Literal `1.0` parses as `Float64`; the UDF wants `Float32`. |
| Seed arrays | `make_array(arrow_cast(1, 'UInt64'), arrow_cast(2, 'UInt64'))` | Array literal elements default to `Int64`. |
| Vector literals | `'[0.1, 0.2, 0.3]'::vector` | Operators `<->`, `<=>`, `<#>` require the `vector` type; a bare array/`List(Float32)` is not accepted. |
| Sub-vector / quantize args | `(embedding)::FLOAT[]`, `arrow_cast(0, 'Int32')` | `subvector` wants `Float32[]` and `Int32` positions. |

Representative source: `dbt-benostreamdb/test_project/.../test_all_graph_functions.sql`,
`test_graph_macros.sql`, and the notebook's `graph_neighbors` / `shortest_path` cells.

---

## 2. Root causes

**R1 — Strict UDF signatures.** Graph/vector UDFs are registered with exact signatures, e.g.
`Exact([UInt64, UInt64, UInt32])`, and the planner refuses to coerce `Int64`/`Utf8`/`Float64` into
them. This is the single biggest source of casts. Error form:

```
Failed to coerce arguments to satisfy a call to 'louvain_communities':
coercion from Utf8, Utf8, Float32, Float32 to the signature
Exact([UInt64, UInt64, Float32, Float32]) failed
```

**R2 — Node ids as raw columns.** Graph UDFs take the *endpoint columns* as arguments
(`graph_neighbors(source, target, seed, hops)`), so the caller must project and cast table columns.
This couples every query to the edge-table schema.

**R3 — Graph table functions are not registered in the embedded session.** The docs advertise the
ergonomic form `SELECT * FROM graph_neighbors('doc_links', seed, hops)`, but in the embedded session
we get `table function 'graph_neighbors' not found`, forcing the column-argument + cast form.

**R4 — Vector literals require `::vector`.** pgvector-style operators need the `vector` domain; a
Python list or `List(Float32)` is not accepted implicitly.

**R5 — Assorted session annoyances hit while wiring the notebook up**
- `session.sql("... ;")` is rejected: `SQL query contains disallowed characters (';' or '--')`. Pasted
  SQL almost always ends in `;`.
- `session.sql("CREATE INDEX ...")` → `Unsupported logical plan: CreateIndex` (DDL must go through
  `add_index` or Flight SQL). Fine, but the error should say so.
- `graph_shortest_path(...)` → `Invalid function 'graph_shortest_path'. Did you mean 'shortest_path'?`
  (name mismatch between docs and engine).

---

## 3. Recommendations

### R1 → coerce at the UDF boundary (highest impact, engine-side)

Give graph/vector UDFs a user-defined signature with an explicit coercion step, or register
overloads for the common input types:

- Node ids: accept `UInt64 | Int64 | Utf8` (parse numeric strings). Cast internally to `UInt64`.
- Counts/positions: accept `UInt32 | Int64 | UInt64`.
- Weights/resolution: accept `Float32 | Float64 | Int64` and cast to the slot type.
- Vectors: accept `List(Float32) | List(Float64) | FixedSizeList | Utf8(literal)`.

Sketch (DataFusion):
```rust
// instead of Signature::exact(vec![UInt64, UInt64, UInt32])
Signature::user_defined(|dtypes| {
    // insert CAST(expr AS UInt64) for any numeric/utf8 node id, etc.
    Ok(DataType::UInt64 /* or resolved signature */)
})
```
Net effect: `graph_neighbors(source, target, 133, 2)` works with **no casts**; `arrow_cast`
disappears from every example.

### R2/R3 → register graph table-valued functions, driven by table metadata

`create_edge_table` already stores `table_type` + `source_column`/`target_column` (surfaced by
`list_graph_tables`). Use that to register TVFs at `register()` time:

```sql
SELECT * FROM graph_neighbors('doc_links', 133, 2);            -- hops
SELECT * FROM shortest_path('doc_links', 2, 7);
SELECT * FROM graph_pagerank('doc_links');                     -- defaults from metadata
SELECT * FROM graph_louvain('doc_links', 1.0);
```
The TVF reads the endpoint columns itself and does the internal casting — the user never names a
column or a type. Keep the column-argument scalar form for advanced use.

### R4 → vector ergonomics

- Accept `List(Float32)` in the `<->` / `<=>` / `<#>` operators (coercion), so `[0.1, 0.2]` and a
  Python list both work.
- Add a Python constructor so users never hand-build string literals:
  ```python
  from benostreamdb import vector
  table.knn("embedding", vector([0.1, 0.2, 0.3]), k=10)
  ```

### Python binding helpers (the real fix for notebooks)

Prefer Python methods that return DataFrames and hide all casts. Some already exist
(`Table.graph_neighbors`, `Table.shortest_path`, `Table.graph_rag_search`) — make them the
*primary* documented surface and extend them:

```python
session = bsdb.Session()
session.register("doc_chunks", doc_chunks)
session.register("doc_links", links)

session.graph_neighbors("doc_links", seed=133, hops=2)      -> DataFrame
session.shortest_path("doc_links", 2, 7)                    -> DataFrame
session.pagerank("doc_links", damping=0.85)                 -> DataFrame
session.louvain("doc_links", resolution=1.0)                -> DataFrame
session.hybrid_search("doc_chunks", text="smoking ...",      -> DataFrame  (server-side RRF)
                       vector_col="embedding", k=10)
```
Each helper: accepts Python `int`/`str`/`list`, builds the correctly-cast SQL (or calls the core
directly), returns a pandas DataFrame, and never asks the caller for a type.

### R5 → smooth the sharp edges

- `Session.sql`: strip a single trailing `;` and allow `--` line comments (or parse+re-emit) instead
  of rejecting the whole string. Users expect to paste SQL verbatim.
- DDL: when `CREATE INDEX` arrives at the embedded session, return a helpful error
  (`"Use table.add_index(col, algo), or send DDL over Flight SQL/Trino"`) instead of
  `Unsupported logical plan`.
- Function names: alias `graph_shortest_path` → `shortest_path` (and audit docs vs engine names) so
  the documented spelling works.
- Diagnostic hint: on a signature-coercion failure, append "try `graph_neighbors('table', seed, hops)`"
  to the error.

---

## 4. Priority

| # | Change | Effort | Impact |
|---|---|---|---|
| 1 | UDF signature coercion (R1) | Med (Rust) | Very high — removes ~90% of casts |
| 2 | Graph TVFs from edge-table metadata (R2/R3) | Med (Rust) | Very high — removes the column-arg + cast form |
| 3 | `Session` DataFrame graph/search helpers (Python) | Low | High — best notebook DX |
| 4 | `Session.sql` tolerate `;` and comments (R5) | Low | Medium |
| 5 | Vector literal coercion + `vector()` helper (R4) | Low–Med | Medium |
| 6 | Better DDL/name diagnostics (R5) | Low | Medium |

If only one thing ships: **#1**. If two: **#1 + #2** — together they make the "one database" story
readable, because the example SQL becomes exactly what a user would write by hand.

---

## 5. Before/after

```sql
-- Today: every call is buried in casts
SELECT DISTINCT unnest(graph_neighbors(
    arrow_cast(source,'UInt64'), arrow_cast(target,'UInt64'),
    arrow_cast(133,'UInt64'), arrow_cast(2,'UInt32'))) AS neighbor
FROM doc_links;

-- After R1 (#1)
SELECT DISTINCT unnest(graph_neighbors(source, target, 133, 2)) AS neighbor FROM doc_links;

-- After R2/R3 (#2)
SELECT * FROM graph_neighbors('doc_links', 133, 2);
```

```python
# After #3 — the form the notebook should teach
session.graph_neighbors("doc_links", seed=133, hops=2)
```

---

## Update: R2/R3 shipped (graph table functions)

The graph table functions now exist and read endpoints from the edge table's own metadata
(`list_graph_tables` reports `table_type=edge` + `source_column`/`target_column`). Verified against a
freshly built edge table:

```sql
SELECT * FROM graph_neighbors('doc_links', 133, 2);   -- cols: node, hop, seed
SELECT * FROM shortest_path('doc_links', 2, 7);       -- cols: node, hop
SELECT * FROM graph_pagerank('edges');                -- cols: node, rank
SELECT * FROM graph_louvain('edges', 1.0);            -- cols: node, community
```

As a result, **the RAG + knowledge-graph notebook is now fully cast-free** — every `arrow_cast(...)`
is gone from both the notebook and the article, replaced by the table-function form. Two follow-ups
from the original plan remain open:

1. The **column-argument scalar form still rejects bare columns** (`graph_neighbors(source, target,
   133, 2)` → `No field named source`); only the table-function form is cast-free today. Either
   finish R1 (coercion on the scalar form) or deprecate the scalar form in favor of the TVFs.
2. `session.sql("... ;")` still rejects trailing semicolons/comments (R5), and `CREATE INDEX` over the
   embedded session still returns `Unsupported logical plan` (use `add_index` / Flight SQL).
