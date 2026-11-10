# ADR-003: Graph RAG Hybrid Memory (In-RAM State, Out-of-Core Edges)

- **Status**: Implemented — G1 (CSR-backed community detection), G2 (default DataFusion memory limit), G3 (bounded Graph RAG materialization), and G4 (independent knobs) have all shipped. Promoted from `plans/graph_rag_memory_architecture.md`; see the per-item notes below.
- **Related**: [`src/core/memory.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/memory.rs), [`src/core/resources.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/resources.rs), [`src/core/cache.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/cache.rs), [`src/core/sql/session.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/sql/session.rs), [`src/core/sql/graph_udf/louvain_communities.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/sql/graph_udf/louvain_communities.rs), [`src/core/sql/graph_udf/leiden_communities.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/sql/graph_udf/leiden_communities.rs), [`src/core/index/csr_graph.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/index/csr_graph.rs)

## Principle

**Move the O(E) working set out of RAM; keep only O(V) state resident.** The CSR is already the out-of-core adjacency (mmap'd, `MADV_RANDOM`). The graph algorithms should *read from it* instead of buffering the edge set. For a 6M-node / 383M-edge Wikipedia graph, O(V) state is ~72 MB while O(E) is ~7.7 GB — a 100x difference.

```mermaid
flowchart LR
    subgraph RAM["In-RAM (O(V))"]
        C[community: Vec u32]
        D[degree: Vec f32]
        T[comm_tot: Vec f32]
    end
    subgraph OOC["Out-of-core (O(E), mmap)"]
        CSR[CSR offsets/edges/dict]
    end
    CSR -->|get_neighbors_raw| C
    C --> D --> T
```

## G1 — CSR-backed community detection (biggest risk)

**Status: implemented.** [`src/core/algorithms/communities.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/algorithms/communities.rs#L1) provides `csr_louvain` / `csr_leiden` (and their `_seeded` variants) reading adjacency from the mmap'd CSR with O(V) resident state; `Table.communities()` routes to the CSR path when a graph index is present.

**Problem.** [`LouvainAccumulator`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/sql/graph_udf/louvain_communities.rs#L104) and [`LeidenAccumulator`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/sql/graph_udf/leiden_communities.rs#L110) hold `sources`/`targets`/`weights` as `Vec`s — ~20 bytes/edge, ~7.7 GB at 383M edges.

**Design.** Add `src/core/algorithms/communities.rs` with `louvain_csr(forward, resolution)` and `leiden_csr(forward, resolution)` that read adjacency from the mmap'd CSR via `get_neighbors_raw` and keep only O(V) state:

- `community: Vec<u32>` (4 B/node)
- `degree: Vec<f32>` (4 B/node)
- `comm_tot: Vec<f32>` (4 B/node)

The local-move and refinement phases are identical to the UDF versions; only the adjacency source changes. `Table.communities()` routes to the CSR path when `has_graph_index("source")`, else falls back to the UDF accumulator (small graphs / SQL convenience). For tables without a CSR, build one on the fly by reusing [`MmapCsrGraph::build_from_file`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/index/csr_graph.rs#L88) into a temp dir.

**Acceptance.** Community detection on the wiki demo runs in bounded RSS (state only), and results match the UDF path on small graphs.

## G2 — Default DataFusion memory limit

**Status: implemented.** [`default_datafusion_memory_bytes`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/resources.rs#L222) derives the limit from `DATAFUSION_MEMORY_FRACTION` (0.5) of the effective memory, overridable with `BSDB_DATAFUSION_MEMORY_GB`, and it is applied to both `Session` and `Table.execute_sql`.

**Problem.** [`BenoStreamSession::new(None)`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/sql/session.rs#L27) sets no limit, and that is what [`bsdb.rs:122`](https://github.com/benolabsai/benostreamdb/blob/main/src/bin/bsdb.rs#L122) and [`graph.rs:58`](https://github.com/benolabsai/benostreamdb/blob/main/src/python/graph.rs#L58) use — so SQL sorts/joins/aggregations never spill.

**Design.** When `memory_limit_bytes` is `None`, default it to `effective_memory_bytes() * 0.5` (leaving headroom for the ingest buffer and index builds). Add `BSDB_DATAFUSION_MEMORY_GB` to override. This turns on DataFusion's disk spilling for the SQL path.

**Acceptance.** A large `ORDER BY`/join spills to disk instead of OOMing.

## G3 — Avoid full materialization in Graph RAG

**Status: implemented.** `subgraph_nodes()` (visited set only, no edge join) is exposed on the Python `Table` and used for `neighborhood_nodes`; the final edge materialization is capped by `max_nodes`.

**Problem.** [`graph_rag_search`](https://github.com/benolabsai/benostreamdb/blob/main/python/benostreamdb/__init__.py#L1134) materializes subgraph edges into pandas just to compute `neighborhood_nodes`; [`summarize_communities`](https://github.com/benolabsai/benostreamdb/blob/main/python/benostreamdb/__init__.py#L1564) does `SELECT * FROM t` over the whole doc table.

**Design.**
- Add `subgraph_nodes()` (visited set only, no edge join) and use it for `neighborhood_nodes`; materialize edges only for the final result, capped by `max_nodes`.
- In `summarize_communities`, replace `SELECT * FROM t` with a projection (`id`, `content`, `title`) and a `WHERE id IN (top_entities)` per community, or stream in batches.

**Acceptance.** Graph RAG peak RSS scales with `max_nodes`, not with the graph size.

## G4 — Individual knobs, not a unified budget (decided)

**Decision.** A single coordinated `MemoryBudget` with a fixed percentage split
was prototyped and **rejected**: operators prefer independent knobs so each
subsystem can be tuned to the actual workload. The individual overrides stand:

| Subsystem | Override |
|-----------|----------|
| Ingest high-water mark | `BSDB_MAX_INGEST_RAM_GB` |
| Heap-trim budget | `BSDB_INGEST_MEMORY_BUDGET_GB` |
| DataFusion query limit | `BSDB_DATAFUSION_MEMORY_GB` |
| Index-build fan-out | `BSDB_INDEX_BUILD_CONCURRENCY` |

**What was kept.** The DataFusion limit is *bounded and configurable*: it
defaults to `DATAFUSION_MEMORY_FRACTION` (50%) of the effective memory, and
`Table.execute_sql` — the path the graph UDFs run on — now applies it too (it
previously used an unbounded `SessionContext::new()`).

## Priority (completed)

1. ✅ **G1** — biggest risk, clear win, reuses the existing CSR.
2. ✅ **G2** — bounded + configurable DataFusion limit, enables spilling.
3. ✅ **G3** — bounded materialization in the Graph RAG path.
4. ✅ **G4** — resolved: independent knobs, no fixed split.
