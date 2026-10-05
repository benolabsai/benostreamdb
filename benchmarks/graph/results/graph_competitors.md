# Graph Competitor Comparison: BenoStreamDB vs NetworkX vs Neo4j+GDS

- **Graph**: 10,000 nodes / 49,975 edges (Barabási–Albert scale-free)
- **Algorithms**: PageRank (damping 0.85, 30 iters), weakly-connected components, shortest path
- **Device**: CPU
- **Host**: AMD Ryzen 9 5900XT 16-Core Processor (Linux)

### Algorithm Latency (ms) — excludes ingestion/load

| Algorithm | BenoStreamDB (Rust CSR) | NetworkX (Python) | Neo4j + GDS |
|---|---|---|---|
| **PageRank** | **16.15 ms** | 70.25 ms | 210 ms |
| **Connected components** | **4.53 ms** | 4.62 ms | 200 ms |
| **Shortest path** | 1.69 ms | **0.39 ms** | n/a (source-tree only) |

### Setup / load time (one-time, excluded above)

| Engine | Load + index/projection |
|---|---|
| BenoStreamDB | 0.13 s (Parquet write + CSR build) |
| NetworkX | 0.06 s (in-memory graph build) |
| Neo4j + GDS | **105.9 s** (Cypher `MERGE` load + `gds.graph.project`) |

### Notes
- **NetworkX** is a pure-Python in-memory reference — its role is a **correctness oracle**, not a performance baseline. It is trivially slow for PageRank but competitive on the tiny CC/shortest-path workloads.
- **Neo4j + GDS** is a real graph database with optimized algorithms, so it is the credible **performance** competitor. Its algorithm latency includes Bolt round-trips + Cypher planning; GDS runs on an in-memory projection (`gds.graph.project`). The 105.9 s load is the Cypher `MERGE` ingestion, not query time.
- BenoStreamDB's graph algorithms are pure Rust over the CSR index (CPU); there is no GPU graph path. cugraph is a competitor-only baseline, not an internal engine.

### Methodology
- BenoStreamDB / NetworkX: `benchmarks/graph/run.py` (10k/50k synthetic graph).
- Neo4j: `benchmarks/competitors/run_competitor.py --engine neo4j` against Neo4j 5.26 + GDS (`gds.pageRank.stream`, `gds.wcc.stream`); load and algorithm timed separately.
