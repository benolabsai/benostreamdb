# Graph Competitor Comparison: BenoStreamDB vs NetworkX vs Neo4j + GDS (+ Memgraph, Kùzu, cuGraph)

- **Graph**: 10,000 nodes / 49,975 edges (Barabási–Albert scale-free)
- **Algorithms**: PageRank (damping 0.85, 30 iters), weakly-connected components, shortest path
- **Device**: CPU — shared Docker envelope (8 CPU / 16 GiB)
- **Host**: AMD Ryzen 9 5900XT 16-Core Processor (Linux)

The authoritative per-engine numbers are the **matrix rendered below** (aggregated
from the per-engine JSON records). This page describes how to read it.

### Engine roles and measurement layer

| Engine | Role | Measurement layer |
|---|---|---|
| BenoStreamDB | CSR graph, pure Rust | embedded (in-process Rust) |
| NetworkX | correctness oracle (not a perf baseline) | embedded (in-process Python) |
| Neo4j 5.26 + GDS | real graph database, credible perf competitor | native GDS in the Neo4j JVM (`gds.*.mutate`; shortest path via `gds.shortestPath.dijkstra.stream`) |
| Memgraph (MAGE) | native in-memory graph engine | native engine (`pagerank.get` / `weakly_connected_components.get`) |
| Kùzu | embedded columnar graph DB | embedded (`page_rank` / `weakly_connected_components` on a projected graph) |
| cuGraph | GPU graph analytics library | embedded (in-process GPU) |

### Notes

- **Neo4j + GDS supports all three algorithms, including shortest path**
  (`gds.shortestPath.dijkstra.stream`, `gds.bfs`). It is not limited to a
  source tree — the previous "n/a (source-tree only)" was incorrect.
- **Latency is the server-side algorithm execution** for Neo4j and Memgraph:
  the GDS/MAGE procedure runs inside the engine and returns a summary row, so the
  number excludes Bolt result transfer. Timing `.stream` from Python (the old
  approach) measured the driver marshalling 100k+ records, not the database; the
  per-record `layer` field records how each engine is measured.
- **Load / projection time** is reported per engine in the matrix; it is one-time
  setup and is excluded from the execution latency.
- **NetworkX** is a pure-Python in-memory reference used for **correctness**, not
  performance. **cuGraph** is a competitor-only GPU baseline, not an internal
  engine; BenoStreamDB's graph path is CPU/Rust.

### Methodology

- All engines: `benchmarks/competitors/run_competitor.py` under the shared Docker
  envelope (see `benchmarks/competitors/docker-compose.bench.yml`), one container
  and the same `--cpus`/`--mem` per participant.
- Neo4j: `gds.graph.project` + `gds.pageRank.mutate` / `gds.wcc.mutate` /
  `gds.shortestPath.dijkstra.stream`; load and algorithm timed separately.
- Memgraph: MAGE query modules over bolt; Kùzu: `algo` extension on a projected
  graph; cuGraph: `cugraph.pagerank` on GPU.
