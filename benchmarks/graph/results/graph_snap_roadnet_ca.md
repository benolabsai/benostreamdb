# Graph Benchmark Results

- **Graph Nodes**: 1,965,206
- **Graph Edges**: 5,533,214
- **Source Dataset**: `snap-roadnet-ca.tsv`

| Algorithm | Engine | Status | Build (s) | Execution Latency (ms) | Result Size | Speedup vs NetworkX |
|---|---|---|---|---|---|---|
| pagerank | benostreamdb | ✅ Pass | 13.794s | 2478.35 ms | 1965206 | 1.41x |
| pagerank | networkx | ✅ Pass | 8.298s | 3499.09 ms | 1965206 | 1.00x (baseline) |
| pagerank | neo4j | ✅ Pass | 81.364s | 31003.24 ms | 1965206 | 9× slower |
| connected_components | benostreamdb | ✅ Pass | 13.657s | 531.19 ms | 2638 | 1.71x |
| connected_components | networkx | ✅ Pass | 7.363s | 908.71 ms | 2638 | 1.00x (baseline) |
| connected_components | neo4j | ✅ Pass | 81.364s | 30717.12 ms | 1965206 | 34× slower |
| shortest_path | benostreamdb | ✅ Pass | 13.648s | 0.08 ms | 13 | 4.56x |
| shortest_path | networkx | ✅ Pass | 8.171s | 0.38 ms | 13 | 1.00x (baseline) |
| shortest_path | neo4j | ✅ Pass | 81.364s | 18.07 ms | 1 | 47× slower |
