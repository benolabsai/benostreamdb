# Graph Benchmark Results

- **Graph Nodes**: 1,965,206
- **Graph Edges**: 5,533,214
- **Source Dataset**: `snap-roadnet-ca.tsv`

| Algorithm | Engine | Status | Build (s) | Execution Latency (ms) | Result Size | Speedup vs NetworkX |
|---|---|---|---|---|---|---|
| pagerank | benostreamdb | ✅ Pass | 15.084s | 3098.10 ms | 1965206 | 1.21x |
| pagerank | networkx | ✅ Pass | 8.802s | 3763.81 ms | 1965206 | 1.00x (baseline) |
| pagerank | neo4j | ✅ Pass | 87.007s | 31525.01 ms | 1965206 | 8× slower |
| connected_components | benostreamdb | ✅ Pass | 15.003s | 660.85 ms | 2638 | 1.49x |
| connected_components | networkx | ✅ Pass | 7.502s | 987.02 ms | 2638 | 1.00x (baseline) |
| connected_components | neo4j | ✅ Pass | 87.007s | 31716.89 ms | 1965206 | 32× slower |
| shortest_path | benostreamdb | ✅ Pass | 15.001s | 0.07 ms | 13 | 5.19x |
| shortest_path | networkx | ✅ Pass | 8.543s | 0.38 ms | 13 | 1.00x (baseline) |
| shortest_path | neo4j | ✅ Pass | 87.007s | 14.19 ms | 1 | 37× slower |
