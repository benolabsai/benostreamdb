# Graph Benchmark Results

- **Graph Nodes**: 1,965,206
- **Graph Edges**: 5,533,214
- **Source Dataset**: `snap-roadnet-ca.tsv`

| Algorithm | Engine | Status | Build (s) | Execution Latency (ms) | Result Size | Speedup vs NetworkX |
|---|---|---|---|---|---|---|
| pagerank | benostreamdb | ✅ Pass | 14.894s | 2994.95 ms | 1965206 | 1.17x |
| pagerank | networkx | ✅ Pass | 9.067s | 3502.93 ms | 1965206 | 1.00x (baseline) |
| pagerank | neo4j | ✅ Pass | 86.255s | 33300.04 ms | 1965206 | 10× slower |
| connected_components | benostreamdb | ✅ Pass | 15.027s | 762.03 ms | 2638 | 1.37x |
| connected_components | networkx | ✅ Pass | 7.939s | 1040.55 ms | 2638 | 1.00x (baseline) |
| connected_components | neo4j | ✅ Pass | 86.255s | 31817.70 ms | 1965206 | 31× slower |
| shortest_path | benostreamdb | ✅ Pass | 15.628s | 0.07 ms | 13 | 5.86x |
| shortest_path | networkx | ✅ Pass | 8.543s | 0.38 ms | 13 | 1.00x (baseline) |
| shortest_path | neo4j | ✅ Pass | 86.255s | 14.69 ms | 1 | 38× slower |
