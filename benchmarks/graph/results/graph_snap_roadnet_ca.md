# Graph Benchmark Results

- **Graph Nodes**: 1,965,206
- **Graph Edges**: 5,533,214
- **Source Dataset**: `snap-roadnet-ca.tsv`

| Algorithm | Engine | Status | Build (s) | Execution Latency (ms) | Result Size | Speedup vs NetworkX |
|---|---|---|---|---|---|---|
| pagerank | benostreamdb | ✅ Pass | 15.058s | 2840.96 ms | 1965206 | 1.26x |
| pagerank | networkx | ✅ Pass | 8.557s | 3578.73 ms | 1965206 | 1.00x (baseline) |
| pagerank | neo4j | ✅ Pass | 87.502s | 32264.40 ms | 1965206 | 9× slower |
| connected_components | benostreamdb | ✅ Pass | 15.006s | 654.12 ms | 2638 | 1.54x |
| connected_components | networkx | ✅ Pass | 7.186s | 1009.16 ms | 2638 | 1.00x (baseline) |
| connected_components | neo4j | ✅ Pass | 87.502s | 31379.62 ms | 1965206 | 31× slower |
| shortest_path | benostreamdb | ✅ Pass | 14.720s | 0.07 ms | 13 | 5.57x |
| shortest_path | networkx | ✅ Pass | 8.439s | 0.38 ms | 13 | 1.00x (baseline) |
| shortest_path | neo4j | ✅ Pass | 87.502s | 12.15 ms | 1 | 32× slower |
