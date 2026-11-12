# Graph Benchmark Results

- **Graph Nodes**: 1,965,206
- **Graph Edges**: 5,533,214
- **Source Dataset**: `snap-roadnet-ca.tsv`

| Algorithm | Engine | Status | Build (s) | Execution Latency (ms) | Result Size | Speedup vs NetworkX |
|---|---|---|---|---|---|---|
| pagerank | benostreamdb | ✅ Pass | 15.215s | 3107.34 ms | 1965206 | 1.30x |
| pagerank | networkx | ✅ Pass | 9.082s | 4030.59 ms | 1965206 | 1.00x (baseline) |
| pagerank | neo4j | ✅ Pass | 89.002s | 33488.85 ms | 1965206 | 8× slower |
| connected_components | benostreamdb | ✅ Pass | 15.105s | 689.28 ms | 2638 | 1.58x |
| connected_components | networkx | ✅ Pass | 7.586s | 1087.69 ms | 2638 | 1.00x (baseline) |
| connected_components | neo4j | ✅ Pass | 89.002s | 31887.24 ms | 1965206 | 29× slower |
| shortest_path | benostreamdb | ✅ Pass | 16.493s | 0.08 ms | 13 | 4.99x |
| shortest_path | networkx | ✅ Pass | 8.682s | 0.39 ms | 13 | 1.00x (baseline) |
| shortest_path | neo4j | ✅ Pass | 89.002s | 26.99 ms | 1 | 69× slower |
