# Graph Benchmark Results

- **Graph Nodes**: 1,965,206
- **Graph Edges**: 5,533,214
- **Source Dataset**: `snap-roadnet-ca.tsv`

| Algorithm | Engine | Status | Build (s) | Execution Latency (ms) | Result Size | Speedup vs NetworkX |
|---|---|---|---|---|---|---|
| pagerank | benostreamdb | ✅ Pass | 38.554s | 15608.17 ms | 1965206 | 4× slower |
| pagerank | networkx | ✅ Pass | 8.187s | 3472.67 ms | 1965206 | 1.00x (baseline) |
| pagerank | neo4j | ✅ Pass | 81.359s | 30626.56 ms | 1965206 | 9× slower |
| connected_components | benostreamdb | ✅ Pass | 38.751s | 2475.01 ms | 2638 | 3× slower |
| connected_components | networkx | ✅ Pass | 6.945s | 906.21 ms | 2638 | 1.00x (baseline) |
| connected_components | neo4j | ✅ Pass | 81.359s | 29541.85 ms | 1965206 | 33× slower |
| shortest_path | benostreamdb | ✅ Pass | 38.226s | 0.32 ms | 13 | 1.18x |
| shortest_path | networkx | ✅ Pass | 8.139s | 0.38 ms | 13 | 1.00x (baseline) |
| shortest_path | neo4j | ✅ Pass | 81.359s | 15.18 ms | 1 | 40× slower |
