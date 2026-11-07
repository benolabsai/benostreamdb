# Graph Benchmark Results

- **Graph Nodes**: 1,965,206
- **Graph Edges**: 5,533,214
- **Source Dataset**: `snap-roadnet-ca.tsv`

| Algorithm | Engine | Status | Build (s) | Execution Latency (ms) | Result Size | Speedup vs NetworkX |
|---|---|---|---|---|---|---|
| pagerank | benostreamdb | ✅ Pass | 13.601s | 2648.42 ms | 1965206 | 1.35x |
| pagerank | networkx | ✅ Pass | 8.390s | 3567.32 ms | 1965206 | 1.00x (baseline) |
| pagerank | neo4j | ✅ Pass | 85.928s | 30538.56 ms | 1965206 | 9× slower |
| connected_components | benostreamdb | ✅ Pass | 13.675s | 560.54 ms | 2638 | 1.68x |
| connected_components | networkx | ✅ Pass | 7.011s | 942.40 ms | 2638 | 1.00x (baseline) |
| connected_components | neo4j | ✅ Pass | 85.928s | 30190.65 ms | 1965206 | 32× slower |
| shortest_path | benostreamdb | ✅ Pass | 13.798s | 0.07 ms | 13 | 5.40x |
| shortest_path | networkx | ✅ Pass | 8.345s | 0.38 ms | 13 | 1.00x (baseline) |
| shortest_path | neo4j | ✅ Pass | 85.928s | 17.57 ms | 1 | 46× slower |
