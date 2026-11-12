# Graph Benchmark Results

- **Graph Nodes**: 1,965,206
- **Graph Edges**: 5,533,214
- **Source Dataset**: `snap-roadnet-ca.tsv`

| Algorithm | Engine | Status | Build (s) | Execution Latency (ms) | Result Size | Speedup vs NetworkX |
|---|---|---|---|---|---|---|
| pagerank | benostreamdb | ✅ Pass | 16.581s | 2928.82 ms | 1965206 | 1.31x |
| pagerank | networkx | ✅ Pass | 8.536s | 3842.94 ms | 1965206 | 1.00x (baseline) |
| pagerank | neo4j | ✅ Pass | 88.385s | 31487.13 ms | 1965206 | 8× slower |
| connected_components | benostreamdb | ✅ Pass | 15.007s | 603.72 ms | 2638 | 1.91x |
| connected_components | networkx | ✅ Pass | 7.515s | 1152.97 ms | 2638 | 1.00x (baseline) |
| connected_components | neo4j | ✅ Pass | 88.385s | 32714.56 ms | 1965206 | 28× slower |
| shortest_path | benostreamdb | ✅ Pass | 15.741s | 0.08 ms | 13 | 5.01x |
| shortest_path | networkx | ✅ Pass | 8.664s | 0.39 ms | 13 | 1.00x (baseline) |
| shortest_path | neo4j | ✅ Pass | 88.385s | 15.24 ms | 1 | 39× slower |
