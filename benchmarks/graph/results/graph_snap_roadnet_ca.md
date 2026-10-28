# Graph Benchmark Results

- **Graph Nodes**: 1,965,206
- **Graph Edges**: 5,533,214
- **Source Dataset**: `snap-roadnet-ca.tsv`

| Algorithm | Engine | Status | Build (s) | Execution Latency (ms) | Result Size | Speedup vs NetworkX |
|---|---|---|---|---|---|---|
| pagerank | benostreamdb | ✅ Pass | 14.040s | 2931.24 ms | 1965206 | 1.39x |
| pagerank | networkx | ✅ Pass | 8.239s | 4086.81 ms | 1965206 | 1.00x (baseline) |
| connected_components | benostreamdb | ✅ Pass | 14.389s | 835.37 ms | 2638 | 1.11x |
| connected_components | networkx | ✅ Pass | 6.972s | 931.27 ms | 2638 | 1.00x (baseline) |
| shortest_path | benostreamdb | ✅ Pass | 13.909s | 102.45 ms | 13 | 0.00x |
| shortest_path | networkx | ✅ Pass | 6.990s | 0.38 ms | 13 | 1.00x (baseline) |
