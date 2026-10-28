# Graph Benchmark Results

- **Graph Nodes**: 875,713
- **Graph Edges**: 5,105,039
- **Source Dataset**: `snap-web-google.tsv`

| Algorithm | Engine | Status | Build (s) | Execution Latency (ms) | Result Size | Speedup vs NetworkX |
|---|---|---|---|---|---|---|
| pagerank | benostreamdb | ✅ Pass | 13.408s | 1797.29 ms | 875713 | 1.80x |
| pagerank | networkx | ✅ Pass | 8.687s | 3229.74 ms | 875713 | 1.00x (baseline) |
| connected_components | benostreamdb | ✅ Pass | 13.399s | 878.88 ms | 2746 | 1.29x |
| connected_components | networkx | ✅ Pass | 7.467s | 1137.95 ms | 2746 | 1.00x (baseline) |
| shortest_path | benostreamdb | ✅ Pass | 13.277s | 142.37 ms | 0 | 0.02x |
| shortest_path | networkx | ✅ Pass | 8.294s | 2.31 ms | 7 | 1.00x (baseline) |
