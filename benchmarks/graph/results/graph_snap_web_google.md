# Graph Benchmark Results

- **Graph Nodes**: 158,508
- **Graph Edges**: 500,000
- **Source Dataset**: `snap-web-google_500000.tsv`

| Algorithm | Engine | Status | Build (s) | Execution Latency (ms) | Result Size | Speedup vs NetworkX |
|---|---|---|---|---|---|---|
| pagerank | benostreamdb | ✅ Pass | 1.352s | 290.56 ms | 158508 | 1.48x |
| pagerank | networkx | ✅ Pass | 0.842s | 430.71 ms | 158508 | 1.00x (baseline) |
| connected_components | benostreamdb | ✅ Pass | 1.266s | 54.30 ms | 1383 | 1.93x |
| connected_components | networkx | ✅ Pass | 0.719s | 104.60 ms | 1383 | 1.00x (baseline) |
