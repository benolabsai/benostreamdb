# Graph Benchmark Results

- **Graph Nodes**: 10,000
- **Graph Edges**: 49,975
- **Source Dataset**: `synth_graph_n10000_e50000_s42.txt`

| Algorithm | Engine | Status | Build (s) | Execution Latency (ms) | Result Size | Speedup vs NetworkX |
|---|---|---|---|---|---|---|
| pagerank | benostreamdb | ✅ Pass | 0.136s | 16.15 ms | 10000 | 4.35x |
| pagerank | networkx | ✅ Pass | 0.084s | 70.25 ms | 10000 | 1.00x (baseline) |
| connected_components | benostreamdb | ✅ Pass | 0.132s | 4.53 ms | 1 | 1.02x |
| connected_components | networkx | ✅ Pass | 0.060s | 4.62 ms | 1 | 1.00x (baseline) |
| shortest_path | benostreamdb | ✅ Pass | 0.131s | 1.69 ms | 3 | 0.23x |
| shortest_path | networkx | ✅ Pass | 0.062s | 0.39 ms | 3 | 1.00x (baseline) |
