# Graph Benchmark Results

- **Graph Nodes**: 10,000
- **Graph Edges**: 49,975
- **Source Dataset**: `synth_graph_n10000_e50000_s42.txt`

| Algorithm | Engine | Status | Build (s) | Execution Latency (ms) | Result Size | Speedup vs NetworkX |
|---|---|---|---|---|---|---|
| pagerank | benostreamdb | ✅ Pass | 0.140s | 16.95 ms | 10000 | 5.11x |
| pagerank | networkx | ✅ Pass | 0.082s | 86.63 ms | 10000 | 1.00x (baseline) |
| connected_components | benostreamdb | ✅ Pass | 0.116s | 4.51 ms | 1 | 1.27x |
| connected_components | networkx | ✅ Pass | 0.076s | 5.75 ms | 1 | 1.00x (baseline) |
| shortest_path | benostreamdb | ✅ Pass | 0.127s | 1.55 ms | 3 | 0.24x |
| shortest_path | networkx | ✅ Pass | 0.067s | 0.38 ms | 3 | 1.00x (baseline) |
