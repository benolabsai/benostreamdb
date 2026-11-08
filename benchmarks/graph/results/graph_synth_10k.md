# Graph Benchmark Results

- **Graph Nodes**: 10,000
- **Graph Edges**: 49,975
- **Source Dataset**: `synth_graph_n10000_e50000_s42.txt`

| Algorithm | Engine | Status | Build (s) | Execution Latency (ms) | Result Size | Speedup vs NetworkX |
|---|---|---|---|---|---|---|
| pagerank | benostreamdb | ✅ Pass | 0.333s | 97.86 ms | 10000 | 0.77x |
| pagerank | networkx | ✅ Pass | 0.081s | 74.91 ms | 10000 | 1.00x (baseline) |
| pagerank | neo4j | ✅ Pass | 1.646s | 259.36 ms | 10000 | 3× slower |
| connected_components | benostreamdb | ✅ Pass | 0.342s | 13.85 ms | 1 | 2× slower |
| connected_components | networkx | ✅ Pass | 0.071s | 5.67 ms | 1 | 1.00x (baseline) |
| connected_components | neo4j | ✅ Pass | 1.646s | 175.70 ms | 10000 | 31× slower |
| shortest_path | benostreamdb | ✅ Pass | 0.295s | 0.60 ms | 3 | 0.60x |
| shortest_path | networkx | ✅ Pass | 0.067s | 0.36 ms | 3 | 1.00x (baseline) |
| shortest_path | neo4j | ✅ Pass | 1.646s | 74.77 ms | 1 | 207× slower |
