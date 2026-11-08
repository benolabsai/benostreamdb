# Graph Benchmark Results

- **Graph Nodes**: 10,000
- **Graph Edges**: 49,975
- **Source Dataset**: `synth_graph_n10000_e50000_s42.txt`

| Algorithm | Engine | Status | Build (s) | Execution Latency (ms) | Result Size | Speedup vs NetworkX |
|---|---|---|---|---|---|---|
| pagerank | benostreamdb | ✅ Pass | 0.143s | 10.30 ms | 10000 | 8.27x |
| pagerank | networkx | ✅ Pass | 0.079s | 85.17 ms | 10000 | 1.00x (baseline) |
| pagerank | neo4j | ✅ Pass | 1.538s | 263.05 ms | 10000 | 3× slower |
| connected_components | benostreamdb | ✅ Pass | 0.150s | 2.40 ms | 1 | 2.57x |
| connected_components | networkx | ✅ Pass | 0.088s | 6.18 ms | 1 | 1.00x (baseline) |
| connected_components | neo4j | ✅ Pass | 1.538s | 205.06 ms | 10000 | 33× slower |
| shortest_path | benostreamdb | ✅ Pass | 0.147s | 0.04 ms | 3 | 9.60x |
| shortest_path | networkx | ✅ Pass | 0.071s | 0.41 ms | 3 | 1.00x (baseline) |
| shortest_path | neo4j | ✅ Pass | 1.538s | 71.48 ms | 1 | 174× slower |
