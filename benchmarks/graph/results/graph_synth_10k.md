# Graph Benchmark Results

- **Graph Nodes**: 10,000
- **Graph Edges**: 49,975
- **Source Dataset**: `synth_graph_n10000_e50000_s42.txt`

| Algorithm | Engine | Status | Build (s) | Execution Latency (ms) | Result Size | Speedup vs NetworkX |
|---|---|---|---|---|---|---|
| pagerank | benostreamdb | ✅ Pass | 0.148s | 11.68 ms | 10000 | 6.22x |
| pagerank | networkx | ✅ Pass | 0.079s | 72.59 ms | 10000 | 1.00x (baseline) |
| pagerank | neo4j | ✅ Pass | 1.585s | 268.41 ms | 10000 | 4× slower |
| connected_components | benostreamdb | ✅ Pass | 0.150s | 2.22 ms | 1 | 2.78x |
| connected_components | networkx | ✅ Pass | 0.073s | 6.17 ms | 1 | 1.00x (baseline) |
| connected_components | neo4j | ✅ Pass | 1.585s | 208.75 ms | 10000 | 34× slower |
| shortest_path | benostreamdb | ✅ Pass | 0.150s | 0.04 ms | 3 | 8.36x |
| shortest_path | networkx | ✅ Pass | 0.069s | 0.32 ms | 3 | 1.00x (baseline) |
| shortest_path | neo4j | ✅ Pass | 1.585s | 75.72 ms | 1 | 235× slower |
