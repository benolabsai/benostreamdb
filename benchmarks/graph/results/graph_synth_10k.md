# Graph Benchmark Results

- **Graph Nodes**: 10,000
- **Graph Edges**: 49,975
- **Source Dataset**: `synth_graph_n10000_e50000_s42.txt`

| Algorithm | Engine | Status | Build (s) | Execution Latency (ms) | Result Size | Speedup vs NetworkX |
|---|---|---|---|---|---|---|
| pagerank | benostreamdb | ✅ Pass | 0.161s | 11.25 ms | 10000 | 7.29x |
| pagerank | networkx | ✅ Pass | 0.085s | 81.93 ms | 10000 | 1.00x (baseline) |
| pagerank | neo4j | ✅ Pass | 1.616s | 278.49 ms | 10000 | 3× slower |
| connected_components | benostreamdb | ✅ Pass | 0.164s | 2.78 ms | 1 | 2.54x |
| connected_components | networkx | ✅ Pass | 0.095s | 7.06 ms | 1 | 1.00x (baseline) |
| connected_components | neo4j | ✅ Pass | 1.616s | 216.87 ms | 10000 | 31× slower |
| shortest_path | benostreamdb | ✅ Pass | 0.171s | 0.05 ms | 3 | 8.60x |
| shortest_path | networkx | ✅ Pass | 0.078s | 0.40 ms | 3 | 1.00x (baseline) |
| shortest_path | neo4j | ✅ Pass | 1.616s | 82.28 ms | 1 | 207× slower |
