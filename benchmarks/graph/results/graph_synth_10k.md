# Graph Benchmark Results

- **Graph Nodes**: 10,000
- **Graph Edges**: 49,975
- **Source Dataset**: `synth_graph_n10000_e50000_s42.txt`

| Algorithm | Engine | Status | Build (s) | Execution Latency (ms) | Result Size | Speedup vs NetworkX |
|---|---|---|---|---|---|---|
| pagerank | benostreamdb | ✅ Pass | 0.154s | 12.16 ms | 10000 | 6.29x |
| pagerank | networkx | ✅ Pass | 0.084s | 76.46 ms | 10000 | 1.00x (baseline) |
| pagerank | neo4j | ✅ Pass | 1.655s | 272.70 ms | 10000 | 4× slower |
| connected_components | benostreamdb | ✅ Pass | 0.161s | 2.63 ms | 1 | 3.24x |
| connected_components | networkx | ✅ Pass | 0.098s | 8.53 ms | 1 | 1.00x (baseline) |
| connected_components | neo4j | ✅ Pass | 1.655s | 211.86 ms | 10000 | 25× slower |
| shortest_path | benostreamdb | ✅ Pass | 0.182s | 0.05 ms | 3 | 9.29x |
| shortest_path | networkx | ✅ Pass | 0.082s | 0.43 ms | 3 | 1.00x (baseline) |
| shortest_path | neo4j | ✅ Pass | 1.655s | 83.21 ms | 1 | 195× slower |
