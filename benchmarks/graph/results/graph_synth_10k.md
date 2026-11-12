# Graph Benchmark Results

- **Graph Nodes**: 10,000
- **Graph Edges**: 49,975
- **Source Dataset**: `synth_graph_n10000_e50000_s42.txt`

| Algorithm | Engine | Status | Build (s) | Execution Latency (ms) | Result Size | Speedup vs NetworkX |
|---|---|---|---|---|---|---|
| pagerank | benostreamdb | ✅ Pass | 0.158s | 12.20 ms | 10000 | 10.65x |
| pagerank | networkx | ✅ Pass | 0.084s | 129.91 ms | 10000 | 1.00x (baseline) |
| pagerank | neo4j | ✅ Pass | 1.720s | 280.40 ms | 10000 | 2× slower |
| connected_components | benostreamdb | ✅ Pass | 0.165s | 4.03 ms | 1 | 1.63x |
| connected_components | networkx | ✅ Pass | 0.091s | 6.57 ms | 1 | 1.00x (baseline) |
| connected_components | neo4j | ✅ Pass | 1.720s | 226.85 ms | 10000 | 35× slower |
| shortest_path | benostreamdb | ✅ Pass | 0.165s | 0.04 ms | 3 | 9.29x |
| shortest_path | networkx | ✅ Pass | 0.078s | 0.40 ms | 3 | 1.00x (baseline) |
| shortest_path | neo4j | ✅ Pass | 1.720s | 75.71 ms | 1 | 187× slower |
