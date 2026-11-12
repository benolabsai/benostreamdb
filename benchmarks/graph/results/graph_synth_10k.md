# Graph Benchmark Results

- **Graph Nodes**: 10,000
- **Graph Edges**: 49,975
- **Source Dataset**: `synth_graph_n10000_e50000_s42.txt`

| Algorithm | Engine | Status | Build (s) | Execution Latency (ms) | Result Size | Speedup vs NetworkX |
|---|---|---|---|---|---|---|
| pagerank | benostreamdb | ✅ Pass | 0.155s | 10.56 ms | 10000 | 7.40x |
| pagerank | networkx | ✅ Pass | 0.083s | 78.16 ms | 10000 | 1.00x (baseline) |
| pagerank | neo4j | ✅ Pass | 1.755s | 258.26 ms | 10000 | 3× slower |
| connected_components | benostreamdb | ✅ Pass | 0.163s | 2.45 ms | 1 | 2.84x |
| connected_components | networkx | ✅ Pass | 0.093s | 6.95 ms | 1 | 1.00x (baseline) |
| connected_components | neo4j | ✅ Pass | 1.755s | 211.91 ms | 10000 | 31× slower |
| shortest_path | benostreamdb | ✅ Pass | 0.158s | 0.04 ms | 3 | 9.70x |
| shortest_path | networkx | ✅ Pass | 0.075s | 0.41 ms | 3 | 1.00x (baseline) |
| shortest_path | neo4j | ✅ Pass | 1.755s | 75.05 ms | 1 | 185× slower |
