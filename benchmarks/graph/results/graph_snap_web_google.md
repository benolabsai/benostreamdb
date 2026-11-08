# Graph Benchmark Results

- **Graph Nodes**: 875,713
- **Graph Edges**: 5,105,039
- **Source Dataset**: `snap-web-google.tsv`

| Algorithm | Engine | Status | Build (s) | Execution Latency (ms) | Result Size | Speedup vs NetworkX |
|---|---|---|---|---|---|---|
| pagerank | benostreamdb | ✅ Pass | 33.697s | 10047.38 ms | 875713 | 4× slower |
| pagerank | networkx | ✅ Pass | 7.957s | 2802.97 ms | 875713 | 1.00x (baseline) |
| pagerank | neo4j | ✅ Pass | 100.861s | 13719.13 ms | 875713 | 5× slower |
| connected_components | benostreamdb | ✅ Pass | 33.885s | 1931.39 ms | 2746 | 0.55x |
| connected_components | networkx | ✅ Pass | 7.911s | 1071.21 ms | 2746 | 1.00x (baseline) |
| connected_components | neo4j | ✅ Pass | 100.861s | 12679.23 ms | 875713 | 12× slower |
| shortest_path | benostreamdb | ✅ Pass | 33.787s | 14.94 ms | 14 | 3× slower |
| shortest_path | networkx | ✅ Pass | 7.660s | 5.53 ms | 14 | 1.00x (baseline) |
| shortest_path | neo4j | ✅ Pass | 100.861s | 247.64 ms | 1 | 45× slower |
