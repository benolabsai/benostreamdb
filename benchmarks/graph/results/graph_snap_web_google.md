# Graph Benchmark Results

- **Graph Nodes**: 875,713
- **Graph Edges**: 5,105,039
- **Source Dataset**: `snap-web-google.tsv`

| Algorithm | Engine | Status | Build (s) | Execution Latency (ms) | Result Size | Speedup vs NetworkX |
|---|---|---|---|---|---|---|
| pagerank | benostreamdb | ✅ Pass | 14.380s | 1671.64 ms | 875713 | 1.85x |
| pagerank | networkx | ✅ Pass | 8.530s | 3096.92 ms | 875713 | 1.00x (baseline) |
| pagerank | neo4j | ✅ Pass | 103.101s | 14145.95 ms | 875713 | 5× slower |
| connected_components | benostreamdb | ✅ Pass | 14.302s | 741.49 ms | 2746 | 1.60x |
| connected_components | networkx | ✅ Pass | 8.344s | 1187.57 ms | 2746 | 1.00x (baseline) |
| connected_components | neo4j | ✅ Pass | 103.101s | 14174.65 ms | 875713 | 12× slower |
| shortest_path | benostreamdb | ✅ Pass | 15.371s | 3.55 ms | 14 | 1.68x |
| shortest_path | networkx | ✅ Pass | 8.138s | 5.96 ms | 14 | 1.00x (baseline) |
| shortest_path | neo4j | ✅ Pass | 103.101s | 243.15 ms | 1 | 41× slower |
