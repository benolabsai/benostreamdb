# Graph Benchmark Results

- **Graph Nodes**: 875,713
- **Graph Edges**: 5,105,039
- **Source Dataset**: `snap-web-google.tsv`

| Algorithm | Engine | Status | Build (s) | Execution Latency (ms) | Result Size | Speedup vs NetworkX |
|---|---|---|---|---|---|---|
| pagerank | benostreamdb | ✅ Pass | 14.532s | 1680.61 ms | 875713 | 1.87x |
| pagerank | networkx | ✅ Pass | 8.706s | 3135.56 ms | 875713 | 1.00x (baseline) |
| pagerank | neo4j | ✅ Pass | 106.414s | 13660.53 ms | 875713 | 4× slower |
| connected_components | benostreamdb | ✅ Pass | 14.555s | 774.09 ms | 2746 | 1.53x |
| connected_components | networkx | ✅ Pass | 8.344s | 1183.56 ms | 2746 | 1.00x (baseline) |
| connected_components | neo4j | ✅ Pass | 106.414s | 14206.43 ms | 875713 | 12× slower |
| shortest_path | benostreamdb | ✅ Pass | 15.066s | 4.47 ms | 14 | 1.28x |
| shortest_path | networkx | ✅ Pass | 8.520s | 5.71 ms | 14 | 1.00x (baseline) |
| shortest_path | neo4j | ✅ Pass | 106.414s | 311.97 ms | 1 | 55× slower |
