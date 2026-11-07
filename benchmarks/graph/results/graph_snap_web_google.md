# Graph Benchmark Results

- **Graph Nodes**: 875,713
- **Graph Edges**: 5,105,039
- **Source Dataset**: `snap-web-google.tsv`

| Algorithm | Engine | Status | Build (s) | Execution Latency (ms) | Result Size | Speedup vs NetworkX |
|---|---|---|---|---|---|---|
| pagerank | benostreamdb | ✅ Pass | 13.196s | 1718.56 ms | 875713 | 1.81x |
| pagerank | networkx | ✅ Pass | 8.228s | 3109.58 ms | 875713 | 1.00x (baseline) |
| pagerank | neo4j | ✅ Pass | 100.376s | 13538.15 ms | 875713 | 4× slower |
| connected_components | benostreamdb | ✅ Pass | 12.999s | 512.15 ms | 2746 | 2.09x |
| connected_components | networkx | ✅ Pass | 7.897s | 1070.21 ms | 2746 | 1.00x (baseline) |
| connected_components | neo4j | ✅ Pass | 100.376s | 13911.76 ms | 875713 | 13× slower |
| shortest_path | benostreamdb | ✅ Pass | 13.859s | 2.82 ms | 14 | 1.91x |
| shortest_path | networkx | ✅ Pass | 8.358s | 5.39 ms | 14 | 1.00x (baseline) |
| shortest_path | neo4j | ✅ Pass | 100.376s | 282.51 ms | 1 | 52× slower |
