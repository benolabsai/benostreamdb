# Graph Benchmark Results

- **Graph Nodes**: 875,713
- **Graph Edges**: 5,105,039
- **Source Dataset**: `snap-web-google.tsv`

| Algorithm | Engine | Status | Build (s) | Execution Latency (ms) | Result Size | Speedup vs NetworkX |
|---|---|---|---|---|---|---|
| pagerank | benostreamdb | ✅ Pass | 12.971s | 1585.76 ms | 875713 | 1.77x |
| pagerank | networkx | ✅ Pass | 8.124s | 2806.53 ms | 875713 | 1.00x (baseline) |
| pagerank | neo4j | ✅ Pass | 99.864s | 13820.54 ms | 875713 | 5× slower |
| connected_components | benostreamdb | ✅ Pass | 12.907s | 534.38 ms | 2746 | 2.07x |
| connected_components | networkx | ✅ Pass | 7.927s | 1105.61 ms | 2746 | 1.00x (baseline) |
| connected_components | neo4j | ✅ Pass | 99.864s | 13431.46 ms | 875713 | 12× slower |
| shortest_path | benostreamdb | ✅ Pass | 12.718s | 3.97 ms | 14 | 1.31x |
| shortest_path | networkx | ✅ Pass | 7.920s | 5.21 ms | 14 | 1.00x (baseline) |
| shortest_path | neo4j | ✅ Pass | 99.864s | 229.13 ms | 1 | 44× slower |
