# Graph Benchmark Results

- **Graph Nodes**: 875,713
- **Graph Edges**: 5,105,039
- **Source Dataset**: `snap-web-google.tsv`

| Algorithm | Engine | Status | Build (s) | Execution Latency (ms) | Result Size | Speedup vs NetworkX |
|---|---|---|---|---|---|---|
| pagerank | benostreamdb | ✅ Pass | 12.871s | 1551.67 ms | 875713 | 1.82x |
| pagerank | networkx | ✅ Pass | 8.164s | 2822.08 ms | 875713 | 1.00x (baseline) |
| pagerank | neo4j | ✅ Pass | 102.764s | 13551.35 ms | 875713 | 5× slower |
| connected_components | benostreamdb | ✅ Pass | 13.167s | 487.47 ms | 2746 | 2.19x |
| connected_components | networkx | ✅ Pass | 7.969s | 1065.70 ms | 2746 | 1.00x (baseline) |
| connected_components | neo4j | ✅ Pass | 102.764s | 13616.74 ms | 875713 | 13× slower |
| shortest_path | benostreamdb | ✅ Pass | 12.843s | 2.35 ms | 14 | 2.28x |
| shortest_path | networkx | ✅ Pass | 7.688s | 5.34 ms | 14 | 1.00x (baseline) |
| shortest_path | neo4j | ✅ Pass | 102.764s | 253.93 ms | 1 | 48× slower |
