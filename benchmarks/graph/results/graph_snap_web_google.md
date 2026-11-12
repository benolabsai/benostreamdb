# Graph Benchmark Results

- **Graph Nodes**: 875,713
- **Graph Edges**: 5,105,039
- **Source Dataset**: `snap-web-google.tsv`

| Algorithm | Engine | Status | Build (s) | Execution Latency (ms) | Result Size | Speedup vs NetworkX |
|---|---|---|---|---|---|---|
| pagerank | benostreamdb | ✅ Pass | 14.390s | 1707.76 ms | 875713 | 1.86x |
| pagerank | networkx | ✅ Pass | 8.768s | 3169.81 ms | 875713 | 1.00x (baseline) |
| pagerank | neo4j | ✅ Pass | 102.359s | 14605.43 ms | 875713 | 5× slower |
| connected_components | benostreamdb | ✅ Pass | 14.263s | 617.65 ms | 2746 | 1.90x |
| connected_components | networkx | ✅ Pass | 8.296s | 1174.84 ms | 2746 | 1.00x (baseline) |
| connected_components | neo4j | ✅ Pass | 102.359s | 14350.01 ms | 875713 | 12× slower |
| shortest_path | benostreamdb | ✅ Pass | 14.348s | 4.20 ms | 14 | 1.39x |
| shortest_path | networkx | ✅ Pass | 8.528s | 5.85 ms | 14 | 1.00x (baseline) |
| shortest_path | neo4j | ✅ Pass | 102.359s | 278.57 ms | 1 | 48× slower |
