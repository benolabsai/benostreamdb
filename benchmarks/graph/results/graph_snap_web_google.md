# Graph Benchmark Results

- **Graph Nodes**: 875,713
- **Graph Edges**: 5,105,039
- **Source Dataset**: `snap-web-google.tsv`

| Algorithm | Engine | Status | Build (s) | Execution Latency (ms) | Result Size | Speedup vs NetworkX |
|---|---|---|---|---|---|---|
| pagerank | benostreamdb | ✅ Pass | 14.393s | 1860.95 ms | 875713 | 1.74x |
| pagerank | networkx | ✅ Pass | 8.905s | 3235.44 ms | 875713 | 1.00x (baseline) |
| pagerank | neo4j | ✅ Pass | 100.480s | 14045.77 ms | 875713 | 4× slower |
| connected_components | benostreamdb | ✅ Pass | 14.288s | 775.43 ms | 2746 | 1.50x |
| connected_components | networkx | ✅ Pass | 8.272s | 1166.31 ms | 2746 | 1.00x (baseline) |
| connected_components | neo4j | ✅ Pass | 100.480s | 14148.93 ms | 875713 | 12× slower |
| shortest_path | benostreamdb | ✅ Pass | 14.587s | 1.98 ms | 14 | 2.89x |
| shortest_path | networkx | ✅ Pass | 8.117s | 5.73 ms | 14 | 1.00x (baseline) |
| shortest_path | neo4j | ✅ Pass | 100.480s | 258.91 ms | 1 | 45× slower |
