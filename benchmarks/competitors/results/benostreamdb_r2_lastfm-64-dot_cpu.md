# ANN-Benchmarks: lastfm-64-dot

| Metric | Value |
|---|---|
| Dataset | lastfm-64-dot |
| Train | 20000 x 65 |
| Queries | 500 |
| k | 10 |
| Metric | inner_product |
| Index | hnsw |
| M (complexity) | 16 |
| ef_construction (quality) | 200 |
| ef_search | 200 |
| refine_factor | 2 |
| Cores | 8 |
| RAM (GB) | 16.0 |
| recall@10 | 0.9726 |
| QPS | 580.4 |
| Build time | 5.4s |
| Index size | 22.1 MB |
| p50 latency | 1.49 ms |
| p99 latency | 1.94 ms |
| Pure Index QPS | 1767.5 |
| Pure Index p50 latency | 0.56 ms |
| Pure Index p99 latency | 0.77 ms |
