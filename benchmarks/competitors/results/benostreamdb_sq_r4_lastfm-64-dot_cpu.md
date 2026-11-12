# ANN-Benchmarks: lastfm-64-dot

| Metric | Value |
|---|---|
| Dataset | lastfm-64-dot |
| Train | 20000 x 65 |
| Queries | 500 |
| k | 10 |
| Metric | inner_product |
| Index | hnsw_sq8 |
| M (complexity) | 16 |
| ef_construction (quality) | 200 |
| ef_search | 200 |
| refine_factor | 4 |
| Cores | 8 |
| RAM (GB) | 16.0 |
| recall@10 | 0.9730 |
| QPS | 513.4 |
| Build time | 6.0s |
| Index size | 18.2 MB |
| p50 latency | 1.72 ms |
| p99 latency | 2.32 ms |
| Pure Index QPS | 1730.5 |
| Pure Index p50 latency | 0.57 ms |
| Pure Index p99 latency | 0.80 ms |
