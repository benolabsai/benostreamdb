# ANN-Benchmarks: lastfm-64-dot

| Metric | Value |
|---|---|
| Dataset | lastfm-64-dot |
| Train | 20000 x 65 |
| Queries | 500 |
| k | 10 |
| Metric | inner_product |
| Index | hnsw_tq8 |
| M (complexity) | 16 |
| ef_construction (quality) | 200 |
| ef_search | 200 |
| refine_factor | 1 |
| recall@10 | 0.9570 |
| QPS | 682.1 |
| Build time | 2.8s |
| Index size | 19.4 MB |
| p50 latency | 1.31 ms |
| p99 latency | 2.02 ms |
| Pure Index QPS | 1126.1 |
| Pure Index p50 latency | 0.86 ms |
| Pure Index p99 latency | 1.45 ms |
