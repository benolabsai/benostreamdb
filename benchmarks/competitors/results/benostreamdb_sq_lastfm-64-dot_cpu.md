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
| refine_factor | 1 |
| Cores | 8 |
| RAM (GB) | 16.0 |
| recall@10 | 0.9642 |
| QPS | 1017.1 |
| Build time | 5.8s |
| Index size | 18.2 MB |
| p50 latency | 0.96 ms |
| p99 latency | 1.45 ms |
| Pure Index QPS | 1669.7 |
| Pure Index p50 latency | 0.58 ms |
| Pure Index p99 latency | 0.88 ms |
