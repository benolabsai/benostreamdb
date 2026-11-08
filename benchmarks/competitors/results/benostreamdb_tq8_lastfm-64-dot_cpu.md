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
| Cores | 8 |
| RAM (GB) | 16.0 |
| recall@10 | 0.9526 |
| QPS | 1084.7 |
| Build time | 2.7s |
| Index size | 19.4 MB |
| p50 latency | 0.72 ms |
| p99 latency | 0.96 ms |
| Pure Index QPS | 1846.6 |
| Pure Index p50 latency | 0.54 ms |
| Pure Index p99 latency | 0.65 ms |
