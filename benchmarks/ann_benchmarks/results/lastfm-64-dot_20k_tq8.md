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
| recall@10 | 0.9582 |
| QPS | 832.2 |
| Build time | 3.6s |
| Index size | 19.4 MB |
| p50 latency | 1.03 ms |
| p99 latency | 4.86 ms |
| Pure Index QPS | 1078.4 |
| Pure Index p50 latency | 0.89 ms |
| Pure Index p99 latency | 1.64 ms |
