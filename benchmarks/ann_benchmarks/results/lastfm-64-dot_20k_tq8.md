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
| recall@10 | 0.8834 |
| QPS | 1038.7 |
| Build time | 2.5s |
| Index size | 19.4 MB |
| p50 latency | 0.79 ms |
| p99 latency | 4.98 ms |
| Pure Index QPS | 1420.0 |
| Pure Index p50 latency | 0.68 ms |
| Pure Index p99 latency | 1.06 ms |
