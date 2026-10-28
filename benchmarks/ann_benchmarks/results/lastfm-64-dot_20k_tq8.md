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
| recall@10 | 0.9108 |
| QPS | 1206.2 |
| Build time | 1.2s |
| Index size | 19.5 MB |
| p50 latency | 0.63 ms |
| p99 latency | 0.99 ms |
| Pure Index QPS | 2327.1 |
| Pure Index p50 latency | 0.42 ms |
| Pure Index p99 latency | 0.63 ms |
