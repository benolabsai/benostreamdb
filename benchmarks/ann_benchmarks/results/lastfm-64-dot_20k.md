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
| refine_factor | 1 |
| recall@10 | 0.9722 |
| QPS | 807.4 |
| Build time | 4.8s |
| Index size | 22.1 MB |
| p50 latency | 1.06 ms |
| p99 latency | 3.60 ms |
| Pure Index QPS | 1462.5 |
| Pure Index p50 latency | 0.65 ms |
| Pure Index p99 latency | 1.52 ms |
