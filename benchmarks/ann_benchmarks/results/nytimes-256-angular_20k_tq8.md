# ANN-Benchmarks: nytimes-256-angular

| Metric | Value |
|---|---|
| Dataset | nytimes-256-angular |
| Train | 20000 x 256 |
| Queries | 500 |
| k | 10 |
| Metric | cosine |
| Index | hnsw_tq8 |
| M (complexity) | 16 |
| ef_construction (quality) | 200 |
| ef_search | 200 |
| recall@10 | 0.9656 |
| QPS | 954.8 |
| Build time | 1.1s |
| Index size | 52.7 MB |
| p50 latency | 0.82 ms |
| p99 latency | 1.35 ms |
| Pure Index QPS | 1623.2 |
| Pure Index p50 latency | 0.59 ms |
| Pure Index p99 latency | 1.20 ms |
