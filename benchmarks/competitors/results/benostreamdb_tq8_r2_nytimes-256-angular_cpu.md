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
| refine_factor | 2 |
| Cores | 8 |
| RAM (GB) | 16.0 |
| recall@10 | 0.9854 |
| QPS | 483.6 |
| Build time | 1.5s |
| Index size | 52.7 MB |
| p50 latency | 2.04 ms |
| p99 latency | 2.52 ms |
| Pure Index QPS | 873.5 |
| Pure Index p50 latency | 1.13 ms |
| Pure Index p99 latency | 1.45 ms |
