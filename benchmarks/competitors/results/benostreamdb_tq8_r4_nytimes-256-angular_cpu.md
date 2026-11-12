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
| refine_factor | 4 |
| Cores | 8 |
| RAM (GB) | 16.0 |
| recall@10 | 0.9854 |
| QPS | 365.2 |
| Build time | 1.6s |
| Index size | 52.7 MB |
| p50 latency | 2.43 ms |
| p99 latency | 3.24 ms |
| Pure Index QPS | 840.8 |
| Pure Index p50 latency | 1.17 ms |
| Pure Index p99 latency | 1.52 ms |
