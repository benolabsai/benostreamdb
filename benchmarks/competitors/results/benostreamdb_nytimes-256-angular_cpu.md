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
| Cores | 8 |
| RAM (GB) | default |
| recall@10 | 0.2682 |
| QPS | 1241.2 |
| Build time | 0.9s |
| Index size | 49.7 MB |
| p50 latency | 0.66 ms |
| p99 latency | 0.96 ms |
| Pure Index QPS | 2408.1 |
| Pure Index p50 latency | 0.41 ms |
| Pure Index p99 latency | 0.53 ms |
