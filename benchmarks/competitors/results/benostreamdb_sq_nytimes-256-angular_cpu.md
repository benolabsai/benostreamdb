# ANN-Benchmarks: nytimes-256-angular

| Metric | Value |
|---|---|
| Dataset | nytimes-256-angular |
| Train | 20000 x 256 |
| Queries | 500 |
| k | 10 |
| Metric | cosine |
| Index | hnsw_sq8 |
| M (complexity) | 16 |
| ef_construction (quality) | 200 |
| ef_search | 200 |
| refine_factor | 1 |
| Cores | 8 |
| RAM (GB) | 16.0 |
| recall@10 | 0.9788 |
| QPS | 560.3 |
| Build time | 2.9s |
| Index size | 52.7 MB |
| p50 latency | 1.63 ms |
| p99 latency | 2.14 ms |
| Pure Index QPS | 792.5 |
| Pure Index p50 latency | 1.24 ms |
| Pure Index p99 latency | 1.60 ms |
