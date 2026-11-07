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
| recall@10 | 0.8862 |
| QPS | 1130.9 |
| Build time | 1.3s |
| Index size | 52.7 MB |
| p50 latency | 0.69 ms |
| p99 latency | 1.75 ms |
| Pure Index QPS | 1602.7 |
| Pure Index p50 latency | 0.58 ms |
| Pure Index p99 latency | 1.70 ms |
