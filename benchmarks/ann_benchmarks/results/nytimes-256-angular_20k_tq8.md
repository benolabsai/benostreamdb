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
| refine_factor | 1 |
| recall@10 | 0.9756 |
| QPS | 601.9 |
| Build time | 1.2s |
| Index size | 52.7 MB |
| p50 latency | 1.45 ms |
| p99 latency | 2.16 ms |
| Pure Index QPS | 876.6 |
| Pure Index p50 latency | 1.12 ms |
| Pure Index p99 latency | 1.49 ms |
