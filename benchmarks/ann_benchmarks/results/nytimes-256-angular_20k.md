# ANN-Benchmarks: nytimes-256-angular

| Metric | Value |
|---|---|
| Dataset | nytimes-256-angular |
| Train | 20000 x 256 |
| Queries | 500 |
| k | 10 |
| Metric | cosine |
| Index | hnsw |
| M (complexity) | 16 |
| ef_construction (quality) | 200 |
| ef_search | 200 |
| refine_factor | 1 |
| recall@10 | 0.9854 |
| QPS | 749.1 |
| Build time | 1.4s |
| Index size | 68.0 MB |
| p50 latency | 1.15 ms |
| p99 latency | 1.93 ms |
| Pure Index QPS | 1208.9 |
| Pure Index p50 latency | 0.79 ms |
| Pure Index p99 latency | 1.75 ms |
