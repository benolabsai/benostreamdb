# Multi-Client Concurrency Scaling Benchmark

- **Engine**: BenoStreamDB
- **Host**: x86_64 (Linux)
- **Workload**: Concurrent ANN Vector Queries (HNSW-TQ8, Top-10)
- **Queries per Concurrency Tier**: 1,000

| Concurrency (Threads) | Throughput (QPS) | Scaling Speedup | p50 Latency | p90 Latency | p99 Latency |
|---|---|---|---|---|---|
| **1** | **755.7** | **1.00x** | 1.21 ms | 1.60 ms | 2.37 ms |
| **2** | **1314.3** | **1.74x** | 1.43 ms | 1.79 ms | 2.77 ms |
| **4** | **1617.3** | **2.14x** | 2.38 ms | 3.02 ms | 3.84 ms |
| **8** | **1452.1** | **1.92x** | 5.32 ms | 7.13 ms | 9.27 ms |
| **16** | **1359.8** | **1.80x** | 7.83 ms | 13.44 ms | 18.79 ms |
| **32** | **1270.4** | **1.68x** | 12.39 ms | 30.03 ms | 40.69 ms |
