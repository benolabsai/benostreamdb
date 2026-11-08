# Multi-Client Concurrency Scaling Benchmark

- **Engine**: BenoStreamDB
- **Host**: x86_64 (Linux)
- **Workload**: Concurrent ANN Vector Queries (HNSW-TQ8, Top-10)
- **Queries per Concurrency Tier**: 1,000

| Concurrency (Threads) | Throughput (QPS) | Scaling Speedup | p50 Latency | p90 Latency | p99 Latency |
|---|---|---|---|---|---|
| **1** | **853.6** | **1.00x** | 1.12 ms | 1.24 ms | 1.54 ms |
| **2** | **1492.8** | **1.75x** | 1.26 ms | 1.50 ms | 2.16 ms |
| **4** | **1707.8** | **2.00x** | 2.26 ms | 2.83 ms | 3.55 ms |
| **8** | **1516.7** | **1.78x** | 5.05 ms | 6.75 ms | 8.34 ms |
| **16** | **1417.1** | **1.66x** | 9.83 ms | 14.29 ms | 19.05 ms |
| **32** | **1380.5** | **1.62x** | 13.26 ms | 28.72 ms | 37.97 ms |
