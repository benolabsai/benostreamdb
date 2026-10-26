# Multi-Client Concurrency Scaling Benchmark

- **Engine**: BenoStreamDB
- **Host**: x86_64 (Linux)
- **Workload**: Concurrent ANN Vector Queries (HNSW-TQ8, Top-10)
- **Queries per Concurrency Tier**: 500

| Concurrency (Threads) | Throughput (QPS) | Scaling Speedup | p50 Latency | p90 Latency | p99 Latency |
|---|---|---|---|---|---|
| **1** | **909.9** | **1.00x** | 1.05 ms | 1.17 ms | 1.50 ms |
| **2** | **1725.4** | **1.90x** | 1.09 ms | 1.31 ms | 1.62 ms |
| **4** | **2385.7** | **2.62x** | 1.60 ms | 1.94 ms | 2.42 ms |
| **8** | **2222.0** | **2.44x** | 3.40 ms | 4.31 ms | 6.26 ms |
| **16** | **1996.1** | **2.19x** | 7.21 ms | 10.01 ms | 13.05 ms |
| **32** | **1838.3** | **2.02x** | 4.49 ms | 7.37 ms | 13.49 ms |
