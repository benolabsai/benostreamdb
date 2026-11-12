# Multi-Client Concurrency Scaling Benchmark

- **Engine**: BenoStreamDB
- **Host**: x86_64 (Linux)
- **Workload**: Concurrent ANN Vector Queries (HNSW-TQ8, Top-10)
- **Queries per Concurrency Tier**: 1,000

| Concurrency (Threads) | Throughput (QPS) | Scaling Speedup | p50 Latency | p90 Latency | p99 Latency |
|---|---|---|---|---|---|
| **1** | **554.4** | **1.00x** | 1.73 ms | 2.05 ms | 2.59 ms |
| **2** | **1048.7** | **1.89x** | 1.81 ms | 2.21 ms | 2.77 ms |
| **4** | **1523.0** | **2.75x** | 2.51 ms | 3.09 ms | 3.89 ms |
| **8** | **1382.0** | **2.49x** | 5.59 ms | 7.19 ms | 8.98 ms |
| **16** | **1216.1** | **2.19x** | 11.44 ms | 16.63 ms | 23.05 ms |
| **32** | **1164.9** | **2.10x** | 8.50 ms | 14.48 ms | 32.02 ms |
