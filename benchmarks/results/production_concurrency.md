# Multi-Client Concurrency Scaling Benchmark

- **Engine**: BenoStreamDB
- **Host**: x86_64 (Linux)
- **Workload**: Concurrent ANN Vector Queries (HNSW-TQ8, Top-10)
- **Queries per Concurrency Tier**: 1,000

| Concurrency (Threads) | Throughput (QPS) | Scaling Speedup | p50 Latency | p90 Latency | p99 Latency |
|---|---|---|---|---|---|
| **1** | **1132.2** | **1.00x** | 0.85 ms | 0.96 ms | 1.22 ms |
| **2** | **2108.0** | **1.86x** | 0.90 ms | 1.06 ms | 1.30 ms |
| **4** | **2792.7** | **2.47x** | 1.37 ms | 1.71 ms | 2.03 ms |
| **8** | **2512.3** | **2.22x** | 3.03 ms | 3.98 ms | 5.08 ms |
| **16** | **2318.6** | **2.05x** | 4.38 ms | 7.76 ms | 10.77 ms |
| **32** | **2059.3** | **1.82x** | 4.59 ms | 11.02 ms | 19.62 ms |
