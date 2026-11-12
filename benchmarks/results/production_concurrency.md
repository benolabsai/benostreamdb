# Multi-Client Concurrency Scaling Benchmark

- **Engine**: BenoStreamDB
- **Host**: x86_64 (Linux)
- **Workload**: Concurrent ANN Vector Queries (HNSW-TQ8, Top-10)
- **Queries per Concurrency Tier**: 1,000

| Concurrency (Threads) | Throughput (QPS) | Scaling Speedup | p50 Latency | p90 Latency | p99 Latency |
|---|---|---|---|---|---|
| **1** | **589.5** | **1.00x** | 1.63 ms | 1.84 ms | 2.43 ms |
| **2** | **1092.3** | **1.85x** | 1.72 ms | 2.07 ms | 2.66 ms |
| **4** | **1519.1** | **2.58x** | 2.49 ms | 3.05 ms | 3.83 ms |
| **8** | **1412.7** | **2.40x** | 5.44 ms | 7.21 ms | 9.02 ms |
| **16** | **1294.0** | **2.20x** | 11.14 ms | 15.99 ms | 19.86 ms |
| **32** | **1292.6** | **2.19x** | 7.48 ms | 11.67 ms | 30.00 ms |
