# Multi-Client Concurrency Scaling Benchmark

- **Engine**: BenoStreamDB
- **Host**: x86_64 (Linux)
- **Workload**: Concurrent ANN Vector Queries (HNSW-TQ8, Top-10)
- **Queries per Concurrency Tier**: 1,000

| Concurrency (Threads) | Throughput (QPS) | Scaling Speedup | p50 Latency | p90 Latency | p99 Latency |
|---|---|---|---|---|---|
| **1** | **832.9** | **1.00x** | 1.15 ms | 1.30 ms | 1.66 ms |
| **2** | **1429.5** | **1.72x** | 1.33 ms | 1.56 ms | 2.28 ms |
| **4** | **1646.4** | **1.98x** | 2.34 ms | 2.96 ms | 3.77 ms |
| **8** | **1481.6** | **1.78x** | 5.15 ms | 6.81 ms | 8.82 ms |
| **16** | **1429.5** | **1.72x** | 10.56 ms | 15.03 ms | 18.84 ms |
| **32** | **1354.4** | **1.63x** | 7.50 ms | 21.01 ms | 37.39 ms |
