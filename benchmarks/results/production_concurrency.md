# Multi-Client Concurrency Scaling Benchmark

- **Engine**: BenoStreamDB
- **Host**: x86_64 (Linux)
- **Workload**: Concurrent ANN Vector Queries (HNSW-TQ8, Top-10)
- **Queries per Concurrency Tier**: 1,000

| Concurrency (Threads) | Throughput (QPS) | Scaling Speedup | p50 Latency | p90 Latency | p99 Latency |
|---|---|---|---|---|---|
| **1** | **1251.7** | **1.00x** | 0.77 ms | 0.85 ms | 1.03 ms |
| **2** | **2492.7** | **1.99x** | 0.77 ms | 0.85 ms | 0.98 ms |
| **4** | **3285.4** | **2.62x** | 1.16 ms | 1.41 ms | 1.72 ms |
| **8** | **2828.3** | **2.26x** | 2.69 ms | 3.49 ms | 4.60 ms |
| **16** | **2620.5** | **2.09x** | 5.18 ms | 7.69 ms | 9.89 ms |
| **32** | **2407.1** | **1.92x** | 7.70 ms | 15.55 ms | 22.33 ms |
