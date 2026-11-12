# Multi-Client Concurrency Scaling Benchmark

- **Engine**: BenoStreamDB
- **Host**: x86_64 (Linux)
- **Workload**: Concurrent ANN Vector Queries (HNSW-TQ8, Top-10)
- **Queries per Concurrency Tier**: 1,000

| Concurrency (Threads) | Throughput (QPS) | Scaling Speedup | p50 Latency | p90 Latency | p99 Latency |
|---|---|---|---|---|---|
| **1** | **473.2** | **1.00x** | 2.04 ms | 2.33 ms | 2.94 ms |
| **2** | **898.2** | **1.90x** | 2.14 ms | 2.49 ms | 3.07 ms |
| **4** | **1477.1** | **3.12x** | 2.58 ms | 3.09 ms | 4.09 ms |
| **8** | **1432.5** | **3.03x** | 5.37 ms | 6.86 ms | 8.43 ms |
| **16** | **1331.3** | **2.81x** | 10.77 ms | 15.11 ms | 20.37 ms |
| **32** | **1231.4** | **2.60x** | 9.34 ms | 25.62 ms | 41.32 ms |
