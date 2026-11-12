# Multi-Client Concurrency Scaling Benchmark

- **Engine**: BenoStreamDB
- **Host**: x86_64 (Linux)
- **Workload**: Concurrent ANN Vector Queries (HNSW-TQ8, Top-10)
- **Queries per Concurrency Tier**: 1,000

| Concurrency (Threads) | Throughput (QPS) | Scaling Speedup | p50 Latency | p90 Latency | p99 Latency |
|---|---|---|---|---|---|
| **1** | **465.8** | **1.00x** | 2.08 ms | 2.32 ms | 2.95 ms |
| **2** | **896.3** | **1.92x** | 2.16 ms | 2.47 ms | 2.92 ms |
| **4** | **1460.3** | **3.14x** | 2.61 ms | 3.05 ms | 4.10 ms |
| **8** | **1425.5** | **3.06x** | 5.37 ms | 6.82 ms | 8.45 ms |
| **16** | **1271.2** | **2.73x** | 11.71 ms | 16.49 ms | 21.36 ms |
| **32** | **1240.8** | **2.66x** | 10.03 ms | 26.21 ms | 37.34 ms |
