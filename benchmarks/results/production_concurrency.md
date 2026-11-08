# Multi-Client Concurrency Scaling Benchmark

- **Engine**: BenoStreamDB
- **Host**: x86_64 (Linux)
- **Workload**: Concurrent ANN Vector Queries (HNSW-TQ8, Top-10)
- **Queries per Concurrency Tier**: 1,000

| Concurrency (Threads) | Throughput (QPS) | Scaling Speedup | p50 Latency | p90 Latency | p99 Latency |
|---|---|---|---|---|---|
| **1** | **141.8** | **1.00x** | 6.97 ms | 7.39 ms | 8.10 ms |
| **2** | **233.5** | **1.65x** | 8.50 ms | 9.21 ms | 10.22 ms |
| **4** | **338.1** | **2.38x** | 11.76 ms | 13.25 ms | 14.64 ms |
| **8** | **402.2** | **2.84x** | 19.33 ms | 25.95 ms | 35.13 ms |
| **16** | **442.2** | **3.12x** | 34.31 ms | 49.21 ms | 59.88 ms |
| **32** | **463.1** | **3.27x** | 54.39 ms | 97.73 ms | 152.16 ms |
