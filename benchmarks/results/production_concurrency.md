# Multi-Client Concurrency Scaling Benchmark

- **Engine**: BenoStreamDB
- **Host**: x86_64 (Linux)
- **Workload**: Concurrent ANN Vector Queries (HNSW-TQ8, Top-10)
- **Queries per Concurrency Tier**: 1,000

| Concurrency (Threads) | Throughput (QPS) | Scaling Speedup | p50 Latency | p90 Latency | p99 Latency |
|---|---|---|---|---|---|
| **1** | **509.0** | **1.00x** | 1.86 ms | 2.35 ms | 3.22 ms |
| **2** | **956.9** | **1.88x** | 1.98 ms | 2.48 ms | 3.32 ms |
| **4** | **1225.3** | **2.41x** | 3.10 ms | 4.08 ms | 5.36 ms |
| **8** | **1138.0** | **2.24x** | 6.63 ms | 9.11 ms | 13.32 ms |
| **16** | **1128.9** | **2.22x** | 13.07 ms | 18.41 ms | 23.95 ms |
| **32** | **1137.2** | **2.23x** | 11.70 ms | 32.38 ms | 46.50 ms |
