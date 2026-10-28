# Multi-Client Concurrency Scaling Benchmark

- **Engine**: BenoStreamDB
- **Host**: x86_64 (Linux)
- **Workload**: Concurrent ANN Vector Queries (HNSW-TQ8, Top-10)
- **Queries per Concurrency Tier**: 1,000

| Concurrency (Threads) | Throughput (QPS) | Scaling Speedup | p50 Latency | p90 Latency | p99 Latency |
|---|---|---|---|---|---|
| **1** | **900.8** | **1.00x** | 1.06 ms | 1.22 ms | 1.51 ms |
| **2** | **1521.1** | **1.69x** | 1.26 ms | 1.50 ms | 2.07 ms |
| **4** | **1576.0** | **1.75x** | 2.46 ms | 3.09 ms | 3.83 ms |
| **8** | **1449.8** | **1.61x** | 5.28 ms | 7.14 ms | 9.31 ms |
| **16** | **1342.3** | **1.49x** | 11.25 ms | 15.88 ms | 20.77 ms |
| **32** | **1309.8** | **1.45x** | 7.33 ms | 21.46 ms | 34.59 ms |
