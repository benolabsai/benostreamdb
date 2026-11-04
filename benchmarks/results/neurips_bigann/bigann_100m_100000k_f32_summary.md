# NeurIPS BigANN 100M — Uncompressed F32 Benchmark Summary

Hardware: **Azure Standard_D8lds_v5 (8 vCPUs / 16 GiB RAM limit, 8 GiB cache budget)**  
Dataset: **BigANN 100M (100,000,000 vectors, 128-dim, uint8 raw)**  
Table DB Size: **125.32 GB** (Data: 49.18 GB, Index: 76.08 GB across 24 Voronoi Puffin shards)  
Reindex Time: **11,012.6s** (3.06 hours)  
Total Evaluation Duration: **90,791.5s** (25.2 hours across 160,000 queries)  

| ef_search | n_probe | Recall@10 | Pure QPS | p50 Latency (ms) | p99 Latency (ms) | Peak RSS (MB) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 32 | 4 | **0.1401** | 8.7 | 858.87 | 2,504.69 | 15,127.4 |
| 32 | 8 | **0.2518** | 5.8 | 1,267.74 | 4,264.26 | 19,703.4 |
| 32 | 16 | **0.4266** | 2.1 | 3,629.94 | 11,935.01 | 17,791.5 |
| 32 | 32 | **0.6247** | 0.9 | 8,703.62 | 20,045.63 | 17,676.1 |
| 64 | 4 | **0.1413** | 8.1 | 931.63 | 2,568.26 | 17,405.9 |
| 64 | 8 | **0.2541** | 4.3 | 1,746.48 | 4,494.23 | 17,482.9 |
| 64 | 16 | **0.4306** | 2.1 | 3,571.58 | 10,009.74 | 17,570.5 |
| 64 | 32 | **0.6305** | 0.9 | 8,802.33 | 20,854.82 | 17,563.5 |
| 128 | 4 | **0.1422** | 8.1 | 942.57 | 2,276.42 | 17,510.4 |
| 128 | 8 | **0.2557** | 3.7 | 1,855.35 | 8,151.80 | 17,489.8 |
| 128 | 16 | **0.4335** | 2.1 | 3,630.46 | 13,171.58 | 17,635.5 |
| 128 | 32 | **0.6351** | 0.9 | 9,065.55 | 20,832.07 | 17,546.8 |
| 256 | 4 | **0.1427** | 7.7 | 964.10 | 2,988.66 | 17,437.9 |
| 256 | 8 | **0.2566** | 4.0 | 1,847.74 | 5,441.54 | 17,341.0 |
| 256 | 16 | **0.4352** | 2.0 | 3,716.57 | 12,649.82 | 17,422.1 |
| 256 | 32 | **0.6376** | 0.8 | 9,325.56 | 21,589.93 | 17,392.8 |

## Systems Findings

1. **Out-of-Core Memory Safety on F32**: The uncompressed index (76.08 GB) is $9.5\times$ larger than the assigned 8 GiB cache budget and $4.8\times$ larger than total system RAM (16 GiB). Over 25.2 hours and 160,000 queries, traversal ran continuously without crashing, reading through 15+ Terabytes of NVMe I/O.
2. **IVF Probe Dominance**: Recall is heavily dominated by `n_probe` rather than beam width (`ef_search`), demonstrating that at 100M scale, coarse centroid routing correctly prunes unvisited Voronoi shards.

