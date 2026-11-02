# NeurIPS BigANN 10M — Complete Quantizer Ablation (BenoStreamDB)

Hardware Envelope: **Azure Standard_D8lds_v5 (8 vCPUs / 16 GiB RAM)**
Protocol: **Standardized Pre-Warmed Cache (500 probe warmup, posix_fadvise cache purge)**

## 1. High-Level Mode Comparison

| Mode | Quantizer Type | Bits | Index Size | Reindex Time | Recall@10 Range | Pure QPS Range | p50 Latency | Peak RSS |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **F32** | Uncompressed IEEE-754 Float32 | 32 | 7.61 GB | 1382.9s | 0.7395 – **0.9694** | 26.3 – **119.3** | 66.0 – 295.9 ms | 9.90 GiB |
| **BQ8** | BenoQuant (H·D Rotated INT8) | 8 | 4.03 GB | 943.2s | 0.7323 – **0.9455** | 60.5 – **163.6** | 48.3 – 130.8 ms | 4.99 GiB |
| **SQ8** | Native Scalar Quantization (INT8) | 8 | 4.03 GB | 875.1s | 0.7313 – **0.9449** | 60.2 – **147.0** | 53.6 – 130.4 ms | 5.18 GiB |
| **BQ4** | BenoQuant (H·D + f16 scalar) | 4 | 3.45 GB | 1298.4s | 0.4984 – **0.5799** | 57.9 – **144.4** | 50.0 – 136.6 ms | 4.59 GiB |
| **PQ** | Product Quantization (PQ-16) | 8 | 2.99 GB | 837.3s | 0.3353 – **0.3732** | 47.2 – **132.2** | 59.7 – 168.0 ms | 4.51 GiB |

## 2. Global Pareto-Optimal Frontier (Across All Modes)

| Mode | ef_search | n_probe | Recall@10 | Pure QPS | p50 Latency | p99 Latency | Peak RSS | Frontier Significance |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **BQ8** |  64 |  4 | **0.7451** | **163.6** |  48.28 ms |  67.18 ms | 4.91 GiB | Throughput Peak (Overall Fastest) |
| **SQ8** | 128 |  4 | **0.7538** | **146.0** |  53.88 ms |  79.20 ms | 5.07 GiB | High QPS Scalar Point |
| **SQ8** | 256 |  4 | **0.7586** | **145.0** |  54.34 ms |  79.18 ms | 5.15 GiB | High QPS Deep Beam |
| **BQ8** |  64 |  8 | **0.8480** | **133.0** |  59.06 ms |  88.06 ms | 4.94 GiB | 85% Recall Range Leader |
| **SQ8** | 256 |  8 | **0.8631** | **120.9** |  65.34 ms |  96.22 ms | 5.06 GiB | 86% Recall Scalar Point |
| **BQ8** | 256 |  8 | **0.8640** | **107.7** |  73.51 ms | 105.37 ms | 4.92 GiB | 86.4% Recall Point |
| **BQ8** | 128 | 16 | **0.9160** | ** 97.0** |  80.33 ms | 122.49 ms | 4.94 GiB | 91.6% Recall (>90 QPS) |
| **SQ8** | 256 | 16 | **0.9210** | ** 89.2** |  88.72 ms | 126.74 ms | 5.10 GiB | 92.1% Recall Scalar Point |
| **BQ8** | 256 | 16 | **0.9214** | ** 82.3** |  96.41 ms | 135.19 ms | 4.90 GiB | 92.14% Recall Point |
| **BQ8** | 128 | 32 | **0.9399** | ** 65.1** | 121.52 ms | 179.98 ms | 4.93 GiB | 94.0% Recall Anchor |
| **BQ8** | 256 | 32 | **0.9455** | ** 61.4** | 129.75 ms | 190.52 ms | 4.97 GiB | Quantized Recall Ceiling (94.55%) |
| **F32** |  64 | 32 | **0.9502** | ** 28.5** | 277.37 ms | 458.08 ms | 9.75 GiB | Continuous Threshold (>=95% Recall) |
| **F32** | 128 | 32 | **0.9631** | ** 28.4** | 278.03 ms | 456.17 ms | 9.73 GiB | High-Fidelity Retrieval (96.3%) |
| **F32** | 256 | 32 | **0.9694** | ** 26.3** | 295.95 ms | 520.30 ms | 9.76 GiB | Global Recall Ceiling (96.94%) |
