# Benchmark rollup (2026-11-11T22:48:41-05:00)

Hardware profile: cpus=8 mem=16g — see hardware_profile.txt

| Engine | Device | Dataset | Recall@k | QPS | p50 (ms) | p99 (ms) | Build (s) | Index (MB) |
|---|---|---|---|---|---|---|---|---|
| faiss | cpu | gist-960-euclidean | 0.9916 | 2484.0 | 0.401 | 0.651 | 1.368 | 79.68 |
| faiss_sq | cpu | gist-960-euclidean | 0.9812 | 3392.0 | 0.3 | 0.446 | 0.936 | 22.09 |
| hnswlib | cpu | gist-960-euclidean | 0.9908 | 1911.6 | 0.535 | 0.789 | 3.089 | 79.77 |
| lancedb | cpu | gist-960-euclidean | 0.9976 | 276.2 | 3.389 | 4.458 | 4.31 | 156.75 |
| lancedb_sq | cpu | gist-960-euclidean | 0.964 | 366.0 | 2.603 | 3.124 | 2.509 | 99.14 |
| milvus | cpu | gist-960-euclidean | 0.8864 | 475.0 | 1.871 | 3.16 | 4.06 | 76.8 |
| milvus_sq | cpu | gist-960-euclidean | 0.8864 | 509.9 | 1.74 | 2.625 | 4.444 | 76.8 |
| opensearch | cpu | gist-960-euclidean | 0.9938 | 237.7 | 4.191 | 5.26 | 20.626 | 352.87 |
| pgvector | cpu | gist-960-euclidean | 0.9988 | 332.9 | 3.042 | 3.884 | 24.783 | 247.3 |
| qdrant | cpu | gist-960-euclidean | 0.9972 | 542.4 | 1.808 | 2.569 | 10.516 | 76.8 |
| qdrant_sq | cpu | gist-960-euclidean | 0.954 | 710.2 | 1.376 | 1.891 | 4.434 | 19.2 |
| weaviate | cpu | gist-960-euclidean | 0.9778 | 878.2 | 1.126 | 1.623 | 6.436 | 76.8 |
| weaviate_sq | cpu | gist-960-euclidean | 0.9782 | 894.9 | 1.1 | 1.469 | 6.415 | 19.2 |
