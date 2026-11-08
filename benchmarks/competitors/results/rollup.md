# Benchmark rollup (2026-11-08T09:46:42-05:00)

Hardware profile: cpus=8 mem=16g — see hardware_profile.txt

| Engine | Device | Dataset | Recall@k | QPS | p50 (ms) | p99 (ms) | Build (s) | Index (MB) |
|---|---|---|---|---|---|---|---|---|
| benostreamdb | cpu | gist-960-euclidean | 0.9812 | 806.8 | 1.237 | 1.52 | 3.135 | 237.25 |
| benostreamdb_sq | cpu | gist-960-euclidean | 0.9636 | 322.7 | 3.039 | 4.409 | 5.317 | 179.69 |
| benostreamdb_tq8 | cpu | gist-960-euclidean | 0.9528 | 1772.3 | 0.552 | 0.676 | 2.878 | 180.94 |
| faiss | cpu | gist-960-euclidean | 0.9916 | 3096.5 | 0.334 | 0.437 | 1.129 | 79.68 |
| faiss_sq | cpu | gist-960-euclidean | 0.9812 | 3948.3 | 0.26 | 0.382 | 0.84 | 22.09 |
| hnswlib | cpu | gist-960-euclidean | 0.9912 | 2363.8 | 0.44 | 0.578 | 2.785 | 79.77 |
| lancedb | cpu | gist-960-euclidean | 0.999 | 311.8 | 2.987 | 3.473 | 4.214 | 156.75 |
| lancedb_sq | cpu | gist-960-euclidean | 0.9648 | 342.8 | 2.776 | 3.739 | 2.483 | 99.15 |
| milvus | cpu | gist-960-euclidean | 0.8864 | 532.6 | 1.647 | 2.131 | 4.589 | 76.8 |
| milvus_sq | cpu | gist-960-euclidean | 0.8864 | 568.5 | 1.617 | 1.984 | 4.505 | 76.8 |
| opensearch | cpu | gist-960-euclidean | 0.9958 | 238.3 | 4.035 | 5.451 | 17.793 | 352.9 |
| pgvector | cpu | gist-960-euclidean | 0.9988 | 406.3 | 2.487 | 3.211 | 24.565 | 247.28 |
| qdrant | cpu | gist-960-euclidean | 0.9982 | 623.0 | 1.582 | 2.052 | 4.211 | 76.8 |
| qdrant_sq | cpu | gist-960-euclidean | 0.957 | 781.5 | 1.209 | 2.106 | 4.105 | 19.2 |
| weaviate | cpu | gist-960-euclidean | 0.9778 | 1053.2 | 0.942 | 1.307 | 6.07 | 76.8 |
| weaviate_sq | cpu | gist-960-euclidean | 0.9784 | 1054.3 | 0.936 | 1.256 | 6.149 | 19.2 |
