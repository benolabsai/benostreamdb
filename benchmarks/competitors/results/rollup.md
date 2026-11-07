# Benchmark rollup (2026-11-07T10:59:05-05:00)

Hardware profile: cpus=8 mem=16g — see hardware_profile.txt

| Engine | Device | Dataset | Recall@k | QPS | p50 (ms) | p99 (ms) | Build (s) | Index (MB) |
|---|---|---|---|---|---|---|---|---|
| benostreamdb | cpu | gist-960-euclidean | 0.9766 | 805.2 | 1.224 | 1.558 | 3.224 | 237.26 |
| benostreamdb_sq | cpu | gist-960-euclidean | 0.727 | 2046.7 | 0.482 | 0.607 | 2.866 | 179.68 |
| benostreamdb_tq8 | cpu | gist-960-euclidean | 0.9526 | 2013.8 | 0.491 | 0.606 | 2.83 | 180.94 |
| faiss | cpu | gist-960-euclidean | 0.9916 | 2986.7 | 0.348 | 0.458 | 1.124 | 79.68 |
| faiss_sq | cpu | gist-960-euclidean | 0.9812 | 3825.5 | 0.267 | 0.389 | 0.86 | 22.09 |
| hnswlib | cpu | gist-960-euclidean | 0.991 | 2036.0 | 0.505 | 0.787 | 2.825 | 79.77 |
| lancedb | cpu | gist-960-euclidean | 0.9988 | 285.6 | 3.208 | 4.19 | 4.318 | 156.75 |
| lancedb_sq | cpu | gist-960-euclidean | 0.9652 | 378.8 | 2.537 | 3.232 | 2.536 | 99.15 |
| milvus | cpu | gist-960-euclidean | 0.8864 | 507.3 | 1.736 | 2.366 | 4.353 | 76.8 |
| milvus_sq | cpu | gist-960-euclidean | 0.8864 | 550.9 | 1.641 | 2.22 | 4.318 | 76.8 |
| opensearch | cpu | gist-960-euclidean | 0.9928 | 232.2 | 4.195 | 5.783 | 18.414 | 582.14 |
| pgvector | cpu | gist-960-euclidean | 0.999 | 391.6 | 2.581 | 3.36 | 23.046 | 247.28 |
| qdrant | cpu | gist-960-euclidean | 0.9978 | 566.5 | 1.736 | 2.31 | 4.042 | 76.8 |
| qdrant_sq | cpu | gist-960-euclidean | 0.9564 | 753.5 | 1.275 | 2.153 | 3.968 | 19.2 |
| weaviate | cpu | gist-960-euclidean | 0.9754 | 987.6 | 1.003 | 1.54 | 6.069 | 76.8 |
| weaviate_sq | cpu | gist-960-euclidean | 0.9766 | 997.3 | 0.992 | 1.612 | 6.103 | 19.2 |
