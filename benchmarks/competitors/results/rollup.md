# Benchmark rollup (2026-11-12T16:03:04-05:00)

Hardware profile: cpus=8 mem=16g — see hardware_profile.txt

| Engine | Device | Dataset | Recall@k | QPS | p50 (ms) | p99 (ms) | Build (s) | Index (MB) |
|---|---|---|---|---|---|---|---|---|
| benostreamdb | cpu | gist-960-euclidean | 0.9776 | 703.6 | 1.395 | 1.986 | 3.625 | 237.25 |
| benostreamdb_sq | cpu | gist-960-euclidean | 0.968 | 515.1 | 1.922 | 2.5 | 6.33 | 179.67 |
| benostreamdb_tq8 | cpu | gist-960-euclidean | 0.9558 | 472.8 | 2.089 | 2.891 | 3.547 | 180.93 |
| faiss | cpu | gist-960-euclidean | 0.9916 | 2461.2 | 0.405 | 0.654 | 1.432 | 79.68 |
| faiss_sq | cpu | gist-960-euclidean | 0.9812 | 3261.6 | 0.313 | 0.431 | 0.871 | 22.09 |
| hnswlib | cpu | gist-960-euclidean | 0.991 | 2148.3 | 0.475 | 0.729 | 3.215 | 79.77 |
| lancedb | cpu | gist-960-euclidean | 0.9978 | 267.3 | 3.447 | 4.387 | 4.6 | 156.75 |
| lancedb_sq | cpu | gist-960-euclidean | 0.9648 | 346.0 | 2.755 | 3.885 | 2.462 | 99.14 |
| milvus | cpu | gist-960-euclidean | 0.9944 | 481.0 | 1.897 | 2.464 | 11.865 | 76.8 |
| milvus_sq | cpu | gist-960-euclidean | 0.8568 | 553.7 | 1.651 | 2.266 | 10.546 | 76.8 |
| opensearch | cpu | gist-960-euclidean | 0.9932 | 199.7 | 4.985 | 6.331 | 17.116 | 352.94 |
| pgvector | cpu | gist-960-euclidean | 0.9988 | 335.6 | 3.03 | 3.846 | 24.693 | 247.3 |
| qdrant | cpu | gist-960-euclidean | 0.9976 | 555.2 | 1.756 | 2.525 | 4.092 | 76.8 |
| qdrant_sq | cpu | gist-960-euclidean | 0.9568 | 627.6 | 1.558 | 2.146 | 4.169 | 19.2 |
| weaviate | cpu | gist-960-euclidean | 0.9906 | 675.3 | 1.486 | 2.121 | 6.592 | 76.8 |
| weaviate_sq | cpu | gist-960-euclidean | 0.9914 | 678.8 | 1.443 | 2.131 | 6.512 | 19.2 |
