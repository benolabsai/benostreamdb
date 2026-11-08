# Benchmark rollup (2026-11-07T23:00:08-05:00)

Hardware profile: cpus=8 mem=16g — see hardware_profile.txt

| Engine | Device | Dataset | Recall@k | QPS | p50 (ms) | p99 (ms) | Build (s) | Index (MB) |
|---|---|---|---|---|---|---|---|---|
| benostreamdb | cpu | gist-960-euclidean | 0.9794 | 808.7 | 1.224 | 1.431 | 3.163 | 237.26 |
| benostreamdb_sq | cpu | gist-960-euclidean | 0.7286 | 2022.1 | 0.489 | 0.618 | 2.881 | 179.69 |
| benostreamdb_tq8 | cpu | gist-960-euclidean | 0.954 | 1827.8 | 0.545 | 0.655 | 2.968 | 180.94 |
| faiss | cpu | gist-960-euclidean | 0.9916 | 2717.7 | 0.378 | 0.498 | 1.145 | 79.68 |
| faiss_sq | cpu | gist-960-euclidean | 0.9812 | 3978.0 | 0.257 | 0.412 | 0.823 | 22.09 |
| hnswlib | cpu | gist-960-euclidean | 0.9912 | 2527.3 | 0.409 | 0.555 | 2.724 | 79.77 |
| lancedb | cpu | gist-960-euclidean | 0.9988 | 303.8 | 3.058 | 3.743 | 4.351 | 156.76 |
| lancedb_sq | cpu | gist-960-euclidean | 0.9652 | 392.1 | 2.415 | 3.023 | 2.472 | 99.14 |
| milvus | cpu | gist-960-euclidean | 0.8864 | 529.0 | 1.638 | 2.329 | 4.289 | 76.8 |
| milvus_sq | cpu | gist-960-euclidean | 0.8864 | 578.6 | 1.587 | 2.079 | 4.083 | 76.8 |
| opensearch | cpu | gist-960-euclidean | 0.9954 | 259.6 | 3.83 | 5.018 | 21.729 | 352.81 |
| pgvector | cpu | gist-960-euclidean | 0.9988 | 378.9 | 2.686 | 3.298 | 23.732 | 247.28 |
| qdrant | cpu | gist-960-euclidean | 0.9978 | 633.7 | 1.563 | 1.911 | 4.352 | 76.8 |
| qdrant_sq | cpu | gist-960-euclidean | 0.955 | 849.8 | 1.154 | 1.397 | 4.322 | 19.2 |
| weaviate | cpu | gist-960-euclidean | 0.9772 | 992.5 | 1.002 | 1.437 | 6.064 | 76.8 |
| weaviate_sq | cpu | gist-960-euclidean | 0.978 | 972.5 | 1.006 | 1.52 | 6.064 | 19.2 |
