# Benchmark rollup (2026-11-12T07:21:11-05:00)

Hardware profile: cpus=8 mem=16g — see hardware_profile.txt

| Engine | Device | Dataset | Recall@k | QPS | p50 (ms) | p99 (ms) | Build (s) | Index (MB) |
|---|---|---|---|---|---|---|---|---|
| benostreamdb | cpu | gist-960-euclidean | 0.9782 | 710.6 | 1.388 | 2.022 | 3.057 | 237.26 |
| benostreamdb_sq | cpu | gist-960-euclidean | 0.968 | 491.2 | 2.008 | 2.669 | 6.679 | 179.67 |
| benostreamdb_tq8 | cpu | gist-960-euclidean | 0.9544 | 518.6 | 1.908 | 2.47 | 3.044 | 180.92 |
| faiss | cpu | gist-960-euclidean | 0.9916 | 2556.5 | 0.392 | 0.612 | 1.303 | 79.68 |
| faiss_sq | cpu | gist-960-euclidean | 0.9812 | 3543.5 | 0.288 | 0.421 | 0.889 | 22.09 |
| hnswlib | cpu | gist-960-euclidean | 0.9916 | 2111.9 | 0.475 | 0.744 | 3.041 | 79.77 |
| lancedb | cpu | gist-960-euclidean | 0.9982 | 270.6 | 3.416 | 4.317 | 4.281 | 156.75 |
| lancedb_sq | cpu | gist-960-euclidean | 0.965 | 359.6 | 2.673 | 3.068 | 2.553 | 99.15 |
| milvus | cpu | gist-960-euclidean | 0.995 | 464.0 | 1.986 | 2.747 | 11.434 | 76.8 |
| milvus_sq | cpu | gist-960-euclidean | 0.8568 | 556.7 | 1.65 | 2.164 | 10.42 | 76.8 |
| opensearch | cpu | gist-960-euclidean | 0.9928 | 216.5 | 4.337 | 6.569 | 17.431 | 352.93 |
| pgvector | cpu | gist-960-euclidean | 0.9988 | 336.0 | 3.034 | 3.846 | 24.337 | 247.29 |
| qdrant | cpu | gist-960-euclidean | 0.9988 | 538.7 | 1.822 | 2.507 | 4.341 | 76.8 |
| qdrant_sq | cpu | gist-960-euclidean | 0.9538 | 674.7 | 1.447 | 2.049 | 4.32 | 19.2 |
| weaviate | cpu | gist-960-euclidean | 0.9914 | 683.6 | 1.467 | 1.924 | 6.452 | 76.8 |
| weaviate_sq | cpu | gist-960-euclidean | 0.9902 | 668.4 | 1.495 | 2.051 | 6.457 | 19.2 |
