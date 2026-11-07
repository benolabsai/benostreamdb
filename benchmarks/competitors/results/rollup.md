# Benchmark rollup (2026-11-07T15:17:10-05:00)

Hardware profile: cpus=8 mem=16g — see hardware_profile.txt

| Engine | Device | Dataset | Recall@k | QPS | p50 (ms) | p99 (ms) | Build (s) | Index (MB) |
|---|---|---|---|---|---|---|---|---|
| benostreamdb | cpu | gist-960-euclidean | 0.9798 | 810.9 | 1.23 | 1.477 | 3.282 | 237.25 |
| benostreamdb_sq | cpu | gist-960-euclidean | 0.7286 | 1894.9 | 0.519 | 0.653 | 2.89 | 179.69 |
| benostreamdb_tq8 | cpu | gist-960-euclidean | 0.9518 | 1915.0 | 0.51 | 0.75 | 2.814 | 180.94 |
| faiss | cpu | gist-960-euclidean | 0.9916 | 2744.3 | 0.374 | 0.498 | 1.137 | 79.68 |
| faiss_sq | cpu | gist-960-euclidean | 0.9812 | 3281.6 | 0.311 | 0.447 | 0.843 | 22.09 |
| hnswlib | cpu | gist-960-euclidean | 0.9916 | 2430.1 | 0.425 | 0.572 | 2.808 | 79.77 |
| lancedb | cpu | gist-960-euclidean | 0.9982 | 296.0 | 3.149 | 3.614 | 4.076 | 156.75 |
| lancedb_sq | cpu | gist-960-euclidean | 0.964 | 387.4 | 2.476 | 3.052 | 2.405 | 99.15 |
| milvus | cpu | gist-960-euclidean | 0.8864 | 546.2 | 1.687 | 2.152 | 4.022 | 76.8 |
| milvus_sq | cpu | gist-960-euclidean | 0.8864 | 534.7 | 1.709 | 2.229 | 4.333 | 76.8 |
| opensearch | cpu | gist-960-euclidean | 0.9942 | 260.1 | 3.844 | 4.582 | 18.988 | 352.91 |
| pgvector | cpu | gist-960-euclidean | 0.9988 | 392.0 | 2.583 | 3.231 | 23.863 | 247.28 |
| qdrant | cpu | gist-960-euclidean | 0.9986 | 590.7 | 1.679 | 2.092 | 5.689 | 76.8 |
| qdrant_sq | cpu | gist-960-euclidean | 0.9526 | 694.1 | 1.408 | 1.851 | 9.633 | 19.2 |
| weaviate | cpu | gist-960-euclidean | 0.9792 | 991.9 | 1.002 | 1.349 | 6.168 | 76.8 |
| weaviate_sq | cpu | gist-960-euclidean | 0.9766 | 940.1 | 1.052 | 1.405 | 6.288 | 19.2 |
