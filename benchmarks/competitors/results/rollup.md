# Benchmark rollup (2026-11-12T12:21:48-05:00)

Hardware profile: cpus=8 mem=16g — see hardware_profile.txt

| Engine | Device | Dataset | Recall@k | QPS | p50 (ms) | p99 (ms) | Build (s) | Index (MB) |
|---|---|---|---|---|---|---|---|---|
| benostreamdb | cpu | gist-960-euclidean | 0.9776 | 622.5 | 1.568 | 2.609 | 4.133 | 237.25 |
| benostreamdb_sq | cpu | gist-960-euclidean | 0.968 | 489.6 | 2.02 | 2.698 | 6.195 | 179.67 |
| benostreamdb_tq8 | cpu | gist-960-euclidean | 0.9558 | 468.7 | 2.098 | 2.913 | 3.366 | 180.93 |
| faiss | cpu | gist-960-euclidean | 0.9916 | 2390.7 | 0.411 | 0.671 | 1.457 | 79.68 |
| faiss_sq | cpu | gist-960-euclidean | 0.9812 | 3307.2 | 0.305 | 0.429 | 0.907 | 22.09 |
| hnswlib | cpu | gist-960-euclidean | 0.9914 | 1835.0 | 0.555 | 0.912 | 3.193 | 79.77 |
| lancedb | cpu | gist-960-euclidean | 0.9988 | 257.2 | 3.599 | 4.672 | 4.479 | 156.75 |
| lancedb_sq | cpu | gist-960-euclidean | 0.9654 | 334.6 | 2.827 | 3.563 | 2.479 | 99.15 |
| milvus | cpu | gist-960-euclidean | 0.9946 | 388.8 | 2.38 | 3.847 | 12.647 | 76.8 |
| milvus_sq | cpu | gist-960-euclidean | 0.8568 | 534.6 | 1.69 | 2.53 | 10.852 | 76.8 |
| opensearch | cpu | gist-960-euclidean | 0.9972 | 195.8 | 5.016 | 7.523 | 16.873 | 352.91 |
| pgvector | cpu | gist-960-euclidean | 0.9988 | 347.3 | 2.922 | 4.067 | 26.396 | 247.28 |
| qdrant | cpu | gist-960-euclidean | 0.997 | 487.8 | 2.038 | 2.741 | 4.339 | 76.8 |
| qdrant_sq | cpu | gist-960-euclidean | 0.9598 | 637.1 | 1.51 | 2.405 | 4.27 | 19.2 |
| weaviate | cpu | gist-960-euclidean | 0.9906 | 631.9 | 1.533 | 2.418 | 6.619 | 76.8 |
| weaviate_sq | cpu | gist-960-euclidean | 0.991 | 602.9 | 1.642 | 2.404 | 6.776 | 19.2 |
