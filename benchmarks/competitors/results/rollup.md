# Benchmark rollup (2026-11-07T14:36:38-05:00)

Hardware profile: cpus=8 mem=16g — see hardware_profile.txt

| Engine | Device | Dataset | Recall@k | QPS | p50 (ms) | p99 (ms) | Build (s) | Index (MB) |
|---|---|---|---|---|---|---|---|---|
| benostreamdb | cpu | lastfm-64-dot | 0.8998 | 2231.8 | 0.444 | 0.587 | 4.628 | 22.07 |
| benostreamdb_sq | cpu | lastfm-64-dot | 0.895 | 2283.2 | 0.431 | 0.663 | 4.181 | 18.18 |
| benostreamdb_tq8 | cpu | lastfm-64-dot | 0.8886 | 1807.0 | 0.545 | 0.763 | 2.731 | 19.43 |
| faiss | cpu | lastfm-64-dot | 0.9964 | 4528.0 | 0.071 | 0.175 | 0.214 | 8.08 |
| faiss_sq | cpu | lastfm-64-dot | 0.9826 | 8844.1 | 0.107 | 0.168 | 0.459 | 4.18 |
| hnswlib | cpu | lastfm-64-dot | 0.9914 | 4753.5 | 0.058 | 0.112 | 0.276 | 8.17 |
| lancedb | cpu | lastfm-64-dot | 0.9982 | 385.1 | 2.34 | 2.954 | 0.582 | 13.6 |
| lancedb_sq | cpu | lastfm-64-dot | 0.9208 | 368.0 | 2.402 | 3.941 | 0.575 | 7.9 |
| milvus | cpu | lastfm-64-dot | 0.976 | 643.5 | 1.273 | 2.046 | 2.924 | 5.2 |
| milvus_sq | cpu | lastfm-64-dot | 0.976 | 614.5 | 1.292 | 1.844 | 3.191 | 5.2 |
| opensearch | cpu | lastfm-64-dot | 0.9704 | 799.4 | 1.24 | 1.592 | 3.645 | 32.26 |
| pgvector | cpu | lastfm-64-dot | 0.996 | 1041.4 | 0.799 | 1.065 | 6.238 | 18.51 |
| qdrant | cpu | lastfm-64-dot | 1.0 | 894.5 | 0.93 | 1.366 | 0.65 | 5.2 |
| qdrant_sq | cpu | lastfm-64-dot | 1.0 | 893.2 | 0.94 | 1.361 | 0.628 | 1.3 |
| weaviate | cpu | lastfm-64-dot | 0.9844 | 1424.6 | 0.679 | 1.238 | 4.217 | 5.2 |
| weaviate_sq | cpu | lastfm-64-dot | 0.9812 | 1372.2 | 0.701 | 1.078 | 3.879 | 1.3 |
