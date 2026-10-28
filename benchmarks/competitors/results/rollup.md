# Benchmark rollup (2026-10-28T14:58:09-04:00)

Hardware profile: cpus=8 mem=16g — see hardware_profile.txt

| Engine | Device | Dataset | Recall@k | QPS | p50 (ms) | p99 (ms) | Build (s) | Index (MB) |
|---|---|---|---|---|---|---|---|---|
| benostreamdb | cpu | gist-960-euclidean | 0.9654 | 826.5 | 1.183 | 1.703 | 2.919 | 237.47 |
| benostreamdb_tq8 | cpu | gist-960-euclidean | 0.8358 | 1758.0 | 0.558 | 0.797 | 2.703 | 181.15 |
| faiss | cpu | gist-960-euclidean | 0.9916 | 2539.4 | 0.398 | 0.572 | 1.265 | 79.68 |
| hnswlib | cpu | gist-960-euclidean | 0.9912 | 2344.2 | 0.428 | 0.652 | 3.027 | 79.77 |
| lancedb | cpu | gist-960-euclidean | 0.4714 | 414.2 | 2.305 | 2.879 | 6.807 | 79.07 |
| lancedb_hnsw | cpu | gist-960-euclidean | 0.965 | 423.1 | 2.262 | 2.743 | 2.388 | 99.15 |
| milvus | cpu | gist-960-euclidean | 0.8864 | 3.2 | 399.287 | 400.938 | 3.196 | 76.8 |
| opensearch | cpu | gist-960-euclidean | 0.8538 | 304.4 | 3.122 | 4.708 | 18.868 | 588.52 |
| pgvector | cpu | gist-960-euclidean | 0.9988 | 368.1 | 2.7 | 3.623 | 25.043 | 247.28 |
| qdrant | - | - | unavailable: UnexpectedResponse: Unexpected Response: 400 (Bad Request)
Raw response content:
b'{"status":{"error":"JSON payload (39533934 bytes) is larger than allowed (limit: 33554432 bytes)."},"time":0.0}' | - | - | - | - | - |
| weaviate | cpu | gist-960-euclidean | 0.978 | 937.0 | 1.068 | 1.348 | 6.25 | 76.8 |
