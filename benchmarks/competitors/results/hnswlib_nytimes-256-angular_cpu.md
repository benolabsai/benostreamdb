# Competitor: hnswlib / vector_ann

| Metric | Value |
|---|---|
| engine | hnswlib |
| index_type | hnsw |
| index_precision | f32 |
| dataset | nytimes-256-angular |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 256 |
| index | m=16, ef_construction=200, ef_search=200, metric=cosine |
| queries | 500 |
| k | 10 |
| build_s | 1.135 |
| index_mb | 23.45 |
| recall_at_k | 0.9272 |
| pure_index_qps | 2510.0 |
| pure_p50_ms | 0.24 |
| pure_p99_ms | 0.462 |
| full_row_qps | 2646.3 |
| full_p50_ms | 0.366 |
| full_p99_ms | 0.633 |
| qps | 2510.0 |
| p50_ms | 0.366 |
| p99_ms | 0.633 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=['NVIDIA GeForce RTX 5070 Ti, 16303 MiB', 'NVIDIA GeForce RTX 5070 Ti, 16303 MiB'] |
