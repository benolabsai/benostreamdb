# Competitor: pgvector / vector_ann

| Metric | Value |
|---|---|
| engine | pgvector |
| index_type | hnsw |
| index_precision | f32 |
| dataset | glove-200-angular |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 200 |
| index | m=16, ef_construction=200, ef_search=200, metric=cosine |
| queries | 500 |
| k | 10 |
| build_s | 7.717 |
| index_mb | 42.27 |
| recall_at_k | 0.9652 |
| pure_index_qps | 588.5 |
| pure_p50_ms | 1.554 |
| pure_p99_ms | 2.237 |
| full_row_qps | 619.4 |
| full_p50_ms | 1.628 |
| full_p99_ms | 2.039 |
| qps | 588.5 |
| p50_ms | 1.628 |
| p99_ms | 2.039 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=['NVIDIA GeForce RTX 5070 Ti, 16303 MiB', 'NVIDIA GeForce RTX 5070 Ti, 16303 MiB'] |
