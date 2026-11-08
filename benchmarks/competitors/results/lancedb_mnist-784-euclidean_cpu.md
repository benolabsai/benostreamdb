# Competitor: lancedb / vector_ann

| Metric | Value |
|---|---|
| engine | lancedb |
| index_type | hnsw_flat |
| index_precision | f32 |
| dataset | mnist-784-euclidean |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 784 |
| index | m=16, ef_construction=200, ef_search=200, metric=l2 |
| queries | 500 |
| k | 10 |
| build_s | 2.838 |
| index_mb | 128.83 |
| recall_at_k | 0.9996 |
| pure_index_qps | 348.4 |
| pure_p50_ms | 2.656 |
| pure_p99_ms | 3.283 |
| full_row_qps | 340.4 |
| full_p50_ms | 2.849 |
| full_p99_ms | 4.224 |
| qps | 348.4 |
| p50_ms | 2.849 |
| p99_ms | 4.224 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=['NVIDIA GeForce RTX 5070 Ti, 16303 MiB', 'NVIDIA GeForce RTX 5070 Ti, 16303 MiB'] |
