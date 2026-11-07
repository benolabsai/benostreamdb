# Competitor: lancedb / vector_ann

| Metric | Value |
|---|---|
| engine | lancedb |
| index_type | hnsw_flat |
| index_precision | f32 |
| dataset | lastfm-64-dot |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 65 |
| index | m=16, ef_construction=200, ef_search=200, metric=inner_product |
| queries | 500 |
| k | 10 |
| build_s | 0.57 |
| index_mb | 13.6 |
| recall_at_k | 0.999 |
| pure_index_qps | 388.0 |
| pure_p50_ms | 2.471 |
| pure_p99_ms | 3.081 |
| full_row_qps | 382.6 |
| full_p50_ms | 2.579 |
| full_p99_ms | 3.154 |
| qps | 388.0 |
| p50_ms | 2.579 |
| p99_ms | 3.154 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=['NVIDIA GeForce RTX 5070 Ti, 16303 MiB', 'NVIDIA GeForce RTX 5070 Ti, 16303 MiB'] |
