# Competitor: opensearch / vector_ann

| Metric | Value |
|---|---|
| engine | opensearch |
| index_type | hnsw |
| index_precision | f32 |
| dataset | lastfm-64-dot |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 65 |
| index | m=16, ef_construction=200, ef_search=200, metric=inner_product |
| queries | 500 |
| k | 10 |
| build_s | 3.621 |
| index_mb | 32.26 |
| recall_at_k | 0.9692 |
| pure_index_qps | 727.0 |
| pure_p50_ms | 1.215 |
| pure_p99_ms | 1.548 |
| full_row_qps | 680.7 |
| full_p50_ms | 1.474 |
| full_p99_ms | 1.737 |
| qps | 727.0 |
| p50_ms | 1.474 |
| p99_ms | 1.737 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=['NVIDIA GeForce RTX 5070 Ti, 16303 MiB', 'NVIDIA GeForce RTX 5070 Ti, 16303 MiB'] |
