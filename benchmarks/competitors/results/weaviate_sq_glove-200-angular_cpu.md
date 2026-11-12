# Competitor: weaviate_sq / vector_ann

| Metric | Value |
|---|---|
| engine | weaviate_sq |
| index_type | hnsw_sq |
| index_precision | sq |
| dataset | glove-200-angular |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 200 |
| index | m=16, ef_construction=200, ef_search=200, metric=cosine |
| queries | 500 |
| k | 10 |
| build_s | 4.601 |
| index_mb | 4.0 |
| recall_at_k | 0.947 |
| pure_index_qps | 661.1 |
| refine_factor | 1 |
| pure_p50_ms | 1.341 |
| pure_p99_ms | 2.244 |
| full_row_qps | 619.8 |
| full_p50_ms | 1.595 |
| full_p99_ms | 2.087 |
| qps | 661.1 |
| p50_ms | 1.595 |
| p99_ms | 2.087 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=['NVIDIA GeForce RTX 5070 Ti, 16303 MiB', 'NVIDIA GeForce RTX 5070 Ti, 16303 MiB'] |
