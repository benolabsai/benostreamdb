# Competitor: weaviate_sq / vector_ann

| Metric | Value |
|---|---|
| engine | weaviate_sq |
| index_type | hnsw_sq |
| index_precision | sq |
| dataset | lastfm-64-dot |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 65 |
| index | m=16, ef_construction=200, ef_search=200, metric=inner_product |
| queries | 500 |
| k | 10 |
| build_s | 3.883 |
| index_mb | 1.3 |
| recall_at_k | 0.9906 |
| pure_index_qps | 1064.1 |
| refine_factor | 1 |
| pure_p50_ms | 0.771 |
| pure_p99_ms | 1.125 |
| full_row_qps | 1129.5 |
| full_p50_ms | 0.862 |
| full_p99_ms | 1.255 |
| qps | 1064.1 |
| p50_ms | 0.862 |
| p99_ms | 1.255 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=['NVIDIA GeForce RTX 5070 Ti, 16303 MiB', 'NVIDIA GeForce RTX 5070 Ti, 16303 MiB'] |
