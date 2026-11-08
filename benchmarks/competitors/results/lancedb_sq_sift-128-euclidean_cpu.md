# Competitor: lancedb_sq / vector_ann

| Metric | Value |
|---|---|
| engine | lancedb_sq |
| index_type | hnsw_sq |
| index_precision | sq |
| dataset | sift-128-euclidean |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 128 |
| index | m=16, ef_construction=200, ef_search=200, metric=l2 |
| queries | 500 |
| k | 10 |
| build_s | 0.696 |
| index_mb | 16.19 |
| recall_at_k | 0.9884 |
| pure_index_qps | 378.5 |
| pure_p50_ms | 2.493 |
| pure_p99_ms | 3.892 |
| full_row_qps | 358.2 |
| full_p50_ms | 2.686 |
| full_p99_ms | 4.029 |
| qps | 378.5 |
| p50_ms | 2.686 |
| p99_ms | 4.029 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=['NVIDIA GeForce RTX 5070 Ti, 16303 MiB', 'NVIDIA GeForce RTX 5070 Ti, 16303 MiB'] |
