# Competitor: lancedb_sq / vector_ann

| Metric | Value |
|---|---|
| engine | lancedb_sq |
| index_type | hnsw_sq |
| index_precision | sq |
| dataset | mnist-784-euclidean |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 784 |
| index | m=16, ef_construction=200, ef_search=200, metric=l2 |
| queries | 500 |
| k | 10 |
| build_s | 2.092 |
| index_mb | 81.79 |
| recall_at_k | 0.9996 |
| pure_index_qps | 286.0 |
| refine_factor | 2 |
| pure_p50_ms | 3.433 |
| pure_p99_ms | 4.509 |
| full_row_qps | 266.9 |
| full_p50_ms | 3.585 |
| full_p99_ms | 4.765 |
| qps | 266.9 |
| p50_ms | 3.585 |
| p99_ms | 4.765 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
