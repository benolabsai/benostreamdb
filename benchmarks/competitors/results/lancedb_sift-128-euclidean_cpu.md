# Competitor: lancedb / vector_ann

| Metric | Value |
|---|---|
| engine | lancedb |
| index_type | hnsw_flat |
| index_precision | f32 |
| dataset | sift-128-euclidean |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 128 |
| index | m=16, ef_construction=200, ef_search=200, metric=l2 |
| queries | 500 |
| k | 10 |
| build_s | 0.737 |
| index_mb | 23.85 |
| recall_at_k | 1.0 |
| pure_index_qps | 393.5 |
| refine_factor | 1 |
| pure_p50_ms | 2.396 |
| pure_p99_ms | 3.073 |
| full_row_qps | 396.1 |
| full_p50_ms | 2.494 |
| full_p99_ms | 3.049 |
| qps | 393.5 |
| p50_ms | 2.494 |
| p99_ms | 3.049 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
