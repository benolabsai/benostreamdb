# Competitor: lancedb / vector_ann

| Metric | Value |
|---|---|
| engine | lancedb |
| index_type | hnsw_flat |
| index_precision | f32 |
| dataset | glove-100-angular |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 100 |
| index | m=16, ef_construction=200, ef_search=200, metric=cosine |
| queries | 500 |
| k | 10 |
| build_s | 0.842 |
| index_mb | 19.27 |
| recall_at_k | 0.976 |
| pure_index_qps | 338.8 |
| pure_p50_ms | 2.631 |
| pure_p99_ms | 3.407 |
| full_row_qps | 359.6 |
| full_p50_ms | 2.748 |
| full_p99_ms | 3.453 |
| qps | 338.8 |
| p50_ms | 2.748 |
| p99_ms | 3.453 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
