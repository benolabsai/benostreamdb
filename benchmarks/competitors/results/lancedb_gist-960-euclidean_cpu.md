# Competitor: lancedb / vector_ann

| Metric | Value |
|---|---|
| engine | lancedb |
| index_type | hnsw_flat |
| index_precision | f32 |
| dataset | gist-960-euclidean |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 960 |
| index | m=16, ef_construction=200, ef_search=200, metric=l2 |
| queries | 500 |
| k | 10 |
| build_s | 4.236 |
| index_mb | 156.75 |
| recall_at_k | 0.9986 |
| pure_index_qps | 286.3 |
| refine_factor | 1 |
| pure_p50_ms | 3.437 |
| pure_p99_ms | 4.536 |
| full_row_qps | 259.7 |
| full_p50_ms | 3.707 |
| full_p99_ms | 5.088 |
| qps | 259.7 |
| p50_ms | 3.707 |
| p99_ms | 5.088 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
