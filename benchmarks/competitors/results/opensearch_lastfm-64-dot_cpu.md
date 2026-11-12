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
| build_s | 3.799 |
| index_mb | 32.25 |
| recall_at_k | 0.9732 |
| pure_index_qps | 669.8 |
| refine_factor | 1 |
| pure_p50_ms | 1.327 |
| pure_p99_ms | 1.669 |
| full_row_qps | 607.6 |
| full_p50_ms | 1.627 |
| full_p99_ms | 2.022 |
| qps | 669.8 |
| p50_ms | 1.627 |
| p99_ms | 2.022 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
