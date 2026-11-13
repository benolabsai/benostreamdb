# Competitor: lancedb_sq / vector_ann

| Metric | Value |
|---|---|
| engine | lancedb_sq |
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
| build_s | 0.644 |
| index_mb | 7.89 |
| recall_at_k | 0.9206 |
| pure_index_qps | 400.9 |
| refine_factor | 1 |
| pure_p50_ms | 2.476 |
| pure_p99_ms | 2.917 |
| full_row_qps | 376.1 |
| full_p50_ms | 2.55 |
| full_p99_ms | 3.192 |
| qps | 376.1 |
| p50_ms | 2.55 |
| p99_ms | 3.192 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
