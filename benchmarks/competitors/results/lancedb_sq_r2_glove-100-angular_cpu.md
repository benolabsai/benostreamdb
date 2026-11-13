# Competitor: lancedb_sq / vector_ann

| Metric | Value |
|---|---|
| engine | lancedb_sq |
| index_type | hnsw_sq |
| index_precision | sq |
| dataset | glove-100-angular |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 100 |
| index | m=16, ef_construction=200, ef_search=200, metric=cosine |
| queries | 500 |
| k | 10 |
| build_s | 0.774 |
| index_mb | 13.28 |
| recall_at_k | 0.9804 |
| pure_index_qps | 298.3 |
| refine_factor | 2 |
| pure_p50_ms | 3.345 |
| pure_p99_ms | 4.037 |
| full_row_qps | 289.9 |
| full_p50_ms | 3.352 |
| full_p99_ms | 4.034 |
| qps | 289.9 |
| p50_ms | 3.352 |
| p99_ms | 4.034 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
