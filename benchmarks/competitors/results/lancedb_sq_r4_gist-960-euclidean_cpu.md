# Competitor: lancedb_sq / vector_ann

| Metric | Value |
|---|---|
| engine | lancedb_sq |
| index_type | hnsw_sq |
| index_precision | sq |
| dataset | gist-960-euclidean |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 960 |
| index | m=16, ef_construction=200, ef_search=200, metric=l2 |
| queries | 500 |
| k | 10 |
| build_s | 2.652 |
| index_mb | 99.14 |
| recall_at_k | 0.9986 |
| pure_index_qps | 219.7 |
| refine_factor | 4 |
| pure_p50_ms | 4.323 |
| pure_p99_ms | 6.064 |
| full_row_qps | 215.3 |
| full_p50_ms | 4.461 |
| full_p99_ms | 6.979 |
| qps | 219.7 |
| p50_ms | 4.461 |
| p99_ms | 6.979 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
