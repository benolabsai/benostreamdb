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
| build_s | 2.495 |
| index_mb | 99.15 |
| recall_at_k | 0.9974 |
| pure_index_qps | 277.4 |
| refine_factor | 2 |
| pure_p50_ms | 3.465 |
| pure_p99_ms | 4.263 |
| full_row_qps | 265.4 |
| full_p50_ms | 3.665 |
| full_p99_ms | 5.381 |
| qps | 277.4 |
| p50_ms | 3.665 |
| p99_ms | 5.381 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
