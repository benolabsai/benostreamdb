# Competitor: lancedb_sq / vector_ann

| Metric | Value |
|---|---|
| engine | lancedb_sq |
| index_type | hnsw_sq |
| index_precision | sq |
| dataset | fashion-mnist-784-euclidean |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 784 |
| index | m=16, ef_construction=200, ef_search=200, metric=l2 |
| queries | 500 |
| k | 10 |
| build_s | 2.05 |
| index_mb | 81.67 |
| recall_at_k | 1.0 |
| pure_index_qps | 255.0 |
| refine_factor | 4 |
| pure_p50_ms | 3.728 |
| pure_p99_ms | 5.127 |
| full_row_qps | 260.5 |
| full_p50_ms | 3.715 |
| full_p99_ms | 5.424 |
| qps | 255.0 |
| p50_ms | 3.715 |
| p99_ms | 5.424 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
