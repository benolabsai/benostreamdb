# Competitor: qdrant / vector_ann

| Metric | Value |
|---|---|
| engine | qdrant |
| index_type | flat |
| index_precision | f32 |
| dataset | gist-960-euclidean |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 960 |
| index | m=16, ef_construction=200, ef_search=200, metric=l2 |
| queries | 500 |
| k | 10 |
| build_s | 4.1 |
| index_mb | 76.8 |
| recall_at_k | 0.9974 |
| pure_index_qps | 537.9 |
| refine_factor | 1 |
| pure_p50_ms | 1.826 |
| pure_p99_ms | 2.513 |
| full_row_qps | 363.4 |
| full_p50_ms | 2.751 |
| full_p99_ms | 3.454 |
| qps | 363.4 |
| p50_ms | 2.751 |
| p99_ms | 3.454 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
