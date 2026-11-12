# Competitor: pgvector / vector_ann

| Metric | Value |
|---|---|
| engine | pgvector |
| index_type | hnsw |
| index_precision | f32 |
| dataset | mnist-784-euclidean |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 784 |
| index | m=16, ef_construction=200, ef_search=200, metric=l2 |
| queries | 500 |
| k | 10 |
| build_s | 12.852 |
| index_mb | 166.35 |
| recall_at_k | 1.0 |
| pure_index_qps | 744.7 |
| pure_p50_ms | 1.328 |
| pure_p99_ms | 2.005 |
| full_row_qps | 640.4 |
| full_p50_ms | 1.524 |
| full_p99_ms | 2.463 |
| qps | 744.7 |
| p50_ms | 1.524 |
| p99_ms | 2.463 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
