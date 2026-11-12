# Competitor: pgvector / vector_ann

| Metric | Value |
|---|---|
| engine | pgvector |
| index_type | hnsw |
| index_precision | f32 |
| dataset | sift-128-euclidean |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 128 |
| index | m=16, ef_construction=200, ef_search=200, metric=l2 |
| queries | 500 |
| k | 10 |
| build_s | 4.371 |
| index_mb | 29.2 |
| recall_at_k | 0.9998 |
| pure_index_qps | 1045.2 |
| pure_p50_ms | 0.965 |
| pure_p99_ms | 1.294 |
| full_row_qps | 975.2 |
| full_p50_ms | 1.021 |
| full_p99_ms | 1.456 |
| qps | 1045.2 |
| p50_ms | 1.021 |
| p99_ms | 1.456 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
