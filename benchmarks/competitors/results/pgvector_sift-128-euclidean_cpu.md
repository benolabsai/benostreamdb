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
| build_s | 4.204 |
| index_mb | 29.22 |
| recall_at_k | 0.998 |
| pure_index_qps | 1011.9 |
| refine_factor | 1 |
| pure_p50_ms | 0.991 |
| pure_p99_ms | 1.48 |
| full_row_qps | 1009.1 |
| full_p50_ms | 0.986 |
| full_p99_ms | 1.443 |
| qps | 1011.9 |
| p50_ms | 0.986 |
| p99_ms | 1.443 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
