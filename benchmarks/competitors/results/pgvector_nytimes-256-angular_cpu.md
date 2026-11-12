# Competitor: pgvector / vector_ann

| Metric | Value |
|---|---|
| engine | pgvector |
| index_type | hnsw |
| index_precision | f32 |
| dataset | nytimes-256-angular |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 256 |
| index | m=16, ef_construction=200, ef_search=200, metric=cosine |
| queries | 500 |
| k | 10 |
| build_s | 9.912 |
| index_mb | 50.92 |
| recall_at_k | 0.9476 |
| pure_index_qps | 472.3 |
| refine_factor | 1 |
| pure_p50_ms | 2.099 |
| pure_p99_ms | 2.706 |
| full_row_qps | 442.7 |
| full_p50_ms | 2.235 |
| full_p99_ms | 3.008 |
| qps | 472.3 |
| p50_ms | 2.235 |
| p99_ms | 3.008 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
