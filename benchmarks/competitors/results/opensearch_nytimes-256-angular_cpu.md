# Competitor: opensearch / vector_ann

| Metric | Value |
|---|---|
| engine | opensearch |
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
| build_s | 15.74 |
| index_mb | 214.62 |
| recall_at_k | 0.978 |
| pure_index_qps | 93.0 |
| refine_factor | 1 |
| pure_p50_ms | 10.484 |
| pure_p99_ms | 14.859 |
| full_row_qps | 86.3 |
| full_p50_ms | 11.484 |
| full_p99_ms | 14.858 |
| qps | 93.0 |
| p50_ms | 11.484 |
| p99_ms | 14.858 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
