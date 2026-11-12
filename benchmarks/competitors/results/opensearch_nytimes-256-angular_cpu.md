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
| build_s | 15.473 |
| index_mb | 217.08 |
| recall_at_k | 0.9794 |
| pure_index_qps | 94.1 |
| pure_p50_ms | 10.557 |
| pure_p99_ms | 13.206 |
| full_row_qps | 87.5 |
| full_p50_ms | 11.26 |
| full_p99_ms | 15.792 |
| qps | 94.1 |
| p50_ms | 11.26 |
| p99_ms | 15.792 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
