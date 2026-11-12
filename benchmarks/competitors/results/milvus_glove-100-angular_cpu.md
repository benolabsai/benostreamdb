# Competitor: milvus / vector_ann

| Metric | Value |
|---|---|
| engine | milvus |
| index_type | hnsw |
| index_precision | f32 |
| dataset | glove-100-angular |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 100 |
| index | m=16, ef_construction=200, ef_search=200, metric=cosine |
| queries | 500 |
| k | 10 |
| build_s | 8.544 |
| index_mb | 8.0 |
| recall_at_k | 0.984 |
| pure_index_qps | 504.3 |
| refine_factor | 1 |
| pure_p50_ms | 1.635 |
| pure_p99_ms | 2.615 |
| full_row_qps | 464.3 |
| full_p50_ms | 2.114 |
| full_p99_ms | 2.774 |
| qps | 504.3 |
| p50_ms | 2.114 |
| p99_ms | 2.774 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
