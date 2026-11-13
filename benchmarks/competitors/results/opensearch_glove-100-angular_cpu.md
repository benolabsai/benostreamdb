# Competitor: opensearch / vector_ann

| Metric | Value |
|---|---|
| engine | opensearch |
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
| build_s | 8.489 |
| index_mb | 47.06 |
| recall_at_k | 0.994 |
| pure_index_qps | 255.5 |
| refine_factor | 1 |
| pure_p50_ms | 3.901 |
| pure_p99_ms | 5.113 |
| full_row_qps | 232.4 |
| full_p50_ms | 4.274 |
| full_p99_ms | 5.83 |
| qps | 232.4 |
| p50_ms | 4.274 |
| p99_ms | 5.83 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
