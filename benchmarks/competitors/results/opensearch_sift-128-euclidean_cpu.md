# Competitor: opensearch / vector_ann

| Metric | Value |
|---|---|
| engine | opensearch |
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
| build_s | 3.51 |
| index_mb | 28.39 |
| recall_at_k | 0.9998 |
| pure_index_qps | 645.7 |
| refine_factor | 1 |
| pure_p50_ms | 1.548 |
| pure_p99_ms | 1.982 |
| full_row_qps | 572.8 |
| full_p50_ms | 1.724 |
| full_p99_ms | 2.419 |
| qps | 645.7 |
| p50_ms | 1.724 |
| p99_ms | 2.419 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
