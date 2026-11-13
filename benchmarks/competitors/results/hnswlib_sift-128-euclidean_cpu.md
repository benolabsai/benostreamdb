# Competitor: hnswlib / vector_ann

| Metric | Value |
|---|---|
| engine | hnswlib |
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
| build_s | 0.398 |
| index_mb | 13.21 |
| recall_at_k | 1.0 |
| pure_index_qps | 8998.1 |
| refine_factor | 1 |
| pure_p50_ms | 0.101 |
| pure_p99_ms | 0.258 |
| full_row_qps | 5951.6 |
| full_p50_ms | 0.158 |
| full_p99_ms | 0.324 |
| qps | 5951.6 |
| p50_ms | 0.158 |
| p99_ms | 0.324 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
