# Competitor: hnswlib / vector_ann

| Metric | Value |
|---|---|
| engine | hnswlib |
| index_type | hnsw |
| index_precision | f32 |
| dataset | gist-960-euclidean |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 960 |
| index | m=16, ef_construction=200, ef_search=200, metric=l2 |
| queries | 500 |
| k | 10 |
| build_s | 3.107 |
| index_mb | 79.77 |
| recall_at_k | 0.9916 |
| pure_index_qps | 2002.7 |
| refine_factor | 1 |
| pure_p50_ms | 0.51 |
| pure_p99_ms | 0.77 |
| full_row_qps | 1190.5 |
| full_p50_ms | 0.85 |
| full_p99_ms | 1.116 |
| qps | 2002.7 |
| p50_ms | 0.85 |
| p99_ms | 1.116 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
