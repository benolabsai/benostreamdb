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
| build_s | 3.122 |
| index_mb | 79.77 |
| recall_at_k | 0.992 |
| pure_index_qps | 2101.6 |
| pure_p50_ms | 0.475 |
| pure_p99_ms | 0.767 |
| full_row_qps | 1231.9 |
| full_p50_ms | 0.812 |
| full_p99_ms | 1.073 |
| qps | 2101.6 |
| p50_ms | 0.812 |
| p99_ms | 1.073 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
