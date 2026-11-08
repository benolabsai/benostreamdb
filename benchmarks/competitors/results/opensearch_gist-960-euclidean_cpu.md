# Competitor: opensearch / vector_ann

| Metric | Value |
|---|---|
| engine | opensearch |
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
| build_s | 21.729 |
| index_mb | 352.81 |
| recall_at_k | 0.9954 |
| pure_index_qps | 259.6 |
| pure_p50_ms | 3.83 |
| pure_p99_ms | 5.018 |
| full_row_qps | 150.9 |
| full_p50_ms | 6.642 |
| full_p99_ms | 7.304 |
| qps | 259.6 |
| p50_ms | 6.642 |
| p99_ms | 7.304 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=['NVIDIA GeForce RTX 5070 Ti, 16303 MiB', 'NVIDIA GeForce RTX 5070 Ti, 16303 MiB'] |
