# Competitor: opensearch / vector_ann

| Metric | Value |
|---|---|
| engine | opensearch |
| index_type | hnsw |
| index_precision | f32 |
| dataset | glove-200-angular |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 200 |
| index | m=16, ef_construction=200, ef_search=200, metric=cosine |
| queries | 500 |
| k | 10 |
| build_s | 12.365 |
| index_mb | 96.06 |
| recall_at_k | 0.9852 |
| pure_index_qps | 159.4 |
| pure_p50_ms | 5.938 |
| pure_p99_ms | 9.451 |
| full_row_qps | 132.0 |
| full_p50_ms | 7.411 |
| full_p99_ms | 10.783 |
| qps | 159.4 |
| p50_ms | 7.411 |
| p99_ms | 10.783 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=['NVIDIA GeForce RTX 5070 Ti, 16303 MiB', 'NVIDIA GeForce RTX 5070 Ti, 16303 MiB'] |
