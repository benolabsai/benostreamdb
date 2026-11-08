# Competitor: weaviate / vector_ann

| Metric | Value |
|---|---|
| engine | weaviate |
| index_type | flat |
| index_precision | f32 |
| dataset | gist-960-euclidean |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 960 |
| index | m=16, ef_construction=200, ef_search=200, metric=l2 |
| queries | 500 |
| k | 10 |
| build_s | 6.07 |
| index_mb | 76.8 |
| recall_at_k | 0.9778 |
| pure_index_qps | 1053.2 |
| pure_p50_ms | 0.942 |
| pure_p99_ms | 1.307 |
| full_row_qps | 621.6 |
| full_p50_ms | 1.561 |
| full_p99_ms | 2.176 |
| qps | 1053.2 |
| p50_ms | 1.561 |
| p99_ms | 2.176 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=['NVIDIA GeForce RTX 5070 Ti, 16303 MiB', 'NVIDIA GeForce RTX 5070 Ti, 16303 MiB'] |
