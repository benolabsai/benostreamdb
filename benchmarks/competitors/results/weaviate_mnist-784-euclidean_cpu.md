# Competitor: weaviate / vector_ann

| Metric | Value |
|---|---|
| engine | weaviate |
| index_type | flat |
| index_precision | f32 |
| dataset | mnist-784-euclidean |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 784 |
| index | m=16, ef_construction=200, ef_search=200, metric=l2 |
| queries | 500 |
| k | 10 |
| build_s | 5.144 |
| index_mb | 62.72 |
| recall_at_k | 0.9988 |
| pure_index_qps | 1087.7 |
| pure_p50_ms | 0.903 |
| pure_p99_ms | 1.221 |
| full_row_qps | 736.2 |
| full_p50_ms | 1.337 |
| full_p99_ms | 1.732 |
| qps | 1087.7 |
| p50_ms | 1.337 |
| p99_ms | 1.732 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=['NVIDIA GeForce RTX 5070 Ti, 16303 MiB', 'NVIDIA GeForce RTX 5070 Ti, 16303 MiB'] |
