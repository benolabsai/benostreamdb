# Competitor: milvus_sq / vector_ann

| Metric | Value |
|---|---|
| engine | milvus_sq |
| index_type | ivf_sq8 |
| index_precision | sq |
| dataset | sift-128-euclidean |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 128 |
| index | m=16, ef_construction=200, ef_search=200, metric=l2 |
| queries | 500 |
| k | 10 |
| build_s | 8.338 |
| index_mb | 10.24 |
| recall_at_k | 0.9324 |
| pure_index_qps | 656.8 |
| refine_factor | 1 |
| pure_p50_ms | 1.343 |
| pure_p99_ms | 2.127 |
| full_row_qps | 580.4 |
| full_p50_ms | 1.649 |
| full_p99_ms | 2.674 |
| qps | 656.8 |
| p50_ms | 1.649 |
| p99_ms | 2.674 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=['NVIDIA GeForce RTX 5070 Ti, 16303 MiB', 'NVIDIA GeForce RTX 5070 Ti, 16303 MiB'] |
