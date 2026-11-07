# Competitor: milvus_sq / vector_ann

| Metric | Value |
|---|---|
| engine | milvus_sq |
| index_type | ivf_sq8 |
| index_precision | sq |
| dataset | glove-100-angular |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 100 |
| index | m=16, ef_construction=200, ef_search=200, metric=cosine |
| queries | 500 |
| k | 10 |
| build_s | 3.291 |
| index_mb | 8.0 |
| recall_at_k | 0.8474 |
| pure_index_qps | 593.5 |
| pure_p50_ms | 1.303 |
| pure_p99_ms | 2.119 |
| full_row_qps | 500.8 |
| full_p50_ms | 1.92 |
| full_p99_ms | 2.987 |
| qps | 593.5 |
| p50_ms | 1.92 |
| p99_ms | 2.987 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=['NVIDIA GeForce RTX 5070 Ti, 16303 MiB', 'NVIDIA GeForce RTX 5070 Ti, 16303 MiB'] |
