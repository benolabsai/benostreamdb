# Competitor: faiss_sq / vector_ann

| Metric | Value |
|---|---|
| engine | faiss_sq |
| index_type | hnsw_sq |
| index_precision | sq |
| dataset | glove-100-angular |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 100 |
| index | m=16, ef_construction=200, ef_search=200, metric=cosine |
| queries | 500 |
| k | 10 |
| build_s | 1.9 |
| index_mb | 4.88 |
| recall_at_k | 0.9704 |
| pure_index_qps | 1593.5 |
| pure_p50_ms | 0.477 |
| pure_p99_ms | 0.742 |
| full_row_qps | 2085.8 |
| full_p50_ms | 0.489 |
| full_p99_ms | 0.656 |
| qps | 1593.5 |
| p50_ms | 0.489 |
| p99_ms | 0.656 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=['NVIDIA GeForce RTX 5070 Ti, 16303 MiB', 'NVIDIA GeForce RTX 5070 Ti, 16303 MiB'] |
