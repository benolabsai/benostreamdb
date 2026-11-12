# Competitor: faiss_sq / vector_ann

| Metric | Value |
|---|---|
| engine | faiss_sq |
| index_type | hnsw_sq |
| index_precision | sq |
| dataset | sift-128-euclidean |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 128 |
| index | m=16, ef_construction=200, ef_search=200, metric=l2 |
| queries | 500 |
| k | 10 |
| build_s | 0.337 |
| index_mb | 5.44 |
| recall_at_k | 0.9916 |
| pure_index_qps | 8846.1 |
| refine_factor | 1 |
| pure_p50_ms | 0.112 |
| pure_p99_ms | 0.175 |
| full_row_qps | 8850.6 |
| full_p50_ms | 0.115 |
| full_p99_ms | 0.14 |
| qps | 8846.1 |
| p50_ms | 0.115 |
| p99_ms | 0.14 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=['NVIDIA GeForce RTX 5070 Ti, 16303 MiB', 'NVIDIA GeForce RTX 5070 Ti, 16303 MiB'] |
