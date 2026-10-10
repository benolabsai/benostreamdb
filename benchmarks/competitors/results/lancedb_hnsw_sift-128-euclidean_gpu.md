# Competitor: lancedb_hnsw / vector_ann

| Metric | Value |
|---|---|
| engine | lancedb_hnsw |
| dataset | sift-128-euclidean |
| workload | vector_ann |
| device | gpu |
| n | 20000 |
| dim | 128 |
| index | m=16, ef_construction=200, ef_search=200, metric=l2 |
| queries | 500 |
| k | 10 |
| build_s | 0.666 |
| index_mb | 16.19 |
| recall_at_k | 0.9284 |
| qps | 545.3 |
| p50_ms | 1.713 |
| p99_ms | 2.267 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=32, ram_gb=121.4, os=Linux-7.0.0-34-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=['NVIDIA GeForce RTX 5070 Ti, 16303 MiB'] |
