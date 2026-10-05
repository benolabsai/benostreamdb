# Competitor: pgvector / vector_ann

| Metric | Value |
|---|---|
| engine | pgvector |
| dataset | sift-128-euclidean |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 128 |
| index | m=16, ef_construction=200, ef_search=200, metric=l2 |
| queries | 500 |
| k | 10 |
| build_s | 2.743 |
| index_mb | 29.22 |
| recall_at_k | 0.9926 |
| qps | 324.9 |
| p50_ms | 2.984 |
| p99_ms | 3.86 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=32, ram_gb=121.4, os=Linux-7.0.0-34-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=['NVIDIA GeForce RTX 5070 Ti, 16303 MiB'] |
