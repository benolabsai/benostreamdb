# Competitor: opensearch / vector_ann

| Metric | Value |
|---|---|
| engine | opensearch |
| dataset | nytimes-256-angular |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 256 |
| index | m=16, ef_construction=200, ef_search=200, metric=cosine |
| queries | 500 |
| k | 10 |
| build_s | 15.471 |
| index_mb | 216.01 |
| recall_at_k | 0.079 |
| qps | 271.1 |
| p50_ms | 3.673 |
| p99_ms | 6.128 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-34-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=['NVIDIA GeForce RTX 5070 Ti, 16303 MiB'] |
