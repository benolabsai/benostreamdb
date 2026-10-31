# Competitor: qdrant / vector_ann

| Metric | Value |
|---|---|
| engine | qdrant |
| dataset | fashion-mnist-784-euclidean |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 784 |
| index | m=16, ef_construction=200, ef_search=200, metric=l2 |
| queries | 500 |
| k | 10 |
| build_s | 9.771 |
| index_mb | 62.72 |
| recall_at_k | 1.0 |
| pure_index_qps | 651.1 |
| pure_p50_ms | 1.497 |
| pure_p99_ms | 1.903 |
| full_row_qps | 358.8 |
| full_p50_ms | 2.752 |
| full_p99_ms | 3.364 |
| qps | 651.1 |
| p50_ms | 2.752 |
| p99_ms | 3.364 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-38-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
