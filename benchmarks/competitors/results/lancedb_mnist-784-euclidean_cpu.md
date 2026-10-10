# Competitor: lancedb / vector_ann

| Metric | Value |
|---|---|
| engine | lancedb |
| dataset | mnist-784-euclidean |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 784 |
| index | m=16, ef_construction=200, ef_search=200, metric=l2 |
| queries | 500 |
| k | 10 |
| build_s | 5.511 |
| index_mb | 64.59 |
| recall_at_k | 0.8262 |
| qps | 466.1 |
| p50_ms | 2.028 |
| p99_ms | 2.6 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-34-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
