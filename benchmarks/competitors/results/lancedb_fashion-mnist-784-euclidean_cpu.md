# Competitor: lancedb / vector_ann

| Metric | Value |
|---|---|
| engine | lancedb |
| dataset | fashion-mnist-784-euclidean |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 784 |
| index | m=16, ef_construction=200, ef_search=200, metric=l2 |
| queries | 500 |
| k | 10 |
| build_s | 5.921 |
| index_mb | 64.59 |
| recall_at_k | 0.7536 |
| qps | 376.0 |
| p50_ms | 2.52 |
| p99_ms | 3.244 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=32, ram_gb=121.4, os=Linux-7.0.0-34-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
