# Competitor: milvus / vector_ann

| Metric | Value |
|---|---|
| engine | milvus |
| dataset | glove-100-angular |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 100 |
| index | m=16, ef_construction=200, ef_search=200, metric=cosine |
| queries | 500 |
| k | 10 |
| build_s | 1.599 |
| index_mb | 0.0 |
| recall_at_k | 0.4408 |
| qps | 3.5 |
| p50_ms | 200.495 |
| p99_ms | 401.04 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-34-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
