# Competitor: hnswlib / vector_ann

| Metric | Value |
|---|---|
| engine | hnswlib |
| dataset | sift-128-euclidean |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 128 |
| index | m=16, ef_construction=200, ef_search=200, metric=l2 |
| queries | 500 |
| k | 10 |
| build_s | 0.47 |
| index_mb | 13.21 |
| recall_at_k | 1.0 |
| qps | 10761.9 |
| p50_ms | 0.09 |
| p99_ms | 0.202 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-34-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
