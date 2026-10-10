# Competitor: opensearch / vector_ann

| Metric | Value |
|---|---|
| engine | opensearch |
| dataset | sift-128-euclidean |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 128 |
| index | m=16, ef_construction=200, ef_search=200, metric=l2 |
| queries | 500 |
| k | 10 |
| build_s | 3.932 |
| index_mb | 1.13 |
| recall_at_k | 0.9506 |
| qps | 1112.7 |
| p50_ms | 0.882 |
| p99_ms | 1.337 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-34-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
