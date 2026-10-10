# Competitor: weaviate / vector_ann

| Metric | Value |
|---|---|
| engine | weaviate |
| dataset | gist-960-euclidean |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 960 |
| index | m=16, ef_construction=200, ef_search=200, metric=l2 |
| queries | 500 |
| k | 10 |
| build_s | 8.205 |
| index_mb | 76.8 |
| recall_at_k | 0.978 |
| qps | 779.1 |
| p50_ms | 1.167 |
| p99_ms | 2.355 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-34-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
