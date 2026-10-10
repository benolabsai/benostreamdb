# Competitor: opensearch / vector_ann

| Metric | Value |
|---|---|
| engine | opensearch |
| dataset | gist-960-euclidean |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 960 |
| index | m=16, ef_construction=200, ef_search=200, metric=l2 |
| queries | 500 |
| k | 10 |
| build_s | 26.805 |
| index_mb | 0.02 |
| recall_at_k | 0.8324 |
| qps | 232.8 |
| p50_ms | 3.427 |
| p99_ms | 22.752 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=32, ram_gb=121.4, os=Linux-7.0.0-34-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
