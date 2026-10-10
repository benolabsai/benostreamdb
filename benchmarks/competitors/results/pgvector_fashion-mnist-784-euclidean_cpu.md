# Competitor: pgvector / vector_ann

| Metric | Value |
|---|---|
| engine | pgvector |
| dataset | fashion-mnist-784-euclidean |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 784 |
| index | m=16, ef_construction=200, ef_search=200, metric=l2 |
| queries | 500 |
| k | 10 |
| build_s | 15.508 |
| index_mb | 166.36 |
| recall_at_k | 0.5042 |
| qps | 23.1 |
| p50_ms | 42.826 |
| p99_ms | 52.429 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=32, ram_gb=121.4, os=Linux-7.0.0-34-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
