# Competitor: pgvector / vector_ann

| Metric | Value |
|---|---|
| engine | pgvector |
| dataset | glove-100-angular |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 100 |
| index | m=16, ef_construction=200, ef_search=200, metric=cosine |
| queries | 500 |
| k | 10 |
| build_s | 4.338 |
| index_mb | 24.85 |
| recall_at_k | 0.462 |
| qps | 748.9 |
| p50_ms | 1.326 |
| p99_ms | 1.937 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=32, ram_gb=121.4, os=Linux-7.0.0-34-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
