# Competitor: pgvector / vector_ann

| Metric | Value |
|---|---|
| engine | pgvector |
| dataset | glove-200-angular |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 200 |
| index | m=16, ef_construction=200, ef_search=200, metric=cosine |
| queries | 500 |
| k | 10 |
| build_s | 6.412 |
| index_mb | 42.27 |
| recall_at_k | 0.176 |
| qps | 614.6 |
| p50_ms | 1.6 |
| p99_ms | 2.7 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=32, ram_gb=121.4, os=Linux-7.0.0-34-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
