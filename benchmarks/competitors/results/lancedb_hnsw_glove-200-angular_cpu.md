# Competitor: lancedb_hnsw / vector_ann

| Metric | Value |
|---|---|
| engine | lancedb_hnsw |
| dataset | glove-200-angular |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 200 |
| index | m=16, ef_construction=200, ef_search=200, metric=cosine |
| queries | 500 |
| k | 10 |
| build_s | 1.15 |
| index_mb | 23.26 |
| recall_at_k | 0.1762 |
| qps | 469.2 |
| p50_ms | 1.996 |
| p99_ms | 2.486 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=32, ram_gb=121.4, os=Linux-7.0.0-34-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
