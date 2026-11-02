# Competitor: qdrant_sq / vector_ann

| Metric | Value |
|---|---|
| engine | qdrant_sq |
| index_type | hnsw_sq |
| index_precision | sq |
| dataset | sift-128-euclidean |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 128 |
| index | m=16, ef_construction=200, ef_search=200, metric=l2 |
| queries | 500 |
| k | 10 |
| build_s | 2.07 |
| index_mb | 2.56 |
| recall_at_k | 1.0 |
| pure_index_qps | 935.0 |
| pure_p50_ms | 1.053 |
| pure_p99_ms | 1.424 |
| full_row_qps | 573.8 |
| full_p50_ms | 1.301 |
| full_p99_ms | 2.54 |
| qps | 935.0 |
| p50_ms | 1.301 |
| p99_ms | 2.54 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-38-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
