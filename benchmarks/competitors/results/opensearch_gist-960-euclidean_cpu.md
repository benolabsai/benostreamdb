# Competitor: opensearch / vector_ann

| Metric | Value |
|---|---|
| engine | opensearch |
| index_type | hnsw |
| index_precision | f32 |
| dataset | gist-960-euclidean |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 960 |
| index | m=16, ef_construction=200, ef_search=200, metric=l2 |
| queries | 500 |
| k | 10 |
| build_s | 17.459 |
| index_mb | 352.95 |
| recall_at_k | 0.9932 |
| pure_index_qps | 235.6 |
| pure_p50_ms | 3.965 |
| pure_p99_ms | 6.209 |
| full_row_qps | 148.6 |
| full_p50_ms | 6.704 |
| full_p99_ms | 7.877 |
| qps | 235.6 |
| p50_ms | 6.704 |
| p99_ms | 7.877 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
