# Competitor: weaviate / vector_ann

| Metric | Value |
|---|---|
| engine | weaviate |
| index_type | flat |
| index_precision | f32 |
| dataset | glove-200-angular |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 200 |
| index | m=16, ef_construction=200, ef_search=200, metric=cosine |
| queries | 500 |
| k | 10 |
| build_s | 4.538 |
| index_mb | 16.0 |
| recall_at_k | 0.9452 |
| pure_index_qps | 666.7 |
| pure_p50_ms | 1.333 |
| pure_p99_ms | 2.054 |
| full_row_qps | 667.0 |
| full_p50_ms | 1.491 |
| full_p99_ms | 1.91 |
| qps | 666.7 |
| p50_ms | 1.491 |
| p99_ms | 1.91 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
