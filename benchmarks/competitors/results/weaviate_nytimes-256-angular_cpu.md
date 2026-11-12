# Competitor: weaviate / vector_ann

| Metric | Value |
|---|---|
| engine | weaviate |
| index_type | flat |
| index_precision | f32 |
| dataset | nytimes-256-angular |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 256 |
| index | m=16, ef_construction=200, ef_search=200, metric=cosine |
| queries | 500 |
| k | 10 |
| build_s | 4.855 |
| index_mb | 20.48 |
| recall_at_k | 0.9272 |
| pure_index_qps | 690.8 |
| refine_factor | 1 |
| pure_p50_ms | 1.441 |
| pure_p99_ms | 1.925 |
| full_row_qps | 591.1 |
| full_p50_ms | 1.678 |
| full_p99_ms | 2.183 |
| qps | 690.8 |
| p50_ms | 1.678 |
| p99_ms | 2.183 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
