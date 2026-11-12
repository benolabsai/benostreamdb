# Competitor: faiss / vector_ann

| Metric | Value |
|---|---|
| engine | faiss |
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
| build_s | 0.694 |
| index_mb | 23.36 |
| recall_at_k | 0.924 |
| pure_index_qps | 3065.4 |
| refine_factor | 1 |
| pure_p50_ms | 0.319 |
| pure_p99_ms | 0.447 |
| full_row_qps | 3118.4 |
| full_p50_ms | 0.319 |
| full_p99_ms | 0.42 |
| qps | 3065.4 |
| p50_ms | 0.319 |
| p99_ms | 0.42 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
