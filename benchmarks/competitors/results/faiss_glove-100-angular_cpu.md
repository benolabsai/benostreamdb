# Competitor: faiss / vector_ann

| Metric | Value |
|---|---|
| engine | faiss |
| index_type | flat |
| index_precision | f32 |
| dataset | glove-100-angular |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 100 |
| index | m=16, ef_construction=200, ef_search=200, metric=cosine |
| queries | 500 |
| k | 10 |
| build_s | 0.431 |
| index_mb | 10.88 |
| recall_at_k | 0.9826 |
| pure_index_qps | 3153.0 |
| refine_factor | 1 |
| pure_p50_ms | 0.156 |
| pure_p99_ms | 0.336 |
| full_row_qps | 6086.1 |
| full_p50_ms | 0.159 |
| full_p99_ms | 0.234 |
| qps | 3153.0 |
| p50_ms | 0.159 |
| p99_ms | 0.234 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
