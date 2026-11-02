# Competitor: weaviate_sq / vector_ann

| Metric | Value |
|---|---|
| engine | weaviate_sq |
| index_type | hnsw_sq |
| index_precision | sq |
| dataset | glove-100-angular |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 100 |
| index | m=16, ef_construction=200, ef_search=200, metric=cosine |
| queries | 500 |
| k | 10 |
| build_s | 4.237 |
| index_mb | 2.0 |
| recall_at_k | 0.9332 |
| pure_index_qps | 884.8 |
| pure_p50_ms | 0.95 |
| pure_p99_ms | 1.722 |
| full_row_qps | 958.2 |
| full_p50_ms | 1.011 |
| full_p99_ms | 1.669 |
| qps | 884.8 |
| p50_ms | 1.011 |
| p99_ms | 1.669 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-38-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
