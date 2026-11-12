# Competitor: weaviate / vector_ann

| Metric | Value |
|---|---|
| engine | weaviate |
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
| build_s | 4.227 |
| index_mb | 8.0 |
| recall_at_k | 0.9756 |
| pure_index_qps | 771.2 |
| pure_p50_ms | 1.119 |
| pure_p99_ms | 1.776 |
| full_row_qps | 808.3 |
| full_p50_ms | 1.231 |
| full_p99_ms | 1.562 |
| qps | 771.2 |
| p50_ms | 1.231 |
| p99_ms | 1.562 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
