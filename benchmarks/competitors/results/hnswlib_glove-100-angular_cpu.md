# Competitor: hnswlib / vector_ann

| Metric | Value |
|---|---|
| engine | hnswlib |
| dataset | glove-100-angular |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 100 |
| index | m=16, ef_construction=200, ef_search=200, metric=cosine |
| queries | 500 |
| k | 10 |
| build_s | 0.552 |
| index_mb | 10.97 |
| recall_at_k | 0.9756 |
| pure_index_qps | 3654.5 |
| pure_p50_ms | 0.127 |
| pure_p99_ms | 0.268 |
| full_row_qps | 5957.9 |
| full_p50_ms | 0.171 |
| full_p99_ms | 0.197 |
| qps | 3654.5 |
| p50_ms | 0.171 |
| p99_ms | 0.197 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-38-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
