# Competitor: milvus_sq / vector_ann

| Metric | Value |
|---|---|
| engine | milvus_sq |
| index_type | hnsw_sq |
| index_precision | sq |
| dataset | gist-960-euclidean |
| workload | vector_ann |
| device | cpu |
| n | 20000 |
| dim | 960 |
| index | m=16, ef_construction=200, ef_search=200, metric=l2 |
| queries | 500 |
| k | 10 |
| build_s | 11.043 |
| index_mb | 76.8 |
| recall_at_k | 0.9838 |
| pure_index_qps | 479.4 |
| refine_factor | 1 |
| pure_p50_ms | 2.052 |
| pure_p99_ms | 2.827 |
| full_row_qps | 374.8 |
| full_p50_ms | 2.618 |
| full_p99_ms | 3.612 |
| qps | 374.8 |
| p50_ms | 2.618 |
| p99_ms | 3.612 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
