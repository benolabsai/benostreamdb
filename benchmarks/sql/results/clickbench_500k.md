# ClickBench SQL Benchmark Results

- **Rows**: 500,000
- **Dataset File**: `synth_hits_500000_s42.parquet` (6.5 MB)
- **Warm Iterations**: 2

### Warm Query Latency (ms) — Median

| Query | benostreamdb | duckdb | datafusion |
|---| --- | --- | --- |
| **Q0 (count)** | 2.0 ms | 0.4 ms | 0.5 ms |
| **Q1 (filter count)** | 6.1 ms | 2.9 ms | 2.5 ms |
| **Q2 (multi-agg)** | 9.6 ms | 3.3 ms | 3.6 ms |
| **Q3 (avg int)** | 5.6 ms | 4.1 ms | 3.4 ms |
| **Q4 (distinct user)** | 9.9 ms | 27.7 ms | 8.1 ms |
| **Q5 (distinct phrase)** | 14.5 ms | 4.9 ms | 2.9 ms |
| **Q6 (min/max date)** | 4.8 ms | 0.4 ms | 0.4 ms |
| **Q7 (group by agg)** | 5.8 ms | 3.5 ms | 3.7 ms |
| **Q8 (group by distinct)** | 15.4 ms | 30.8 ms | 11.9 ms |
| **Q9 (string filter group)** | 22.8 ms | 37.8 ms | 12.6 ms |

### Ingestion & Total Execution Time

| Engine | Ingestion (s) | Total Warm SQL Time (ms) | Status |
|---|---|---|---|
| benostreamdb | 0.18s | 96.4 ms | ✅ Pass |
| duckdb | Direct Parquet Scan | 115.7 ms | ✅ Pass |
| datafusion | Direct Parquet Scan | 49.7 ms | ✅ Pass |
