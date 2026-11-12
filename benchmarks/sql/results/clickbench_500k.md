# ClickBench SQL Benchmark Results

- **Rows**: 500,000
- **Dataset File**: `synth_hits_500000_s42.parquet` (6.5 MB)
- **Warm Iterations**: 2

### Warm Query Latency (ms) — Median

| Query | benostreamdb | duckdb | datafusion |
|---| --- | --- | --- |
| **Q0 (count)** | 2.3 ms | 0.4 ms | 0.5 ms |
| **Q1 (filter count)** | 7.5 ms | 3.0 ms | 2.7 ms |
| **Q2 (multi-agg)** | 9.6 ms | 3.7 ms | 3.4 ms |
| **Q3 (avg int)** | 6.2 ms | 4.2 ms | 3.4 ms |
| **Q4 (distinct user)** | 11.3 ms | 27.7 ms | 9.7 ms |
| **Q5 (distinct phrase)** | 14.6 ms | 5.1 ms | 4.4 ms |
| **Q6 (min/max date)** | 4.9 ms | 0.5 ms | 0.5 ms |
| **Q7 (group by agg)** | 5.8 ms | 3.4 ms | 3.8 ms |
| **Q8 (group by distinct)** | 17.1 ms | 31.0 ms | 13.7 ms |
| **Q9 (string filter group)** | 21.2 ms | 36.6 ms | 15.2 ms |

### Ingestion & Total Execution Time

| Engine | Ingestion (s) | Total Warm SQL Time (ms) | Status |
|---|---|---|---|
| benostreamdb | 0.18s | 100.4 ms | ✅ Pass |
| duckdb | Direct Parquet Scan | 115.6 ms | ✅ Pass |
| datafusion | Direct Parquet Scan | 57.2 ms | ✅ Pass |
