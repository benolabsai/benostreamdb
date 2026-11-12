# ClickBench SQL Benchmark Results

- **Rows**: 500,000
- **Dataset File**: `synth_hits_500000_s42.parquet` (6.5 MB)
- **Warm Iterations**: 2

### Warm Query Latency (ms) — Median

| Query | benostreamdb | duckdb | datafusion |
|---| --- | --- | --- |
| **Q0 (count)** | 2.1 ms | 0.5 ms | 0.5 ms |
| **Q1 (filter count)** | 7.3 ms | 3.0 ms | 2.6 ms |
| **Q2 (multi-agg)** | 10.3 ms | 3.4 ms | 3.6 ms |
| **Q3 (avg int)** | 6.1 ms | 3.1 ms | 3.5 ms |
| **Q4 (distinct user)** | 11.4 ms | 27.9 ms | 8.4 ms |
| **Q5 (distinct phrase)** | 15.7 ms | 4.8 ms | 4.0 ms |
| **Q6 (min/max date)** | 5.4 ms | 0.4 ms | 0.5 ms |
| **Q7 (group by agg)** | 7.2 ms | 4.2 ms | 4.0 ms |
| **Q8 (group by distinct)** | 19.8 ms | 35.9 ms | 13.5 ms |
| **Q9 (string filter group)** | 24.7 ms | 38.5 ms | 15.1 ms |

### Ingestion & Total Execution Time

| Engine | Ingestion (s) | Total Warm SQL Time (ms) | Status |
|---|---|---|---|
| benostreamdb | 0.19s | 110.1 ms | ✅ Pass |
| duckdb | Direct Parquet Scan | 121.7 ms | ✅ Pass |
| datafusion | Direct Parquet Scan | 55.7 ms | ✅ Pass |
