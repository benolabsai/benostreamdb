# ClickBench SQL Benchmark Results

- **Rows**: 500,000
- **Dataset File**: `synth_hits_500000_s42.parquet` (6.5 MB)
- **Warm Iterations**: 2

### Warm Query Latency (ms) — Median

| Query | benostreamdb | duckdb | datafusion |
|---| --- | --- | --- |
| **Q0 (count)** | 2.2 ms | 0.6 ms | 0.8 ms |
| **Q1 (filter count)** | 8.1 ms | 3.2 ms | 3.0 ms |
| **Q2 (multi-agg)** | 12.5 ms | 4.3 ms | 3.7 ms |
| **Q3 (avg int)** | 6.1 ms | 4.3 ms | 4.2 ms |
| **Q4 (distinct user)** | 14.8 ms | 30.3 ms | 10.0 ms |
| **Q5 (distinct phrase)** | 17.1 ms | 5.2 ms | 4.0 ms |
| **Q6 (min/max date)** | 6.3 ms | 0.4 ms | 0.5 ms |
| **Q7 (group by agg)** | 9.0 ms | 3.6 ms | 4.5 ms |
| **Q8 (group by distinct)** | 26.1 ms | 35.6 ms | 15.9 ms |
| **Q9 (string filter group)** | 32.5 ms | 43.4 ms | 16.9 ms |

### Ingestion & Total Execution Time

| Engine | Ingestion (s) | Total Warm SQL Time (ms) | Status |
|---|---|---|---|
| benostreamdb | 0.24s | 134.7 ms | ✅ Pass |
| duckdb | Direct Parquet Scan | 130.9 ms | ✅ Pass |
| datafusion | Direct Parquet Scan | 63.5 ms | ✅ Pass |
