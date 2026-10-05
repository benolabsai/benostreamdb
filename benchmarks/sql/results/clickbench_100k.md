# ClickBench SQL Benchmark Results

- **Rows**: 500,000
- **Dataset File**: `synth_hits_500000_s42.parquet` (6.5 MB)
- **Warm Iterations**: 2

### Warm Query Latency (ms) — Median

| Query | benostreamdb | duckdb | datafusion |
|---| --- | --- | --- |
| **Q0 (count)** | 2.1 ms | 0.3 ms | 0.6 ms |
| **Q1 (filter count)** | 6.7 ms | 2.8 ms | 2.6 ms |
| **Q2 (multi-agg)** | 9.8 ms | 3.4 ms | 3.2 ms |
| **Q3 (avg int)** | 4.4 ms | 3.8 ms | 3.4 ms |
| **Q4 (distinct user)** | 11.5 ms | 27.6 ms | 10.1 ms |
| **Q5 (distinct phrase)** | 14.9 ms | 5.0 ms | 4.3 ms |
| **Q6 (min/max date)** | 5.4 ms | 0.3 ms | 0.5 ms |
| **Q7 (group by agg)** | 6.8 ms | 3.0 ms | 3.6 ms |
| **Q8 (group by distinct)** | 18.1 ms | 33.6 ms | 14.5 ms |
| **Q9 (string filter group)** | 24.9 ms | 57.2 ms | 16.7 ms |

### Ingestion & Total Execution Time

| Engine | Ingestion (s) | Total Warm SQL Time (ms) | Status |
|---|---|---|---|
| benostreamdb | 0.21s | 104.5 ms | ✅ Pass |
| duckdb | Direct Parquet Scan | 137.2 ms | ✅ Pass |
| datafusion | Direct Parquet Scan | 59.4 ms | ✅ Pass |
