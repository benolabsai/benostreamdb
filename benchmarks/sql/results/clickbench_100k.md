# ClickBench SQL Benchmark Results

- **Rows**: 500,000
- **Dataset File**: `synth_hits_500000_s42.parquet` (6.5 MB)
- **Warm Iterations**: 2

### Warm Query Latency (ms) — Median

| Query | benostreamdb | duckdb | datafusion |
|---| --- | --- | --- |
| **Q0 (count)** | 2.5 ms | 0.6 ms | 0.8 ms |
| **Q1 (filter count)** | 8.2 ms | 3.8 ms | 3.3 ms |
| **Q2 (multi-agg)** | 10.7 ms | 4.1 ms | 4.0 ms |
| **Q3 (avg int)** | 5.5 ms | 5.1 ms | 4.8 ms |
| **Q4 (distinct user)** | 13.4 ms | 34.7 ms | 10.1 ms |
| **Q5 (distinct phrase)** | 18.2 ms | 6.2 ms | 3.9 ms |
| **Q6 (min/max date)** | 6.7 ms | 0.5 ms | 0.6 ms |
| **Q7 (group by agg)** | 8.0 ms | 4.5 ms | 4.4 ms |
| **Q8 (group by distinct)** | 19.5 ms | 42.3 ms | 16.2 ms |
| **Q9 (string filter group)** | 27.1 ms | 52.4 ms | 18.5 ms |

### Ingestion & Total Execution Time

| Engine | Ingestion (s) | Total Warm SQL Time (ms) | Status |
|---|---|---|---|
| benostreamdb | 0.24s | 119.7 ms | ✅ Pass |
| duckdb | Direct Parquet Scan | 154.3 ms | ✅ Pass |
| datafusion | Direct Parquet Scan | 66.4 ms | ✅ Pass |
