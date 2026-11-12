# ClickBench SQL Benchmark Results

- **Rows**: 500,000
- **Dataset File**: `synth_hits_500000_s42.parquet` (6.5 MB)
- **Warm Iterations**: 2

### Warm Query Latency (ms) — Median

| Query | benostreamdb | duckdb | datafusion |
|---| --- | --- | --- |
| **Q0 (count)** | 2.5 ms | 0.4 ms | 0.4 ms |
| **Q1 (filter count)** | 7.0 ms | 3.1 ms | 2.5 ms |
| **Q2 (multi-agg)** | 10.5 ms | 3.3 ms | 3.2 ms |
| **Q3 (avg int)** | 5.2 ms | 4.3 ms | 3.2 ms |
| **Q4 (distinct user)** | 10.3 ms | 25.8 ms | 9.8 ms |
| **Q5 (distinct phrase)** | 14.6 ms | 4.8 ms | 3.5 ms |
| **Q6 (min/max date)** | 5.5 ms | 0.4 ms | 0.4 ms |
| **Q7 (group by agg)** | 7.3 ms | 3.5 ms | 3.5 ms |
| **Q8 (group by distinct)** | 17.9 ms | 32.9 ms | 12.6 ms |
| **Q9 (string filter group)** | 24.4 ms | 40.5 ms | 14.3 ms |

### Ingestion & Total Execution Time

| Engine | Ingestion (s) | Total Warm SQL Time (ms) | Status |
|---|---|---|---|
| benostreamdb | 0.19s | 105.2 ms | ✅ Pass |
| duckdb | Direct Parquet Scan | 119.1 ms | ✅ Pass |
| datafusion | Direct Parquet Scan | 53.5 ms | ✅ Pass |
