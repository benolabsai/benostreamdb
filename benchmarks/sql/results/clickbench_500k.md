# ClickBench SQL Benchmark Results

- **Rows**: 500,000
- **Dataset File**: `synth_hits_500000_s42.parquet` (6.5 MB)
- **Warm Iterations**: 2

### Warm Query Latency (ms) — Median

| Query | benostreamdb | duckdb | datafusion |
|---| --- | --- | --- |
| **Q0 (count)** | 12.5 ms | 0.4 ms | 0.4 ms |
| **Q1 (filter count)** | 26.5 ms | 2.8 ms | 2.7 ms |
| **Q2 (multi-agg)** | 39.6 ms | 3.5 ms | 3.9 ms |
| **Q3 (avg int)** | 26.4 ms | 4.0 ms | 3.1 ms |
| **Q4 (distinct user)** | 56.7 ms | 27.6 ms | 7.8 ms |
| **Q5 (distinct phrase)** | 122.8 ms | 4.8 ms | 2.9 ms |
| **Q6 (min/max date)** | 26.4 ms | 0.4 ms | 0.4 ms |
| **Q7 (group by agg)** | 36.3 ms | 3.5 ms | 3.6 ms |
| **Q8 (group by distinct)** | 108.5 ms | 31.5 ms | 12.6 ms |
| **Q9 (string filter group)** | 195.3 ms | 35.6 ms | 12.7 ms |

### Ingestion & Total Execution Time

| Engine | Ingestion (s) | Total Warm SQL Time (ms) | Status |
|---|---|---|---|
| benostreamdb | 1.33s | 651.1 ms | ✅ Pass |
| duckdb | Direct Parquet Scan | 114.2 ms | ✅ Pass |
| datafusion | Direct Parquet Scan | 50.1 ms | ✅ Pass |
