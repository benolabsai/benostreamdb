# ClickBench SQL Benchmark Results

- **Rows**: 500,000
- **Dataset File**: `synth_hits_500000_s42.parquet` (6.5 MB)
- **Warm Iterations**: 2

### Warm Query Latency (ms) — Median

| Query | benostreamdb | duckdb | datafusion |
|---| --- | --- | --- |
| **Q0 (count)** | 2.2 ms | 0.7 ms | 0.4 ms |
| **Q1 (filter count)** | 6.2 ms | 3.5 ms | 2.8 ms |
| **Q2 (multi-agg)** | 9.4 ms | 3.4 ms | 3.8 ms |
| **Q3 (avg int)** | 5.1 ms | 4.1 ms | 3.6 ms |
| **Q4 (distinct user)** | 10.1 ms | 26.4 ms | 8.8 ms |
| **Q5 (distinct phrase)** | 14.4 ms | 5.3 ms | 3.6 ms |
| **Q6 (min/max date)** | 4.8 ms | 0.4 ms | 0.4 ms |
| **Q7 (group by agg)** | 6.7 ms | 4.0 ms | 3.8 ms |
| **Q8 (group by distinct)** | 16.7 ms | 34.1 ms | 14.3 ms |
| **Q9 (string filter group)** | 22.6 ms | 42.5 ms | 15.3 ms |

### Ingestion & Total Execution Time

| Engine | Ingestion (s) | Total Warm SQL Time (ms) | Status |
|---|---|---|---|
| benostreamdb | 0.18s | 98.1 ms | ✅ Pass |
| duckdb | Direct Parquet Scan | 124.3 ms | ✅ Pass |
| datafusion | Direct Parquet Scan | 56.9 ms | ✅ Pass |
