# ClickBench SQL Benchmark Results

- **Rows**: 500,000
- **Dataset File**: `synth_hits_500000_s42.parquet` (6.5 MB)
- **Warm Iterations**: 2

### Warm Query Latency (ms) — Median

| Query | benostreamdb | duckdb | datafusion |
|---| --- | --- | --- |
| **Q0 (count)** | 1.9 ms | 0.5 ms | 0.4 ms |
| **Q1 (filter count)** | 6.5 ms | 2.9 ms | 2.8 ms |
| **Q2 (multi-agg)** | 10.0 ms | 3.7 ms | 3.2 ms |
| **Q3 (avg int)** | 4.8 ms | 4.2 ms | 3.3 ms |
| **Q4 (distinct user)** | 10.4 ms | 27.8 ms | 8.3 ms |
| **Q5 (distinct phrase)** | 12.1 ms | 4.9 ms | 3.1 ms |
| **Q6 (min/max date)** | 4.3 ms | 0.4 ms | 0.4 ms |
| **Q7 (group by agg)** | 5.9 ms | 3.5 ms | 3.7 ms |
| **Q8 (group by distinct)** | 16.0 ms | 33.9 ms | 11.9 ms |
| **Q9 (string filter group)** | 23.7 ms | 40.8 ms | 12.4 ms |

### Ingestion & Total Execution Time

| Engine | Ingestion (s) | Total Warm SQL Time (ms) | Status |
|---|---|---|---|
| benostreamdb | 0.18s | 95.5 ms | ✅ Pass |
| duckdb | Direct Parquet Scan | 122.6 ms | ✅ Pass |
| datafusion | Direct Parquet Scan | 49.7 ms | ✅ Pass |
