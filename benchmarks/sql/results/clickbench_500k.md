# ClickBench SQL Benchmark Results

- **Rows**: 500,000
- **Dataset File**: `synth_hits_500000_s42.parquet` (6.5 MB)
- **Warm Iterations**: 2

### Warm Query Latency (ms) — Median

| Query | benostreamdb | duckdb | datafusion |
|---| --- | --- | --- |
| **Q0 (count)** | 2.0 ms | 0.5 ms | N/A |
| **Q1 (filter count)** | 6.1 ms | 3.1 ms | N/A |
| **Q2 (multi-agg)** | 7.9 ms | 3.2 ms | N/A |
| **Q3 (avg int)** | 4.7 ms | 4.9 ms | N/A |
| **Q4 (distinct user)** | 10.5 ms | 19.4 ms | N/A |
| **Q5 (distinct phrase)** | 14.1 ms | 4.2 ms | N/A |
| **Q6 (min/max date)** | 5.2 ms | 0.5 ms | N/A |
| **Q7 (group by agg)** | 6.0 ms | 3.7 ms | N/A |
| **Q8 (group by distinct)** | 15.4 ms | 21.1 ms | N/A |
| **Q9 (string filter group)** | 21.1 ms | 29.5 ms | N/A |

### Ingestion & Total Execution Time

| Engine | Ingestion (s) | Total Warm SQL Time (ms) | Status |
|---|---|---|---|
| benostreamdb | 0.18s | 92.9 ms | ✅ Pass |
| duckdb | Direct Parquet Scan | 90.1 ms | ✅ Pass |
| datafusion | - | - | ❌ datafusion not installed |
