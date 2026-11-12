# Pathological Filters & Differential Oracle Benchmark (§7.5)

- **Engine**: BenoStreamDB (Apache Iceberg table + DataFusion pushdown)
- **Competitor Oracle**: DuckDB (in-process analytical SQL)
- **Dataset**: ClickBench schema with 200,000 rows (multi-column mixed types)
- **Host**: x86_64 (Linux)
- **Measured DataFusion Session & Planning Overhead**: **0.79 ms** per query

| Filter Test Case | Selectivity | BenoStreamDB Total | Session / Plan Overhead | Pure Compute / Scan | DuckDB p50 | Differential Oracle Match | Status |
|---|---|---|---|---|---|---|---|
| **F1 (Ultra-Selective 0.01%)** | 0.0% | **7.86 ms** | 0.79 ms | **7.07 ms** | 1.79 ms | ✅ 100% Agreement | ✅ PASS |
| **F2 (Moderate Filter 5%)** | 4.98% | **9.50 ms** | 0.79 ms | **8.71 ms** | 1.74 ms | ✅ 100% Agreement | ✅ PASS |
| **F3 (Pathological Low-Selectivity 99.5%)** | 100.0% | **7.38 ms** | 0.79 ms | **6.59 ms** | 0.83 ms | ✅ 100% Agreement | ✅ PASS |
| **F4 (High-Cardinality IN List 50 items)** | 15.82% | **3.26 ms** | 0.79 ms | **2.47 ms** | 1.66 ms | ✅ 100% Agreement | ✅ PASS |
| **F5 (Compound Negation & Range)** | 8.33% | **8.19 ms** | 0.79 ms | **7.40 ms** | 1.27 ms | ✅ 100% Agreement | ✅ PASS |
| **F6 (Prefix String Filter)** | 16.7% | **5.15 ms** | 0.79 ms | **4.36 ms** | 0.88 ms | ✅ 100% Agreement | ✅ PASS |
| **F7 (Unanchored Substring Filter)** | 14.32% | **6.32 ms** | 0.79 ms | **5.53 ms** | 1.02 ms | ✅ 100% Agreement | ✅ PASS |

### DataFusion Execution & Overhead Analysis
1. **Per-Query Session Overhead (0.79 ms)**: In the current Python API (`table.sql()`), each query instantiates a new DataFusion `SessionContext` and registers all standard scalar/aggregate functions, vector operators, and graph UDFs from scratch before planning begins.
2. **Morsel Execution vs Async Channels**: DuckDB executes queries synchronously using C++ morsel work-stealing, whereas DataFusion routes batches across 32 async Tokio channels (`RoundRobinBatch(32)`), introducing channel serialization overhead on sub-million row tables.
3. **Differential Oracle Invariance**: Across all selective, low-selectivity (99.5%), high-cardinality `IN` lists (50 items), and string wildcards, BenoStreamDB achieves 100% mathematical equality against DuckDB with no pathological performance cliff.
