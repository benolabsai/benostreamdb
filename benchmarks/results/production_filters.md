# Pathological Filters & Differential Oracle Benchmark (§7.5)

- **Engine**: BenoStreamDB (Apache Iceberg table + DataFusion pushdown)
- **Competitor Oracle**: DuckDB (in-process analytical SQL)
- **Dataset**: ClickBench schema with 200,000 rows (multi-column mixed types)
- **Host**: x86_64 (Linux)
- **Measured DataFusion Session & Planning Overhead**: **0.91 ms** per query

| Filter Test Case | Selectivity | BenoStreamDB Total | Session / Plan Overhead | Pure Compute / Scan | DuckDB p50 | Differential Oracle Match | Status |
|---|---|---|---|---|---|---|---|
| **F1 (Ultra-Selective 0.01%)** | 0.0% | **10.67 ms** | 0.91 ms | **9.76 ms** | 2.12 ms | ✅ 100% Agreement | ✅ PASS |
| **F2 (Moderate Filter 5%)** | 4.98% | **10.41 ms** | 0.91 ms | **9.50 ms** | 2.48 ms | ✅ 100% Agreement | ✅ PASS |
| **F3 (Pathological Low-Selectivity 99.5%)** | 100.0% | **8.65 ms** | 0.91 ms | **7.74 ms** | 1.06 ms | ✅ 100% Agreement | ✅ PASS |
| **F4 (High-Cardinality IN List 50 items)** | 15.82% | **4.05 ms** | 0.91 ms | **3.14 ms** | 2.37 ms | ✅ 100% Agreement | ✅ PASS |
| **F5 (Compound Negation & Range)** | 8.33% | **10.93 ms** | 0.91 ms | **10.02 ms** | 1.60 ms | ✅ 100% Agreement | ✅ PASS |
| **F6 (Prefix String Filter)** | 16.7% | **7.02 ms** | 0.91 ms | **6.11 ms** | 1.11 ms | ✅ 100% Agreement | ✅ PASS |
| **F7 (Unanchored Substring Filter)** | 14.32% | **7.20 ms** | 0.91 ms | **6.28 ms** | 1.07 ms | ✅ 100% Agreement | ✅ PASS |

### DataFusion Execution & Overhead Analysis
1. **Per-Query Session Overhead (0.91 ms)**: In the current Python API (`table.sql()`), each query instantiates a new DataFusion `SessionContext` and registers all standard scalar/aggregate functions, vector operators, and graph UDFs from scratch before planning begins.
2. **Morsel Execution vs Async Channels**: DuckDB executes queries synchronously using C++ morsel work-stealing, whereas DataFusion routes batches across 32 async Tokio channels (`RoundRobinBatch(32)`), introducing channel serialization overhead on sub-million row tables.
3. **Differential Oracle Invariance**: Across all selective, low-selectivity (99.5%), high-cardinality `IN` lists (50 items), and string wildcards, BenoStreamDB achieves 100% mathematical equality against DuckDB with no pathological performance cliff.
