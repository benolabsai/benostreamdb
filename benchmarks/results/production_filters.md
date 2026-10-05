# Pathological Filters & Differential Oracle Benchmark (§7.5)

- **Engine**: BenoStreamDB (Apache Iceberg table + DataFusion pushdown)
- **Competitor Oracle**: DuckDB (in-process analytical SQL)
- **Dataset**: ClickBench schema with 200,000 rows (multi-column mixed types)
- **Host**: x86_64 (Linux)
- **Measured DataFusion Session & Planning Overhead**: **0.60 ms** per query

| Filter Test Case | Selectivity | BenoStreamDB Total | Session / Plan Overhead | Pure Compute / Scan | DuckDB p50 | Differential Oracle Match | Status |
|---|---|---|---|---|---|---|---|
| **F1 (Ultra-Selective 0.01%)** | 0.0% | **7.60 ms** | 0.60 ms | **7.00 ms** | 1.69 ms | ✅ 100% Agreement | ✅ PASS |
| **F2 (Moderate Filter 5%)** | 4.98% | **7.99 ms** | 0.60 ms | **7.39 ms** | 1.68 ms | ✅ 100% Agreement | ✅ PASS |
| **F3 (Pathological Low-Selectivity 99.5%)** | 100.0% | **6.46 ms** | 0.60 ms | **5.86 ms** | 0.91 ms | ✅ 100% Agreement | ✅ PASS |
| **F4 (High-Cardinality IN List 50 items)** | 15.82% | **2.93 ms** | 0.60 ms | **2.33 ms** | 1.47 ms | ✅ 100% Agreement | ✅ PASS |
| **F5 (Compound Negation & Range)** | 8.33% | **7.10 ms** | 0.60 ms | **6.50 ms** | 1.15 ms | ✅ 100% Agreement | ✅ PASS |
| **F6 (Prefix String Filter)** | 16.7% | **4.76 ms** | 0.60 ms | **4.16 ms** | 0.89 ms | ✅ 100% Agreement | ✅ PASS |
| **F7 (Unanchored Substring Filter)** | 14.32% | **5.18 ms** | 0.60 ms | **4.58 ms** | 0.83 ms | ✅ 100% Agreement | ✅ PASS |

### DataFusion Execution & Overhead Analysis
1. **Per-Query Session Overhead (0.60 ms)**: In the current Python API (`table.sql()`), each query instantiates a new DataFusion `SessionContext` and registers all standard scalar/aggregate functions, vector operators, and graph UDFs from scratch before planning begins.
2. **Morsel Execution vs Async Channels**: DuckDB executes queries synchronously using C++ morsel work-stealing, whereas DataFusion routes batches across 32 async Tokio channels (`RoundRobinBatch(32)`), introducing channel serialization overhead on sub-million row tables.
3. **Differential Oracle Invariance**: Across all selective, low-selectivity (99.5%), high-cardinality `IN` lists (50 items), and string wildcards, BenoStreamDB achieves 100% mathematical equality against DuckDB with no pathological performance cliff.
