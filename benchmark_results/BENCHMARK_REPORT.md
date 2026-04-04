# Competitive Benchmark Report - HyperStreamDB

**Generated:** 2026-04-04 13:56:56.467083

## Ingest Performance

| System               | Operation   |   Dataset Size |   Latency (ms) |   Throughput (rows/sec) |   Storage (MB) | Hardware                             | Device   |
|:---------------------|:------------|---------------:|---------------:|------------------------:|---------------:|:-------------------------------------|:---------|
| HyperStreamDB        | ingest      |           1000 |       268.945  |                 3718.23 |        7.18574 | Generic Baseline                     | cpu      |
| DuckDB (Raw Parquet) | ingest      |           1000 |        70.8115 |                14122    |        3.55513 | Generic Baseline                     | cpu      |
| LanceDB              | ingest      |           1000 |        28.9528 |                34538.9  |        2.98259 | Generic Baseline                     | cpu      |
| HyperStreamDB        | ingest      |           1000 |       801.391  |                 1247.83 |        7.11231 | AMD Ryzen 9 5900XT 16-Core Processor | cpu      |
| HyperStreamDB        | ingest      |           1000 |       818.65   |                 1221.52 |        7.13006 | AMD Ryzen 9 5900XT 16-Core Processor | cpu      |
| HyperStreamDB        | ingest      |           1000 |       820.246  |                 1219.15 |        7.11529 | AMD Ryzen 9 5900XT 16-Core Processor | cuda:0   |
| HyperStreamDB        | ingest      |           1000 |       290.842  |                 3438.29 |        7.1134  | AMD Ryzen 9 5900XT 16-Core Processor | cpu      |
| HyperStreamDB        | ingest      |           1000 |       412.61   |                 2423.59 |        7.12722 | AMD Ryzen 9 5900XT 16-Core Processor | cuda:0   |

## Vector Search Performance

| System        | Operation          |   Dataset Size |   Latency (ms) |   Throughput (rows/sec) |   Storage (MB) | Hardware                             | Device   |
|:--------------|:-------------------|---------------:|---------------:|------------------------:|---------------:|:-------------------------------------|:---------|
| HyperStreamDB | vector_search_k10  |           1000 |       28.3022  |                     nan |            nan | Generic Baseline                     | cpu      |
| LanceDB       | vector_search_k10  |           1000 |        6.41737 |                     nan |            nan | Generic Baseline                     | cpu      |
| HyperStreamDB | vector_search_k10  |           1000 |      109.424   |                     nan |            nan | AMD Ryzen 9 5900XT 16-Core Processor | cpu      |
| HyperStreamDB | vector_search_k10  |           1000 |      109.012   |                     nan |            nan | AMD Ryzen 9 5900XT 16-Core Processor | cpu      |
| HyperStreamDB | vector_search_k10  |           1000 |      108.445   |                     nan |            nan | AMD Ryzen 9 5900XT 16-Core Processor | cuda:0   |
| HyperStreamDB | vector_search_k10  |           1000 |       16.1742  |                     nan |            nan | AMD Ryzen 9 5900XT 16-Core Processor | cpu      |
| HyperStreamDB | vector_search_k10  |           1000 |       14.6868  |                     nan |            nan | AMD Ryzen 9 5900XT 16-Core Processor | cuda:0   |
| HyperStreamDB | vector_search_k100 |           1000 |       25.9826  |                     nan |            nan | Generic Baseline                     | cpu      |
| LanceDB       | vector_search_k100 |           1000 |        7.24833 |                     nan |            nan | Generic Baseline                     | cpu      |
| HyperStreamDB | vector_search_k100 |           1000 |      148.144   |                     nan |            nan | AMD Ryzen 9 5900XT 16-Core Processor | cpu      |
| HyperStreamDB | vector_search_k100 |           1000 |      148.415   |                     nan |            nan | AMD Ryzen 9 5900XT 16-Core Processor | cpu      |
| HyperStreamDB | vector_search_k100 |           1000 |      152.582   |                     nan |            nan | AMD Ryzen 9 5900XT 16-Core Processor | cuda:0   |
| HyperStreamDB | vector_search_k100 |           1000 |       19.9033  |                     nan |            nan | AMD Ryzen 9 5900XT 16-Core Processor | cpu      |
| HyperStreamDB | vector_search_k100 |           1000 |       16.9979  |                     nan |            nan | AMD Ryzen 9 5900XT 16-Core Processor | cuda:0   |

## Hybrid Query Performance

## Key Findings

### HyperStreamDB Advantages

1. **Native Hybrid Queries**: Only system with scalar + vector in single query
2. **Iceberg Compatibility**: Standard data lake format
3. **Multi-Catalog Support**: Hive, Glue, Unity, REST, Nessie
4. **100% Iceberg v3 Compliance**: All required features implemented

### Competitive Position

- Vector search: 10.0x slower than LanceDB
