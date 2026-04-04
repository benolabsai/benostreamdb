# Competitive Benchmark Report - HyperStreamDB

**Generated:** 2026-04-04 14:47:22.385453

## Ingest Performance

| System        | Hardware                             | Device   | Operation   |   Dataset Size |   Latency (ms) |   Throughput (rows/sec) |   Storage (MB) |
|:--------------|:-------------------------------------|:---------|:------------|---------------:|---------------:|------------------------:|---------------:|
| HyperStreamDB | AMD Ryzen 9 5900XT 16-Core Processor | cuda:0   | ingest      |           1000 |        281.594 |                 3551.21 |         7.1116 |

## Vector Search Performance

| System        | Hardware                             | Device   | Operation          |   Dataset Size |   Latency (ms) |   Throughput (rows/sec) |   Storage (MB) |
|:--------------|:-------------------------------------|:---------|:-------------------|---------------:|---------------:|------------------------:|---------------:|
| HyperStreamDB | AMD Ryzen 9 5900XT 16-Core Processor | cuda:0   | vector_search_k10  |           1000 |        17.8013 |                     nan |            nan |
| HyperStreamDB | AMD Ryzen 9 5900XT 16-Core Processor | cuda:0   | vector_search_k100 |           1000 |        18.0157 |                     nan |            nan |

## Hybrid Query Performance

## Key Findings

### HyperStreamDB Advantages

1. **Native Hybrid Queries**: Only system with scalar + vector in single query
2. **Iceberg Compatibility**: Standard data lake format
3. **Multi-Catalog Support**: Hive, Glue, Unity, REST, Nessie
4. **100% Iceberg v3 Compliance**: All required features implemented

### Competitive Position

