# Competitive Benchmark Report - HyperStreamDB

**Generated:** 2026-04-04 14:47:20.445270

## Ingest Performance

| System        | Hardware                             | Device   | Operation   |   Dataset Size |   Latency (ms) |   Throughput (rows/sec) |   Storage (MB) |
|:--------------|:-------------------------------------|:---------|:------------|---------------:|---------------:|------------------------:|---------------:|
| HyperStreamDB | AMD Ryzen 9 5900XT 16-Core Processor | cpu      | ingest      |           1000 |        261.494 |                 3824.18 |        7.11225 |

## Vector Search Performance

| System        | Hardware                             | Device   | Operation          |   Dataset Size |   Latency (ms) |   Throughput (rows/sec) |   Storage (MB) |
|:--------------|:-------------------------------------|:---------|:-------------------|---------------:|---------------:|------------------------:|---------------:|
| HyperStreamDB | AMD Ryzen 9 5900XT 16-Core Processor | cpu      | vector_search_k10  |           1000 |        16.8198 |                     nan |            nan |
| HyperStreamDB | AMD Ryzen 9 5900XT 16-Core Processor | cpu      | vector_search_k100 |           1000 |        18.0798 |                     nan |            nan |

## Hybrid Query Performance

## Key Findings

### HyperStreamDB Advantages

1. **Native Hybrid Queries**: Only system with scalar + vector in single query
2. **Iceberg Compatibility**: Standard data lake format
3. **Multi-Catalog Support**: Hive, Glue, Unity, REST, Nessie
4. **100% Iceberg v3 Compliance**: All required features implemented

### Competitive Position

