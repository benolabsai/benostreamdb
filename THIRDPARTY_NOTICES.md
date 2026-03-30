# Third-Party Notices

HyperStreamDB incorporates concepts and logic from the following open-source projects. We are grateful to the authors for their contributions to the ecosystem.

## Lance / LanceDB
HyperStreamDB's "Vector Shuffling" and ingestion pipeline optimizations (e.g., IVF-based row reordering) are inspired by the [Lance project](https://github.com/lance-format/lance).

- **License**: Apache License 2.0
- **Project URL**: [https://github.com/lancedb/lancedb](https://github.com/lancedb/lancedb)
- **Copyright**: Copyright The Lance Authors

The logic for partitioning and shuffling vectors during ingestion has been adapted for use in HyperStreamDB's Parquet-based storage engine. In accordance with the Apache 2.0 license:
- Any adapted source files in `src/core/` will contain appropriate attribution comments.
- HyperStreamDB is licensed under its own terms, while respecting the original copyright of adapted components.

## hnsw_rs
HyperStreamDB relies on the open-source `hnsw_rs` library for core Hierarchical Navigable Small World (HNSW) graph traversal, which we have locally vendored and patched to support exact pre-filtering.

- **License**: MIT / Apache License 2.0
- **Project URL**: [https://github.com/jean-pierreBoth/hnswlib-rs](https://github.com/jean-pierreBoth/hnswlib-rs)
- **Copyright**: Copyright Jean-Pierre Both and the hnsw_rs authors

The source code is vendored under `vendor/hnsw_rs/` and integrated into HyperStreamDB's graph partitioning strategies in `src/core/index/hnsw_ivf.rs`.
