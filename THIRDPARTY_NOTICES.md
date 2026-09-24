# Third-Party Notices

BenoStreamDB incorporates concepts and logic from the following open-source projects. We are grateful to the authors for their contributions to the ecosystem.

## Lance / LanceDB
BenoStreamDB's "Vector Shuffling" and ingestion pipeline optimizations (e.g., IVF-based row reordering) are inspired by the [Lance project](https://github.com/lance-format/lance).

- **License**: Apache License 2.0
- **Project URL**: [https://github.com/lancedb/lancedb](https://github.com/lancedb/lancedb)
- **Copyright**: Copyright The Lance Authors

The logic for partitioning and shuffling vectors during ingestion has been adapted for use in BenoStreamDB's Parquet-based storage engine. In accordance with the Apache 2.0 license:
- Any adapted source files in `src/core/` will contain appropriate attribution comments.
- BenoStreamDB is licensed under its own terms, while respecting the original copyright of adapted components.

## hnsw_rs
BenoStreamDB relies on the open-source `hnsw_rs` library for core Hierarchical Navigable Small World (HNSW) graph traversal, which we have locally vendored and patched to support exact pre-filtering.

- **License**: MIT / Apache License 2.0
- **Project URL**: [https://github.com/jean-pierreBoth/hnswlib-rs](https://github.com/jean-pierreBoth/hnswlib-rs)
- **Copyright**: Copyright Jean-Pierre Both and the hnsw_rs authors

The source code is internalized under `src/core/index/hnsw_rs/` and integrated into BenoStreamDB's graph partitioning strategies in `src/core/index/hnsw_ivf.rs`.

## Microsoft GraphRAG
BenoStreamDB's DRIFT Search algorithm (`mode='drift'` in `graph_rag_search()`) is adapted from the [Microsoft GraphRAG](https://github.com/microsoft/graphrag) project's DRIFT (Dynamic Reasoning and Inference with Flexible Traversal) search implementation.

- **License**: MIT License
- **Project URL**: [https://github.com/microsoft/graphrag](https://github.com/microsoft/graphrag)
- **Copyright**: Copyright (c) Microsoft Corporation.

The DRIFT search algorithm — including the multi-phase primer/follow-up/reduction architecture, the `DriftAction` search tree, and the `DriftQueryState` traversal state management — has been adapted for use in BenoStreamDB's Python Graph RAG API. The implementation in `python/benostreamdb/__init__.py` contains the adapted algorithm with BenoStreamDB-native graph primitives (Louvain communities, Personalized PageRank, HNSW vector search) replacing the original's LLM-centric context builders.

In accordance with the MIT License, the full license text is reproduced below:

```
MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

