# Architecture

HyperStreamDB is a **Universal Vector & Metadata Streaming** storage engine. It bridges the gap between streaming data management and high-dimensional vector indexing by providing an "Overlay Index" on top of standard file formats (Parquet).

## Core Philosophy

**"Your Data is Standard. Your Index is Custom."**

HyperStreamDB adds a sophisticated sidecar index (Roaring + HNSW) *on top* of standard files.
*   **Compatibility**: You can still read your data with standard tools (Spark, Presto, Pandas) at normal speed.
*   **Performance**: HyperStreamDB-aware readers get **O(log n)** lookup speed using inverted and vector indexes.

## Storage Layout: The "Hybrid Segment"

Data is written in immutable **Segments**. Each segment is a self-contained unit comprising:

1.  **Raw Data**:
    *   `data.vector.bin`: Raw vector data (Generic Tensor support).
    *   `data.metadata.json` / `data.parquet`: Raw metadata.
2.  **Indexes**:
    *   **`idx.meta.<col>.inv`**: Inverted Indexes (RoaringBitmaps) for scalar filtering.
    *   **`idx.vector.ann`**: Vector Index (HNSW, DiskANN) for similarity search.

## The Streaming Read Path

HyperStreamDB enables "Pre-Filtered Vector Search" by combining scalar pruning with vector search.

1.  **Scalar Pruning**: The reader loads the inverted index for the filtered column (e.g., `category`) and identifies relevant rows using efficient bitmap operations.
2.  **Restricted Vector Search**:
    *   If selectivity is high (few rows match), exact distance calculation is performed on candidate vectors.
    *   If selectivity is low, a graph traversal (HNSW) is performed rooted at the candidates.

## Implementation Strategy

*   **Core Engine**: Written in **Rust** for zero-overhead, embeddability (Lambda, Browser via WASM), and low-level system control.
*   **Bindings**: 
    *   **Python**: For data science and AI agents (via PyO3).
    *   **Trino/Spark**: For distributed SQL and ETL (via JNI & Arrow C Data Interface).

## "Serverless Index-Streaming"

HyperStreamDB enables the **Client** to become the database engine. By streaming lightweight, aligned indexes first, a stateless client can perform complex, indexed queries over massive datasets directly from S3, paying **zero** compute cost when idle.
