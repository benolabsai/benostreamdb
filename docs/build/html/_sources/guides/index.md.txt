# HyperStreamDB

HyperStreamDB is a high-performance, indexed streaming lakehouse built in Rust. It is designed to provide ultra-low latency queries on streaming data by maintaining real-time indexes (HNSW for vectors, Inverted for scalars) directly on the storage format.

## Key Features

*   **Streaming-First**: Designed for continuous ingestion and real-time query availability.
*   **Indexed Storage**: Native support for Vector (HNSW) and Scalar (Inverted) indexes, enabling sub-second latency on massive datasets.
*   **Multiglot**: Core in Rust, with high-performance bindings for Python, and connectors for Trino and Spark via JNI.
*   **Zero-Copy**: Leverages Arrow C Data Interface for zero-copy data sharing between the storage engine and compute engines.
*   **Storage Agnostic**: Supports local filesystem, S3, GCS, and Azure Blob Storage.

## Getting Started

Check out the [Architecture](architecture.md) guide to understand how HyperStream works, or jump into the [Python Bindings](integrations/python.md) to start using it today.
