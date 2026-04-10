Welcome to HyperStreamDB
========================

.. image:: /_static/HyperStreamDB.png
   :align: center
   :width: 300px
   :alt: HyperStreamDB Logo

HyperStreamDB is a serverless, hybrid-search database optimized for high-performance vector and scalar queries directly on data lakes (S3, GCS, Azure, Local).

Built on Rust with Apache Arrow and DataFusion, it provides ultra-fast indexing and retrieval without the overhead of traditional database servers.

Key Features
------------

*   **Hybrid Vector Search**: Approximate Nearest Neighbor (ANN) search with HNSW-IVF.
*   **Vectorized SQL**: Full SQL support with pgvector-compatible operators.
*   **Storage-Native**: Native support for Iceberg and Parquet formats.
*   **Hardware Acceleration**: Blazing fast search using CUDA, Metal, ROCm, and AVX-512.
*   **Transactional Snapshots**: ACID-compliant updates via Optimistic Concurrency Control.
*   **Multi-Catalog Support**: Seamless integration with AWS Glue, Nessie, and Hive Metastore.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   guides/installation
   guides/comprehensive_guide

.. toctree::
   :maxdepth: 2
   :caption: User Guides

   guides/python_vector_api
   guides/gpu_setup_guide
   guides/configuration
   guides/concurrency

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/python
   api/rust

.. toctree::
   :maxdepth: 1
   :caption: Roadmap

   roadmap
