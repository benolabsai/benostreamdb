Welcome to BenoStreamDB
========================

.. image:: /_static/BenoStreamDB.png
   :align: center
   :width: 300px
   :alt: BenoStreamDB Logo

BenoStreamDB is a serverless, hybrid-search database optimized for high-performance vector and scalar queries directly on data lakes (S3, GCS, Azure, Local).

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
   guides/architecture

.. toctree::
   :maxdepth: 2
   :caption: User Guides

   guides/python_vector_api
   guides/gpu_setup_guide
   guides/configuration
   guides/concurrency
   guides/catalog_usage
   guides/monitoring
   guides/admin_cli
   guides/graph_rag_edge_tables
   guides/iceberg_v2_v3_api
   guides/pgvector_sql_guide
   guides/opensearch_compatibility
   guides/resource_limits
   guides/no_panic_policy
   guides/benchmarking

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/python
   api/rust

.. toctree::
   :maxdepth: 1
   :caption: Roadmap

   guides/roadmap
