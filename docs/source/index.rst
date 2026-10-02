Welcome to BenoStreamDB
========================

.. image:: /_static/BenoStreamDB.png
   :align: center
   :width: 300px
   :alt: BenoStreamDB Logo

BenoStreamDB is an index-overlay engine for the lakehouse. We layer reconstructible, persistent secondary indexes—HNSW/IVF vector search, BM25 full-text, and CSR graphs—directly onto Apache Iceberg tables in object storage.

Built on Rust with Apache Arrow and DataFusion, it provides ultra-fast indexing and retrieval without the overhead of traditional database servers or data duplication.

Key Features
------------

*   **Serverless by Default**: Run directly against object storage via an embedded library. Zero operational overhead, scales to zero.
*   **Overlay Indexing**: Point BenoStreamDB at an existing Apache Iceberg table you do not own, and build indexes over that data in place. No data duplication.
*   **Compatible REST APIs**: Optional HTTP search gateway that is fully compatible with OpenSearch (Elasticsearch 7.10) and Qdrant Vector REST APIs.
*   **Hardware Accelerated**: Out-of-the-box GPU acceleration using NVIDIA CUDA, AMD ROCm, Apple Metal, and Intel XPU. Includes TurboQuant (TQ8/TQ4).
*   **SQL & pgvector**: Unified DataFusion SQL interface. Execute vector searches using familiar pgvector syntax directly over your lakehouse.
*   **Graph Analytics**: Traverse large-scale CSR graph indexes natively using SQL UDFs directly on your edge tables without a separate graph database.
*   **Ecosystem Connectors**: Native connectors for Apache Spark, Trino, and a complete dbt adapter for seamless data engineering.

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
   guides/iceberg_compatibility
   guides/iceberg_v2_v3_api
   guides/pgvector_sql_guide
   guides/opensearch_compatibility
   guides/qdrant_compatibility
   guides/spark_connector
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
