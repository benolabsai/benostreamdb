Welcome to BenoStreamDB
========================

.. image:: /_static/BenoStreamDB.png
   :align: center
   :width: 300px
   :alt: BenoStreamDB Logo

BenoStreamDB is an index-overlay engine for the lakehouse built in Rust. It layers reconstructible, persistent secondary indexes—HNSW vector search, BM25 Okapi full-text, Roaring Bitmaps, JSON path filters, and CSR graphs—directly onto Apache Iceberg tables in object storage.

Key Architecture Highlights
----------------------------

*   **Serverless by Default**: Run directly against object storage via an embedded library (Python, Rust, JVM) with zero database server overhead.
*   **Advisory Index Overlays**: Build rebuildable secondary indexes alongside standard Parquet data files. The underlying data is never rewritten or locked into a proprietary format.
*   **Vector Search & pgvector SQL**: Query vector indexes directly using standard pgvector SQL operators (``<->``, ``<=>``, ``<#>``) in DataFusion.
*   **Graph Analytics & Graph RAG**: Query CSR graph index overlays using native SQL table functions directly over Iceberg edge tables.
*   **Ecosystem Connectors**: Native connectors for Apache Spark, Trino, and dbt, plus optional Arrow Flight SQL and REST search gateways.
*   **Grounded Hardware Support**: Dynamic batch GPU acceleration via NVIDIA CUDA (runtime NVRTC), Apple Metal, and Vulkan/WGPU, with SIMD-optimized CPU execution as the fast, low-latency default.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   guides/installation
   guides/architecture
   guides/adr_001

.. toctree::
   :maxdepth: 2
   :caption: Connectors & Integrations

   integrations/python
   integrations/spark
   integrations/trino
   integrations/dbt
   integrations/flight_sql
   integrations/search_gateway
   integrations/mcp
   integrations/java_jni

.. toctree::
   :maxdepth: 2
   :caption: Table Formats & Catalogs

   guides/iceberg_compatibility
   guides/iceberg_v2_v3_api
   guides/catalog_usage

.. toctree::
   :maxdepth: 2
   :caption: Query & Search

   guides/pgvector_sql_guide
   guides/python_vector_api
   guides/graph_rag_edge_tables
   guides/gpu_setup_guide

.. toctree::
   :maxdepth: 2
   :caption: Operations & Hardening

   guides/configuration
   guides/concurrency
   guides/resource_limits
   guides/admin_cli
   guides/monitoring
   guides/recovery_runbook
   guides/no_panic_policy
   guides/dependency_risk

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/01_ai_retrieval_mcp
   tutorials/02_json_document_retrieval
   tutorials/03_graph_traversal

.. toctree::
   :maxdepth: 2
   :caption: Benchmarks & Evaluation

   guides/benchmarking
   guides/benchmarking_plan
   guides/production_readiness

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/python
   api/rust

.. toctree::
   :maxdepth: 1
   :caption: Roadmap

   guides/roadmap
