# Integrations

BenoStreamDB is designed to be accessible from multiple languages and processing engines. The core engine is written in Rust, but we provide:

*   **[Python Bindings](python.md)**: Native Python integration via PyO3 for Data Science and AI workflows.
*   **[Arrow Flight / Flight SQL](../../server/flight_sql/README.md)**: Native high-performance Arrow data transport and standard Flight SQL database connectivity (Optional Server Interface).
*   **[Trino Connector](trino.md)**: Distributed SQL queries over BenoStreamDB tables.
*   **[Spark Connector](spark.md)**: Batch ETL and structured streaming.
*   **[dbt Adapter](../../dbt-benostreamdb/README.md)**: Modern analytics engineering and data transformation.
*   **[Java JNI](java_jni.md)**: Low-level access for JVM languages.
