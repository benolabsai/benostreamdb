use lazy_static::lazy_static;
use prometheus::{
    register_histogram, register_int_counter, register_int_gauge, Histogram, IntCounter, IntGauge,
};

lazy_static! {
    /// Total number of rows ingested
    pub static ref INGEST_ROWS_TOTAL: IntCounter = register_int_counter!(
        "hyperstreamdb_ingest_rows_total",
        "Total number of rows ingested"
    )
    .unwrap();
    
    /// Query latency in seconds
    pub static ref QUERY_LATENCY_SECONDS: Histogram = register_histogram!(
        "hyperstreamdb_query_latency_seconds",
        "Query latency in seconds"
    )
    .unwrap();

    /// Compaction duration in seconds
    pub static ref COMPACTION_DURATION_SECONDS: Histogram = register_histogram!(
        "hyperstreamdb_compaction_duration_seconds",
        "Compaction duration in seconds"
    )
    .unwrap();

    /// Number of active parquet files
     pub static ref ACTIVE_FILES_GAUGE: IntGauge = register_int_gauge!(
        "hyperstreamdb_active_files",
        "Number of active parquet files in the table"
    ).unwrap();
}
