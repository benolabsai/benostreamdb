// Copyright (c) 2026 Richard Albright. All rights reserved.

use lazy_static::lazy_static;
use prometheus::{
    register_histogram, register_int_counter, register_int_counter_vec, register_int_gauge, Histogram, IntCounter, IntCounterVec, IntGauge,
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

    /// Cache hits across various system caches
    pub static ref CACHE_HITS_TOTAL: IntCounterVec = register_int_counter_vec!(
        "hyperstreamdb_cache_hits_total",
        "Total number of cache hits",
        &["cache_name"]
    ).unwrap();

    /// Cache misses across various system caches
    pub static ref CACHE_MISSES_TOTAL: IntCounterVec = register_int_counter_vec!(
        "hyperstreamdb_cache_misses_total",
        "Total number of cache misses",
        &["cache_name"]
    ).unwrap();

    /// Total I/O bytes read
    pub static ref IO_BYTES_READ_TOTAL: IntCounter = register_int_counter!(
        "hyperstreamdb_io_bytes_read_total",
        "Total number of bytes read from storage"
    ).unwrap();

    /// Total I/O bytes written
    pub static ref IO_BYTES_WRITTEN_TOTAL: IntCounter = register_int_counter!(
        "hyperstreamdb_io_bytes_written_total",
        "Total number of bytes written to storage"
    ).unwrap();

    /// Search latency in seconds (for vector and keyword searches)
    pub static ref SEARCH_LATENCY_SECONDS: Histogram = register_histogram!(
        "hyperstreamdb_search_latency_seconds",
        "Search operation latency in seconds"
    ).unwrap();

    /// Commit duration in seconds (manifest flush)
    pub static ref COMMIT_DURATION_SECONDS: Histogram = register_histogram!(
        "hyperstreamdb_commit_duration_seconds",
        "Commit (manifest flush) duration in seconds"
    ).unwrap();

    /// Index build duration in seconds (HNSW-IVF construction)
    pub static ref INDEX_BUILD_DURATION_SECONDS: Histogram = register_histogram!(
        "hyperstreamdb_index_build_duration_seconds",
        "Index build (HNSW-IVF) duration in seconds"
    ).unwrap();

    /// Number of active segments (distinct from active parquet files)
    pub static ref ACTIVE_SEGMENTS_GAUGE: IntGauge = register_int_gauge!(
        "hyperstreamdb_active_segments",
        "Number of active segments in the table"
    ).unwrap();

    /// Number of manifest commit conflicts (concurrent writers)
    pub static ref MANIFEST_CONFLICTS_TOTAL: IntCounter = register_int_counter!(
        "hyperstreamdb_manifest_conflicts_total",
        "Number of manifest commit conflicts detected"
    ).unwrap();
}
