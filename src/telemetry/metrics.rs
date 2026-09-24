// Copyright (c) 2026 Richard Albright. All rights reserved.

// Metric registration uses static, compile-time-constant names and help strings.
// The `prometheus` `register_*` macros are only fallible on an invalid name or a
// duplicate registration — a programming error caught by the tests, never a
// runtime condition — and the crate offers no infallible constructor. This is a
// documented residual invariant in NO_PANIC_POLICY.md; failure happens once at
// first access, not on a request or ingest path.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use lazy_static::lazy_static;
use prometheus::{
    register_histogram, register_int_counter, register_int_counter_vec, register_int_gauge,
    Histogram, IntCounter, IntCounterVec, IntGauge,
};

lazy_static! {
    /// Total number of rows ingested
    pub static ref INGEST_ROWS_TOTAL: IntCounter = register_int_counter!(
        "benostreamdb_ingest_rows_total",
        "Total number of rows ingested"
    )
    .unwrap();

    /// Query latency in seconds
    pub static ref QUERY_LATENCY_SECONDS: Histogram = register_histogram!(
        "benostreamdb_query_latency_seconds",
        "Query latency in seconds"
    )
    .unwrap();

    /// Compaction duration in seconds
    pub static ref COMPACTION_DURATION_SECONDS: Histogram = register_histogram!(
        "benostreamdb_compaction_duration_seconds",
        "Compaction duration in seconds"
    )
    .unwrap();

    /// Number of active parquet files
    pub static ref ACTIVE_FILES_GAUGE: IntGauge = register_int_gauge!(
        "benostreamdb_active_files",
        "Number of active parquet files in the table"
    ).unwrap();

    /// Cache hits across various system caches
    pub static ref CACHE_HITS_TOTAL: IntCounterVec = register_int_counter_vec!(
        "benostreamdb_cache_hits_total",
        "Total number of cache hits",
        &["cache_name"]
    ).unwrap();

    /// Cache misses across various system caches
    pub static ref CACHE_MISSES_TOTAL: IntCounterVec = register_int_counter_vec!(
        "benostreamdb_cache_misses_total",
        "Total number of cache misses",
        &["cache_name"]
    ).unwrap();

    /// Total I/O bytes read
    pub static ref IO_BYTES_READ_TOTAL: IntCounter = register_int_counter!(
        "benostreamdb_io_bytes_read_total",
        "Total number of bytes read from storage"
    ).unwrap();

    /// Total I/O bytes written
    pub static ref IO_BYTES_WRITTEN_TOTAL: IntCounter = register_int_counter!(
        "benostreamdb_io_bytes_written_total",
        "Total number of bytes written to storage"
    ).unwrap();

    /// Search latency in seconds (for vector and keyword searches)
    pub static ref SEARCH_LATENCY_SECONDS: Histogram = register_histogram!(
        "benostreamdb_search_latency_seconds",
        "Search operation latency in seconds"
    ).unwrap();

    /// Commit duration in seconds (manifest flush)
    pub static ref COMMIT_DURATION_SECONDS: Histogram = register_histogram!(
        "benostreamdb_commit_duration_seconds",
        "Commit (manifest flush) duration in seconds"
    ).unwrap();

    /// Index build duration in seconds (HNSW-IVF construction)
    pub static ref INDEX_BUILD_DURATION_SECONDS: Histogram = register_histogram!(
        "benostreamdb_index_build_duration_seconds",
        "Index build (HNSW-IVF) duration in seconds"
    ).unwrap();

    /// Number of active segments (distinct from active parquet files)
    pub static ref ACTIVE_SEGMENTS_GAUGE: IntGauge = register_int_gauge!(
        "benostreamdb_active_segments",
        "Number of active segments in the table"
    ).unwrap();

    /// Number of manifest commit conflicts (concurrent writers)
    pub static ref MANIFEST_CONFLICTS_TOTAL: IntCounter = register_int_counter!(
        "benostreamdb_manifest_conflicts_total",
        "Number of manifest commit conflicts detected"
    ).unwrap();

    /// Process RSS in bytes, sampled while ingest is running.
    pub static ref INGEST_RSS_BYTES_GAUGE: IntGauge = register_int_gauge!(
        "benostreamdb_ingest_rss_bytes",
        "Resident set size of the process in bytes, sampled during ingest"
    ).unwrap();

    /// Times ingestion paused on the ingest RAM high-water mark.
    pub static ref INGEST_BACKPRESSURE_PAUSES_TOTAL: IntCounter = register_int_counter!(
        "benostreamdb_ingest_backpressure_pauses_total",
        "Number of times ingestion paused on the ingest RAM high-water mark"
    ).unwrap();

    /// Free bytes on the filesystem backing a local table, sampled at flush time.
    pub static ref FREE_DISK_BYTES_GAUGE: IntGauge = register_int_gauge!(
        "benostreamdb_free_disk_bytes",
        "Free bytes on the filesystem that will hold new segment files"
    ).unwrap();

    /// Duration of each ingest back-pressure pause, in seconds.
    pub static ref INGEST_BACKPRESSURE_PAUSE_SECONDS: Histogram = register_histogram!(
        "benostreamdb_ingest_backpressure_pause_seconds",
        "Duration of each ingest back-pressure pause on the RAM high-water mark, in seconds"
    ).unwrap();

    /// Time spent waiting for a segment index-build permit, in seconds.
    pub static ref INDEX_BUILD_GATE_WAIT_SECONDS: Histogram = register_histogram!(
        "benostreamdb_index_build_gate_wait_seconds",
        "Time spent waiting for a segment index-build permit"
    ).unwrap();
}
