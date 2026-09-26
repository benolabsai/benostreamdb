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

/// Latency histogram buckets, in seconds, tuned for database workloads.
///
/// The Prometheus defaults are too coarse to resolve sub-millisecond vector
/// search, so the low end starts at 100µs and the high end reaches 10s to
/// cover compaction and large manifest commits.
fn latency_buckets() -> Vec<f64> {
    vec![
        0.0001, 0.00025, 0.0005, // 100µs–500µs: in-memory vector search
        0.001, 0.0025, 0.005, // 1ms–5ms
        0.01, 0.025, 0.05, // 10ms–50ms
        0.1, 0.25, 0.5, // 100ms–500ms
        1.0, 2.5, 5.0, 10.0, // 1s–10s: compaction / large commits
    ]
}

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
        "Query latency in seconds",
        latency_buckets()
    )
    .unwrap();

    /// Compaction duration in seconds
    pub static ref COMPACTION_DURATION_SECONDS: Histogram = register_histogram!(
        "benostreamdb_compaction_duration_seconds",
        "Compaction duration in seconds",
        latency_buckets()
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
        "Search operation latency in seconds",
        latency_buckets()
    ).unwrap();

    /// Commit duration in seconds (manifest flush)
    pub static ref COMMIT_DURATION_SECONDS: Histogram = register_histogram!(
        "benostreamdb_commit_duration_seconds",
        "Commit (manifest flush) duration in seconds",
        latency_buckets()
    ).unwrap();

    /// Index build duration in seconds (HNSW-IVF construction)
    pub static ref INDEX_BUILD_DURATION_SECONDS: Histogram = register_histogram!(
        "benostreamdb_index_build_duration_seconds",
        "Index build (HNSW-IVF) duration in seconds",
        latency_buckets()
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
        "Duration of each ingest back-pressure pause on the RAM high-water mark, in seconds",
        latency_buckets()
    ).unwrap();

    /// Time spent waiting for a segment index-build permit, in seconds.
    pub static ref INDEX_BUILD_GATE_WAIT_SECONDS: Histogram = register_histogram!(
        "benostreamdb_index_build_gate_wait_seconds",
        "Time spent waiting for a segment index-build permit",
        latency_buckets()
    ).unwrap();

    // ------------------------------------------------------------------
    // Merge-on-read delete-path instrumentation (temporary; see
    // plans/production_readiness_plan.md). These break the 8-28s step
    // latency down by phase so the dominant contributor is measurable.
    // ------------------------------------------------------------------

    /// Wall time of each phase of `load_merged_deletes_inner`, in seconds.
    /// Label `phase` is one of: `cache_key`, `merged_cache_lookup`,
    /// `position_join_all`, `position_convert`, `position_merge`,
    /// `deletion_vector`, `equality`, `merged_cache_insert`, `total`.
    pub static ref MERGED_DELETES_PHASE_SECONDS: prometheus::HistogramVec = prometheus::register_histogram_vec!(
        "benostreamdb_merged_deletes_phase_seconds",
        "Wall time of each phase of load_merged_deletes_inner, in seconds",
        &["phase"],
        latency_buckets()
    ).unwrap();

    /// Outcome of the merged-deletes cache lookup. Label `result` is
    /// `hit` or `miss`.
    pub static ref MERGED_DELETES_CACHE_TOTAL: IntCounterVec = register_int_counter_vec!(
        "benostreamdb_merged_deletes_cache_total",
        "Merged-deletes cache lookups by outcome",
        &["result"]
    ).unwrap();

    /// Outcome of the per-file parsed-delete cache lookup. Label `result`
    /// is `hit` or `miss`.
    pub static ref DELETE_FILE_CACHE_TOTAL: IntCounterVec = register_int_counter_vec!(
        "benostreamdb_delete_file_cache_total",
        "Per-file parsed-delete cache lookups by outcome",
        &["result"]
    ).unwrap();

    /// Number of delete files merged in a single `load_merged_deletes_inner`
    /// call, by content kind. Label `kind` is `position`, `deletion_vector`,
    /// or `equality`.
    pub static ref MERGED_DELETES_FILES_TOTAL: IntCounterVec = register_int_counter_vec!(
        "benostreamdb_merged_deletes_files_total",
        "Delete files merged by content kind",
        &["kind"]
    ).unwrap();

    /// Number of times `load_merged_deletes` was invoked (per reader).
    pub static ref MERGED_DELETES_CALLS_TOTAL: IntCounter = register_int_counter!(
        "benostreamdb_merged_deletes_calls_total",
        "Number of load_merged_deletes invocations"
    ).unwrap();

    // ------------------------------------------------------------------
    // Parquet read-path instrumentation (temporary; see
    // plans/production_readiness_plan.md). Breaks `stream_row_groups` setup
    // down by phase so the dominant contributor is measurable.
    // ------------------------------------------------------------------

    /// Wall time of each phase of `stream_row_groups`, in seconds. Label
    /// `phase` is one of: `meta`, `projection`, `deletes`, `build`, `total`.
    pub static ref READ_PHASE_SECONDS: prometheus::HistogramVec = prometheus::register_histogram_vec!(
        "benostreamdb_read_phase_seconds",
        "Wall time of each phase of stream_row_groups, in seconds",
        &["phase"],
        latency_buckets()
    ).unwrap();

    /// Outcome of the parquet-metadata cache lookup. Label `result` is
    /// `hit` or `miss`.
    pub static ref PARQUET_META_CACHE_TOTAL: IntCounterVec = register_int_counter_vec!(
        "benostreamdb_parquet_meta_cache_total",
        "Parquet-metadata cache lookups by outcome",
        &["result"]
    ).unwrap();
}

/// Render the merge-on-read delete-path instrumentation as a human-readable
/// per-phase breakdown. Used by the baseline workload to print where the
/// step latency actually goes. Returns an empty string if nothing has been
/// recorded yet.
pub fn dump_merged_deletes_metrics() -> String {
    let mut out = String::new();
    for mf in prometheus::gather() {
        let name = mf.get_name();
        if !name.starts_with("benostreamdb_merged_deletes")
            && !name.starts_with("benostreamdb_delete_file_cache")
        {
            continue;
        }
        for m in mf.get_metric() {
            let labels: Vec<String> = m
                .get_label()
                .iter()
                .map(|l| format!("{}={}", l.get_name(), l.get_value()))
                .collect();
            let label_str = if labels.is_empty() {
                String::new()
            } else {
                format!("{{{}}}", labels.join(","))
            };
            if m.has_histogram() {
                let h = m.get_histogram();
                out.push_str(&format!(
                    "{}{} count={} sum={:.6}s\n",
                    name,
                    label_str,
                    h.get_sample_count(),
                    h.get_sample_sum()
                ));
            } else if m.has_counter() {
                out.push_str(&format!(
                    "{}{} {}\n",
                    name,
                    label_str,
                    m.get_counter().get_value()
                ));
            }
        }
    }
    out
}

/// Render the parquet read-path instrumentation as a human-readable per-phase
/// breakdown. Returns an empty string if nothing has been recorded yet.
pub fn dump_read_metrics() -> String {
    let mut out = String::new();
    for mf in prometheus::gather() {
        let name = mf.get_name();
        if !name.starts_with("benostreamdb_read_phase")
            && !name.starts_with("benostreamdb_parquet_meta_cache")
        {
            continue;
        }
        for m in mf.get_metric() {
            let labels: Vec<String> = m
                .get_label()
                .iter()
                .map(|l| format!("{}={}", l.get_name(), l.get_value()))
                .collect();
            let label_str = if labels.is_empty() {
                String::new()
            } else {
                format!("{{{}}}", labels.join(","))
            };
            if m.has_histogram() {
                let h = m.get_histogram();
                out.push_str(&format!(
                    "{}{} count={} sum={:.6}s\n",
                    name,
                    label_str,
                    h.get_sample_count(),
                    h.get_sample_sum()
                ));
            } else if m.has_counter() {
                out.push_str(&format!(
                    "{}{} {}\n",
                    name,
                    label_str,
                    m.get_counter().get_value()
                ));
            }
        }
    }
    out
}
