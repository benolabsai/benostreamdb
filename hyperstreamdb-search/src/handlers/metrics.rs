// Copyright (c) 2026 Richard Albright. All rights reserved.

//! `GET /metrics` — Prometheus text format (plan 5.2.2 telemetry) and the
//! router-level request-tracking middleware that feeds the collectors.

use std::sync::Arc;
use std::time::Instant;

use axum::body::Body;
use axum::extract::{Request, State};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use crate::state::AppState;

/// `GET /metrics` — Prometheus text format (version 0.0.4).
///
/// Exposes the plan-5.2.2 telemetry: the query-latency histogram
/// (`hypersearch_http_request_duration_seconds` with `route="search"`),
/// request counters, ingestion counters, index table-cache hit/miss
/// counters, and the active-requests gauge.
pub async fn metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Response::builder()
        .status(200)
        .header("Content-Type", "text/plain; version=0.0.4")
        .body(Body::from(state.metrics.gather_text()))
        .unwrap()
}

/// Coarse route class for metric labels. Index names are never a label
/// (unbounded cardinality); this keeps the label set fixed at a handful
/// of values.
fn route_class(path: &str) -> &'static str {
    match path {
        "/" => "root",
        "/metrics" => "metrics",
        "/_health" | "/_cluster/health" | "/_cluster/stats" | "/_cat/indices" => "cluster",
        _ if path.ends_with("/_search") => "search",
        _ if path.contains("/_doc") => "doc",
        _ if path.ends_with("/_bulk") => "bulk",
        _ if path.ends_with("/_count") => "count",
        _ if path.ends_with("/_mapping") => "mapping",
        _ if path.ends_with("/_refresh") => "refresh",
        _ => "index",
    }
}

/// Router-level middleware: gauge in-flight requests, count requests by
/// method + route class, and observe per-route latency.
pub async fn track_request(
    State(state): State<Arc<AppState>>,
    req: Request,
    next: Next,
) -> Response {
    let class = route_class(req.uri().path());
    let method = req.method().as_str().to_ascii_lowercase();
    let m = &state.metrics;

    m.active_requests.inc();
    let started = Instant::now();
    let res = next.run(req).await;
    m.active_requests.dec();

    m.http_requests_total
        .with_label_values(&[method.as_str(), class])
        .inc();
    m.http_request_duration_seconds
        .with_label_values(&[method.as_str(), class])
        .observe(started.elapsed().as_secs_f64());
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Metrics;

    #[test]
    fn route_class_buckets_all_registered_routes() {
        assert_eq!(route_class("/"), "root");
        assert_eq!(route_class("/metrics"), "metrics");
        assert_eq!(route_class("/_health"), "cluster");
        assert_eq!(route_class("/_cluster/health"), "cluster");
        assert_eq!(route_class("/_cluster/stats"), "cluster");
        assert_eq!(route_class("/_cat/indices"), "cluster");
        assert_eq!(route_class("/_bulk"), "bulk");
        assert_eq!(route_class("/idx/_bulk"), "bulk");
        assert_eq!(route_class("/idx/_search"), "search");
        assert_eq!(route_class("/idx/_count"), "count");
        assert_eq!(route_class("/idx/_mapping"), "mapping");
        assert_eq!(route_class("/idx/_doc"), "doc");
        assert_eq!(route_class("/idx/_doc/abc"), "doc");
        assert_eq!(route_class("/idx/_refresh"), "refresh");
        assert_eq!(route_class("/idx"), "index");
    }

    #[test]
    fn gather_text_exposes_every_planned_collector_family() {
        let m = Metrics::new();
        m.active_requests.inc();
        m.http_requests_total
            .with_label_values(&["post", "search"])
            .inc();
        m.http_request_duration_seconds
            .with_label_values(&["post", "search"])
            .observe(0.05);
        m.docs_indexed_total.with_label_values(&["created"]).inc();
        m.index_cache_hits_total.inc();
        m.index_cache_misses_total.inc();

        let text = m.gather_text();
        for needle in [
            "hypersearch_active_requests 1",
            "hypersearch_http_requests_total{method=\"post\",route=\"search\"} 1",
            "hypersearch_http_request_duration_seconds_bucket",
            "hypersearch_http_request_duration_seconds_sum{method=\"post\",route=\"search\"} 0.05",
            "hypersearch_docs_indexed_total{result=\"created\"} 1",
            "hypersearch_index_cache_hits_total 1",
            "hypersearch_index_cache_misses_total 1",
        ] {
            assert!(text.contains(needle), "missing {needle:?} in:\n{text}");
        }
    }
}
