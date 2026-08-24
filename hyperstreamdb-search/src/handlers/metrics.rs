// Copyright (c) 2026 Richard Albright. All rights reserved.

//! `GET /metrics` — Prometheus text format from the global registry.

use axum::http::StatusCode;
use axum::response::IntoResponse;
use prometheus::{gather, TextEncoder};

pub async fn metrics() -> impl IntoResponse {
    let encoder = TextEncoder::new();
    let metric_families = gather();
    let mut result = String::new();
    encoder
        .encode_utf8(&metric_families, &mut result)
        .unwrap_or_default();
    axum::response::Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "text/plain; version=0.0.4")
        .body(axum::body::Body::from(result))
        .unwrap()
}
