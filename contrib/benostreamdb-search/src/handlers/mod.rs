// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Axum handlers for the ES-compatible API.

pub mod aggs;
pub mod bulk;
pub mod cluster;
pub mod docs;
pub mod graph_search;
pub mod indices;
pub mod mapping;
pub mod metrics;
pub mod qdrant;
pub mod query_string;
pub mod search;
pub mod snapshots;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use benostreamdb::BenoStreamError;
use serde::Serialize;

use crate::es_types::EsError;

/// Convert a `Result<T, BenoStreamError>` into an HTTP response, mapping
/// errors to the ES-style JSON envelope (`{"error": {...}, "status": N}`).
pub(crate) fn es_response<T: Serialize>(result: Result<T, BenoStreamError>) -> Response {
    es_response_with_status(StatusCode::OK, result)
}

/// Like [`es_response`], but lets the caller choose the success status
/// (e.g. 201 for document creation). Errors always use their own status.
pub(crate) fn es_response_with_status<T: Serialize>(
    status: StatusCode,
    result: Result<T, BenoStreamError>,
) -> Response {
    let mut resp = match result {
        Ok(value) => (status, Json(value)).into_response(),
        Err(err) => {
            tracing::error!(%err, "request failed");
            let es = EsError::from(err);
            let status =
                StatusCode::from_u16(es.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            (status, Json(es)).into_response()
        }
    };
    resp.headers_mut().insert(
        axum::http::HeaderName::from_static("x-elastic-product"),
        axum::http::HeaderValue::from_static("Elasticsearch"),
    );
    resp.headers_mut().insert(
        axum::http::HeaderName::from_static("x-opensearch-version"),
        axum::http::HeaderValue::from_static("3.0.0"),
    );
    resp
}
