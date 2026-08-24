// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Axum handlers for the ES-compatible API.

pub mod cluster;
pub mod metrics;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use hyperstreamdb::HyperstreamError;
use serde::Serialize;

use crate::es_types::EsError;

/// Convert a `Result<T, HyperstreamError>` into an HTTP response, mapping
/// errors to the ES-style JSON envelope (`{"error": {...}, "status": N}`).
pub(crate) fn es_response<T: Serialize>(result: Result<T, HyperstreamError>) -> Response {
    match result {
        Ok(value) => Json(value).into_response(),
        Err(err) => {
            tracing::error!(%err, "request failed");
            let es = EsError::from(err);
            let status =
                StatusCode::from_u16(es.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            (status, Json(es)).into_response()
        }
    }
}
