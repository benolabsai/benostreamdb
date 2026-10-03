// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Axum middleware wrapping the shared [`benostreamdb::core::auth::AuthConfig`].
//!
//! The verification logic (API key + HS256/RS256 JWT) lives in the core crate
//! so the search gateway and the Flight SQL server share one implementation.
//! This module only adapts it to axum and decides which paths are public.

pub use benostreamdb::core::auth::AuthConfig;

use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::{header, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

/// Paths that never require auth (health checks, cluster info).
fn is_public_path(cfg: &AuthConfig, method: &Method, path: &str) -> bool {
    // OpenSearch/ES health + cluster info.
    if path == "/_health" || path == "/_cluster/health" || path == "/_cat/health" {
        return true;
    }
    // Qdrant health probes.
    if path == "/healthz" || path == "/readyz" || path == "/livez" {
        return true;
    }
    if method == Method::GET && path == "/" {
        return true;
    }
    if cfg.metrics_public() && path == "/metrics" {
        return true;
    }
    false
}

/// Axum middleware enforcing [`AuthConfig`].
pub async fn auth_middleware(
    State(cfg): State<Arc<AuthConfig>>,
    req: Request,
    next: Next,
) -> Response {
    if !cfg.enabled() {
        if cfg.required() {
            return (StatusCode::UNAUTHORIZED, "authentication required").into_response();
        }
        return next.run(req).await;
    }

    if is_public_path(&cfg, req.method(), req.uri().path()) {
        return next.run(req).await;
    }

    // Accept the token from any of the conventions the two APIs use:
    //   * `Authorization: Bearer <jwt>`  (OpenSearch/ES, generic)
    //   * `Authorization: ApiKey <key>`  (Elasticsearch)
    //   * `api-key: <key>`               (Qdrant)
    let headers = req.headers();
    let token = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| {
            v.strip_prefix("Bearer ")
                .or_else(|| v.strip_prefix("ApiKey "))
                .or_else(|| v.strip_prefix("Api-Key "))
        })
        .or_else(|| headers.get("api-key").and_then(|v| v.to_str().ok()))
        .map(|s| s.trim().to_string());

    match token {
        Some(t) => match cfg.verify(&t) {
            Ok(_subject) => next.run(req).await,
            Err(e) => {
                tracing::warn!(error = %e, "authentication failed");
                (StatusCode::UNAUTHORIZED, "invalid credentials").into_response()
            }
        },
        None => (StatusCode::UNAUTHORIZED, "missing bearer token").into_response(),
    }
}
