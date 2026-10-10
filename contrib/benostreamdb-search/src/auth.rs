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
    mut req: Request,
    next: Next,
) -> Response {
    if is_public_path(&cfg, req.method(), req.uri().path()) {
        return next.run(req).await;
    }

    if !cfg.enabled() {
        if cfg.required() {
            return (StatusCode::UNAUTHORIZED, "authentication required").into_response();
        }
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
            Ok(claims) => {
                if is_write_request(req.method(), req.uri().path())
                    && !claims.roles.iter().any(|r| r == "admin")
                {
                    return (StatusCode::FORBIDDEN, "admin role required for writes")
                        .into_response();
                }
                req.extensions_mut().insert(claims);
                next.run(req).await
            }
            Err(e) => {
                tracing::warn!(error = %e, "authentication failed");
                (StatusCode::UNAUTHORIZED, "invalid credentials").into_response()
            }
        },
        None => (StatusCode::UNAUTHORIZED, "missing bearer token").into_response(),
    }
}

/// Whether a request mutates state (and therefore requires the `admin` role).
///
/// `POST` is a write *except* for the read-shaped search endpoints, which both
/// the Elasticsearch and Qdrant APIs express as POSTs.
fn is_write_request(method: &Method, path: &str) -> bool {
    match method.as_str() {
        "PUT" | "DELETE" | "PATCH" => true,
        "POST" => {
            !(path.ends_with("_search")
                || path.ends_with("/search")
                || path.ends_with("/scroll")
                || path.ends_with("_count")
                || path.ends_with("/count")
                || path.ends_with("/_msearch")
                || path.ends_with("/_mget"))
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_paths_are_public() {
        let cfg = AuthConfig::default();
        for p in ["/_health", "/_cluster/health", "/_cat/health", "/healthz", "/readyz", "/livez"] {
            assert!(is_public_path(&cfg, &Method::GET, p), "{p} should be public");
        }
        assert!(!is_public_path(&cfg, &Method::GET, "/my-index/_search"));
        assert!(!is_public_path(&cfg, &Method::POST, "/my-index/_doc"));
    }

    #[test]
    fn metrics_public_only_when_configured() {
        let private = AuthConfig::default();
        assert!(!is_public_path(&private, &Method::GET, "/metrics"));
        let public = AuthConfig::default().with_metrics_public(true);
        assert!(is_public_path(&public, &Method::GET, "/metrics"));
    }

    #[test]
    fn write_detection_matches_api_semantics() {
        // Mutations.
        assert!(is_write_request(&Method::PUT, "/idx/_doc/1"));
        assert!(is_write_request(&Method::DELETE, "/idx"));
        assert!(is_write_request(&Method::PATCH, "/idx/_doc/1"));
        assert!(is_write_request(&Method::POST, "/idx/_doc"));
        assert!(is_write_request(&Method::POST, "/collections/points"));
        // Read-shaped POSTs.
        assert!(!is_write_request(&Method::POST, "/idx/_search"));
        assert!(!is_write_request(&Method::POST, "/collections/points/search"));
        assert!(!is_write_request(&Method::POST, "/idx/_msearch"));
        assert!(!is_write_request(&Method::POST, "/idx/_count"));
        // Reads.
        assert!(!is_write_request(&Method::GET, "/idx/_search"));
    }
}
