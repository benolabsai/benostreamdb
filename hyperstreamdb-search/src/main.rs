// Copyright (c) 2026 Richard Albright. All rights reserved.

//! `hypersearch` — OpenSearch / Elasticsearch 7.10-compatible REST server.
//!
//! Binds to `HYPERSEARCH_BIND:HYPERSEARCH_PORT` (default `127.0.0.1:9200`)
//! and stores indexes under `HYPERSEARCH_STORAGE_URI`
//! (default `file://~/.hyperstreamdb/search`).

use axum::routing::get;
use axum::Router;
use std::net::SocketAddr;
use std::sync::Arc;

use hyperstreamdb_search::handlers::{cluster, metrics};
use hyperstreamdb_search::state::{resolve_storage_uri, AppState};

#[tokio::main]
async fn main() {
    // Structured-logging-only panic hook (no println in this crate).
    std::panic::set_hook(Box::new(|info| {
        tracing::error!(panic = ?info, "hypersearch panicked");
    }));

    let _telemetry_guard = hyperstreamdb::telemetry::tracing::init_tracing("hypersearch")
        .expect("Failed to initialize tracing");

    let bind = std::env::var("HYPERSEARCH_BIND").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port: u16 = std::env::var("HYPERSEARCH_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(9200);
    let addr: SocketAddr = format!("{bind}:{port}")
        .parse()
        .expect("Invalid HYPERSEARCH_BIND/HYPERSEARCH_PORT");

    let cluster_uuid = uuid::Uuid::new_v4().to_string();
    let state = Arc::new(AppState::new(resolve_storage_uri(), cluster_uuid));

    let app = Router::new()
        .route("/", get(cluster::cluster_info))
        .route("/_health", get(cluster::cluster_health))
        .route("/_cluster/health", get(cluster::cluster_health))
        .route("/metrics", get(metrics::metrics))
        .with_state(state)
        .layer(tower_http::trace::TraceLayer::new_for_http());

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("Failed to bind");
    tracing::info!(%addr, "listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("Server error");
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler");
        sigterm.recv().await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    tracing::info!("Shutdown signal received, starting graceful shutdown");
}
