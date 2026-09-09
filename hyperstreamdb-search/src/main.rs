// Copyright (c) 2026 Richard Albright. All rights reserved.

//! `hypersearch` — OpenSearch / Elasticsearch 7.10-compatible REST server.
//!
//! Binds to `HYPERSEARCH_BIND:HYPERSEARCH_PORT` (default `127.0.0.1:9200`)
//! and stores indexes under `HYPERSEARCH_STORAGE_URI`
//! (default `file://~/.hyperstreamdb/search`).

use axum::routing::{get, post, put};
use axum::Router;
use std::net::SocketAddr;
use std::sync::Arc;

use hyperstreamdb_search::handlers::{bulk, cluster, docs, indices, mapping, metrics, search};
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

    let device_str = std::env::var("HYPERSEARCH_DEVICE").unwrap_or_else(|_| "auto".to_string());
    let compute_ctx = hyperstreamdb::core::index::gpu::ComputeContext::from_device_str(&device_str)
        .unwrap_or_else(|e| {
            tracing::warn!(error = %e, device = %device_str, "Failed to initialize requested compute device; falling back to auto-detect");
            hyperstreamdb::core::index::gpu::ComputeContext::auto_detect()
        });
    hyperstreamdb::core::index::gpu::set_thread_gpu_context(Some(compute_ctx.clone()));
    tracing::info!(
        backend = compute_ctx.backend_name(),
        device_id = compute_ctx.device_id,
        gpu_accelerated = compute_ctx.is_gpu(),
        available = compute_ctx.is_available(),
        "Hardware acceleration initialized"
    );

    let state = Arc::new(AppState::with_compute(
        resolve_storage_uri(),
        cluster_uuid,
        compute_ctx,
    ));

    // Optional NRT convenience: periodically flush every index so newly
    // written documents become searchable without an explicit `_refresh`.
    if let Ok(secs) = std::env::var("HYPERSEARCH_AUTO_REFRESH_SECS") {
        if let Ok(secs) = secs.parse::<u64>() {
            if secs > 0 {
                let state = state.clone();
                tokio::spawn(async move {
                    let mut ticker = tokio::time::interval(std::time::Duration::from_secs(secs));
                    ticker.tick().await; // first tick completes immediately
                    loop {
                        ticker.tick().await;
                        let indexes = state.list_indexes().await.unwrap_or_default();
                        for index in indexes {
                            if let Err(e) =
                                hyperstreamdb_search::handlers::docs::refresh_core(&state, &index)
                                    .await
                            {
                                tracing::warn!(index, error = %e, "auto-refresh failed for index");
                            }
                        }
                    }
                });
                tracing::info!(secs, "auto-refresh enabled");
            }
        }
    }

    let app = Router::new()
        .route("/", get(cluster::cluster_info))
        .route("/_health", get(cluster::cluster_health))
        .route("/_cluster/health", get(cluster::cluster_health))
        .route("/_cluster/stats", get(cluster::cluster_stats))
        .route("/_cat/indices", get(cluster::cat_indices))
        .route("/_refresh", post(docs::refresh_all))
        .route("/_bulk", post(bulk::bulk))
        .route("/metrics", get(metrics::metrics))
        // Index CRUD.
        .route(
            "/:index",
            get(indices::get_index)
                .put(indices::create_index)
                .delete(indices::delete_index),
        )
        // Document writes.
        .route("/:index/_doc", post(docs::index_document))
        .route(
            "/:index/_doc/:id",
            post(docs::index_document_id).delete(docs::delete_document),
        )
        .route("/:index/_refresh", post(docs::refresh))
        .route("/:index/_bulk", post(bulk::bulk_indexed))
        .route("/:index/_count", get(search::count))
        .route(
            "/:index/_mapping",
            get(mapping::get_mapping).put(mapping::put_mapping),
        )
        .route(
            "/:index/_search",
            post(search::search).get(search::search_get),
        )
        .with_state(state.clone())
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            metrics::track_request,
        ))
        .layer(tower_http::trace::TraceLayer::new_for_http());

    // Qdrant-compatible API on 6333
    let qdrant_app = Router::new()
        .route(
            "/collections/:collection_name",
            get(hyperstreamdb_search::handlers::qdrant::get_collection)
                .put(hyperstreamdb_search::handlers::qdrant::create_collection)
                .delete(hyperstreamdb_search::handlers::qdrant::delete_collection),
        )
        .route(
            "/collections/:collection_name/index",
            put(hyperstreamdb_search::handlers::qdrant::create_payload_index),
        )
        .route(
            "/collections/:collection_name/points",
            put(hyperstreamdb_search::handlers::qdrant::upsert_points)
                .get(hyperstreamdb_search::handlers::qdrant::retrieve_points),
        )
        .route(
            "/collections/:collection_name/points/search",
            post(hyperstreamdb_search::handlers::qdrant::query_points),
        )
        .route(
            "/collections/:collection_name/points/delete",
            post(hyperstreamdb_search::handlers::qdrant::delete_points),
        )
        .with_state(state.clone())
        .layer(tower_http::trace::TraceLayer::new_for_http());

    let qdrant_bind = std::env::var("QDRANT_BIND").unwrap_or_else(|_| bind.clone());
    let qdrant_port: u16 = std::env::var("QDRANT_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6333);
    let qdrant_addr: SocketAddr = format!("{qdrant_bind}:{qdrant_port}")
        .parse()
        .expect("Invalid QDRANT_BIND/QDRANT_PORT");

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("Failed to bind ES API");
    tracing::info!(%addr, "listening for OpenSearch API");

    let q_listener = tokio::net::TcpListener::bind(qdrant_addr)
        .await
        .expect("Failed to bind Qdrant API");
    tracing::info!(%qdrant_addr, "listening for Qdrant API");

    tokio::spawn(async move {
        if let Err(e) = axum::serve(q_listener, qdrant_app)
            .with_graceful_shutdown(shutdown_signal())
            .await
        {
            tracing::error!("Qdrant server error: {e}");
        }
    });

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
