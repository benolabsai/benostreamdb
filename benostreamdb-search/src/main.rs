// Copyright (c) 2026 Richard Albright. All rights reserved.

//! `bsdb-search` — OpenSearch / Elasticsearch 7.10-compatible REST server.
//!
//! Binds to `BENOSEARCH_BIND:BENOSEARCH_PORT` (default `127.0.0.1:9200`)
//! and stores indexes under `BENOSEARCH_STORAGE_URI`
//! (default `file://~/.benostreamdb/search`).

// No-panic policy for production binaries (see NO_PANIC_POLICY.md).
#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use axum::Router;
use std::net::SocketAddr;
use std::sync::Arc;

use benostreamdb_search::handlers::{
    bulk, cluster, docs, graph_search, indices, mapping, metrics, search,
};
use benostreamdb_search::state::{resolve_catalog, resolve_storage_uri, AppState};

// Heap profiling is opt-in (`--features dhat-heap`). As a global allocator,
// `dhat::Alloc` conflicts with the jemalloc global allocator that the
// `benostreamdb` library installs on Linux — having both in one binary fails
// to link, which previously made this server unbuildable on the default feature
// set.
#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

#[tokio::main]
async fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();
    let num_threads = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            std::cmp::max(
                1,
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
                    / 2,
            )
        });
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap_or_else(|e| tracing::warn!("Failed to initialize rayon thread pool: {}", e));
    // Structured-logging-only panic hook (no println in this crate).
    std::panic::set_hook(Box::new(|info| {
        tracing::error!(panic = ?info, "bsdb-search panicked");
    }));

    let _telemetry_guard = match benostreamdb::telemetry::tracing::init_tracing("bsdb-search") {
        Ok(guard) => guard,
        Err(e) => {
            tracing::error!(error = %e, "Failed to initialize tracing");
            std::process::exit(1);
        }
    };

    let bind = std::env::var("BENOSEARCH_BIND").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port: u16 = std::env::var("BENOSEARCH_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(9200);
    let addr: SocketAddr = match format!("{bind}:{port}").parse() {
        Ok(a) => a,
        Err(e) => {
            tracing::error!(
                error = %e,
                bind = %bind,
                port,
                "Invalid BENOSEARCH_BIND/BENOSEARCH_PORT"
            );
            std::process::exit(1);
        }
    };

    let cluster_uuid = uuid::Uuid::new_v4().to_string();

    let device_str = std::env::var("BENOSEARCH_DEVICE").unwrap_or_else(|_| "auto".to_string());
    let compute_ctx = benostreamdb::core::index::gpu::ComputeContext::from_device_str(&device_str)
        .unwrap_or_else(|e| {
            tracing::warn!(error = %e, device = %device_str, "Failed to initialize requested compute device; falling back to auto-detect");
            benostreamdb::core::index::gpu::ComputeContext::auto_detect()
        });
    benostreamdb::core::index::gpu::set_thread_gpu_context(Some(compute_ctx.clone()));
    tracing::info!(
        backend = compute_ctx.backend_name(),
        device_id = compute_ctx.device_id,
        gpu_accelerated = compute_ctx.is_gpu(),
        available = compute_ctx.is_available(),
        "Hardware acceleration initialized"
    );

    let (catalog_opt, catalog_ns) = resolve_catalog()
        .await
        .map(|(cat, ns)| (Some(cat), ns))
        .unwrap_or((None, "default".to_string()));

    let state = Arc::new(AppState::with_catalog(
        resolve_storage_uri(),
        cluster_uuid,
        compute_ctx,
        catalog_opt,
        catalog_ns,
    ));

    // Optional NRT convenience: periodically flush every index so newly
    // written documents become searchable without an explicit `_refresh`.
    if let Ok(secs) = std::env::var("BENOSEARCH_AUTO_REFRESH_SECS") {
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
                                benostreamdb_search::handlers::docs::refresh_core(&state, &index)
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
        .route("/:index/_graph_search", post(graph_search::graph_search))
        .with_state(state.clone())
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            metrics::track_request,
        ))
        .layer(tower_http::trace::TraceLayer::new_for_http())
        .layer(DefaultBodyLimit::max(100 * 1024 * 1024));

    // Qdrant-compatible API on 6333
    let qdrant_app = benostreamdb_search::handlers::qdrant::router(state.clone())
        .layer(tower_http::trace::TraceLayer::new_for_http())
        .layer(DefaultBodyLimit::max(100 * 1024 * 1024));

    let qdrant_bind = std::env::var("QDRANT_BIND").unwrap_or_else(|_| bind.clone());
    let qdrant_port: u16 = std::env::var("QDRANT_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6333);
    let qdrant_addr: SocketAddr = match format!("{qdrant_bind}:{qdrant_port}").parse() {
        Ok(a) => a,
        Err(e) => {
            tracing::error!(error = %e, "Invalid QDRANT_BIND/QDRANT_PORT");
            std::process::exit(1);
        }
    };

    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(%addr, error = %e, "Failed to bind ES API");
            std::process::exit(1);
        }
    };
    tracing::info!(%addr, "listening for OpenSearch API");

    let q_listener = match tokio::net::TcpListener::bind(qdrant_addr).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(%qdrant_addr, error = %e, "Failed to bind Qdrant API");
            std::process::exit(1);
        }
    };
    tracing::info!(%qdrant_addr, "listening for Qdrant API");

    tokio::spawn(async move {
        if let Err(e) = axum::serve(q_listener, qdrant_app)
            .with_graceful_shutdown(shutdown_signal())
            .await
        {
            tracing::error!("Qdrant server error: {e}");
        }
    });

    if let Err(e) = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
    {
        tracing::error!(error = %e, "bsdb-search server error");
        std::process::exit(1);
    }
}

async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(e) = tokio::signal::ctrl_c().await {
            tracing::error!(error = %e, "failed to install Ctrl+C handler");
            // Never resolve: shutdown still works via the other signal.
            std::future::pending::<()>().await;
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sigterm) => {
                sigterm.recv().await;
            }
            Err(e) => {
                tracing::error!(error = %e, "failed to install SIGTERM handler");
                std::future::pending::<()>().await;
            }
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    tracing::info!("Shutdown signal received, starting graceful shutdown");
}
