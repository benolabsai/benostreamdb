// Copyright (c) 2026 Richard Albright. All rights reserved.

// No-panic policy for production binaries (see NO_PANIC_POLICY.md).
#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

use axum::{
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use benostreamdb::SegmentConfig;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::time::SystemTime;

#[tokio::main]
async fn main() {
    // Task 8: Global panic handler
    std::panic::set_hook(Box::new(|info| {
        tracing::error!(panic = ?info, "Gateway panicked");
    }));

    // Task 3: Use proper telemetry init
    let _telemetry_guard = match benostreamdb::telemetry::tracing::init_tracing("gateway") {
        Ok(guard) => guard,
        Err(e) => {
            eprintln!("Failed to initialize tracing: {e}");
            std::process::exit(1);
        }
    };

    // Task 5: Track start time for uptime
    let start_time = SystemTime::now();

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/metrics", get(metrics_handler))
        .route("/query", post(query_handler))
        .route("/ingest", post(ingest_handler))
        .layer(tower_http::trace::TraceLayer::new_for_http());

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    tracing::info!(%addr, "listening");
    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(%addr, error = %e, "failed to bind gateway listener");
            std::process::exit(1);
        }
    };

    let app_with_state = app.with_state(StartState { start_time });
    let server = axum::serve(listener, app_with_state);

    // Task 4: Graceful shutdown on SIGINT/SIGTERM
    let graceful = server.with_graceful_shutdown(shutdown_signal());
    if let Err(e) = graceful.await {
        tracing::error!(error = %e, "gateway server error");
        std::process::exit(1);
    }
}

struct StartState {
    start_time: SystemTime,
}

impl Clone for StartState {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for StartState {}

async fn health_check(state: axum::extract::State<StartState>) -> impl IntoResponse {
    let uptime = state.start_time.elapsed().unwrap_or_default().as_secs();

    Json(serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_seconds": uptime
    }))
}

async fn metrics_handler() -> impl IntoResponse {
    use prometheus::{gather, TextEncoder};
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
        .unwrap_or_else(|e| {
            tracing::error!(error = %e, "failed to build metrics response");
            axum::response::Response::new(axum::body::Body::empty())
        })
}

async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(e) = tokio::signal::ctrl_c().await {
            tracing::error!(error = %e, "failed to install Ctrl+C handler");
        }
    };

    #[cfg(unix)]
    let terminate = async {
        let mut sigterm =
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!(error = %e, "failed to install SIGTERM handler");
                    return;
                }
            };
        sigterm.recv().await;
        let mut sigint =
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt()) {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!(error = %e, "failed to install SIGINT handler");
                    return;
                }
            };
        sigint.recv().await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    tracing::info!("Shutdown signal received, starting graceful shutdown");
}

#[derive(Deserialize)]
struct QueryRequest {
    filter: String,
    #[allow(dead_code)]
    vector: Option<Vec<f32>>,
}

#[derive(Serialize)]
struct QueryResponse {
    rows: Vec<String>, // Mock result
}

use benostreamdb::core::reader::HybridReader;
use benostreamdb::core::segment::HybridSegmentWriter;
// use object_store::local::LocalFileSystem;
use std::sync::Arc;

async fn query_handler(Json(payload): Json<QueryRequest>) -> impl IntoResponse {
    println!("Received query: filter='{}'", payload.filter);

    // Demonstrate the "Index-First" Read
    // Use factory to support s3://, az://, etc.
    // Ideally this comes from payload or config. defaulting to /tmp for local PoC
    let uri = std::env::var("BENOSTREAM_STORAGE_URI").unwrap_or_else(|_| "file:///tmp".to_string());
    println!("Connecting to storage: {}", uri);
    let store = match benostreamdb::core::storage::create_object_store(&uri) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!(error = %e, %uri, "failed to create object store");
            let response = QueryResponse {
                rows: vec![format!("Error: {e}")],
            };
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(response));
        }
    };

    // Config: path is relative to the store prefix now
    let config = SegmentConfig::new("", "segment_001");
    let reader = HybridReader::new(config, store, &uri);

    // User-supplied: a malformed filter must be a 400, not a panic.
    let filter = match benostreamdb::core::planner::QueryFilter::parse(&payload.filter) {
        Some(f) => f,
        None => {
            let response = QueryResponse {
                rows: vec![format!("Invalid filter: {}", payload.filter)],
            };
            return (StatusCode::BAD_REQUEST, Json(response));
        }
    };
    // Gateway queries all columns by default (None = no projection)
    match reader
        .query_index_first(&filter, None::<std::sync::Arc<Schema>>)
        .await
    {
        Ok(batches) => {
            let total_rows: usize = batches
                .iter()
                .map(|b: &arrow::record_batch::RecordBatch| b.num_rows())
                .sum();
            println!("Query successful. Read {} rows from Parquet.", total_rows);
            let response = QueryResponse {
                rows: vec![format!("{} rows matching index", total_rows)],
            };
            (StatusCode::OK, Json(response))
        }
        Err(e) => {
            eprintln!("Query failed: {}", e);
            let response = QueryResponse {
                rows: vec![format!("Error: {}", e)],
            };
            (StatusCode::INTERNAL_SERVER_ERROR, Json(response))
        }
    }
}

#[derive(Deserialize)]
struct IngestRequest {
    #[allow(dead_code)]
    data: serde_json::Value,
}

use arrow::array::Int32Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

async fn ingest_handler(Json(_payload): Json<IngestRequest>) -> impl IntoResponse {
    println!("Received ingest data request");

    // 1. Create Mock Data (Arrow Batch) for PoC
    let id_array = Int32Array::from(vec![1, 2, 3, 4, 5]);
    let schema = Schema::new(vec![Field::new("id", DataType::Int32, false)]);
    let batch = match RecordBatch::try_new(Arc::new(schema), vec![Arc::new(id_array)]) {
        Ok(b) => b,
        Err(e) => {
            tracing::error!(error = %e, "failed to build mock ingest batch");
            return (StatusCode::INTERNAL_SERVER_ERROR, "Failed to build batch");
        }
    };

    // 2. Configure Writer
    // In a real app, base_path would be S3 bucket or config
    let config = SegmentConfig::new("/tmp", "segment_001");
    let writer = HybridSegmentWriter::new(config);

    // 3. Write Data & Index
    match writer.write_batch(&batch) {
        Ok(_) => (StatusCode::CREATED, "Ingested and Indexed"),
        Err(e) => {
            eprintln!("Error writing segment: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, "Failed to write")
        }
    }
}
