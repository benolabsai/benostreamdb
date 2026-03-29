// Copyright (c) 2026 Richard Albright. All rights reserved.

use axum::{
    routing::{get, post},
    Router, Json,
    http::StatusCode,
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use hyperstreamdb::SegmentConfig;

#[tokio::main]
async fn main() {
    // initialize tracing
    tracing_subscriber::fmt::init();

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/query", post(query_handler))
        .route("/ingest", post(ingest_handler));

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn health_check() -> &'static str {
    "OK"
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

use hyperstreamdb::core::reader::HybridReader;
use hyperstreamdb::core::segment::HybridSegmentWriter;
// use object_store::local::LocalFileSystem;
use std::sync::Arc;

async fn query_handler(Json(payload): Json<QueryRequest>) -> impl IntoResponse {
    println!("Received query: filter='{}'", payload.filter);
    
    // Demonstrate the "Index-First" Read
    // Use factory to support s3://, az://, etc.
    // Ideally this comes from payload or config. defaulting to /tmp for local PoC
    let uri = std::env::var("HYPERSTREAM_STORAGE_URI").unwrap_or_else(|_| "file:///tmp".to_string());
    println!("Connecting to storage: {}", uri);
    let store = hyperstreamdb::core::storage::create_object_store(&uri).expect("Failed to create object store");
    
    // Config: path is relative to the store prefix now
    let config = SegmentConfig::new("", "segment_001");
    let reader = HybridReader::new(config, store, &uri);
    
    let filter = hyperstreamdb::core::planner::QueryFilter::parse(&payload.filter).unwrap();
    // Gateway queries all columns by default (None = no projection)
    match reader.query_index_first(&filter, None::<std::sync::Arc<Schema>>).await {
        Ok(batches) => {
             let total_rows: usize = batches.iter().map(|b: &arrow::record_batch::RecordBatch| b.num_rows()).sum();
             println!("Query successful. Read {} rows from Parquet.", total_rows);
             let response = QueryResponse {
                rows: vec![format!("{} rows matching index", total_rows)]
             };
             (StatusCode::OK, Json(response))
        },
        Err(e) => {
            eprintln!("Query failed: {}", e);
            let response = QueryResponse {
                rows: vec![format!("Error: {}", e)]
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
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(id_array)],
    ).unwrap();

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
