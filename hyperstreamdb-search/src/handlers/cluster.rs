// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Cluster-level endpoints: `GET /`, `GET /_health`, `GET /_cluster/health`,
//! `GET /_cluster/stats`, and `GET /_cat/indices`.

use axum::extract::State;
use axum::response::IntoResponse;
use axum::response::Response;
use std::sync::Arc;

use super::es_response;
use crate::es_types::{ClusterHealth, ClusterInfo, VersionInfo};
use crate::state::AppState;
use crate::TAGLINE;

/// `GET /` — cluster root info (ES 7.10 shape).
pub async fn cluster_info(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let info = ClusterInfo {
        name: "hypersearch-1".to_string(),
        cluster_name: "hypersearch".to_string(),
        cluster_uuid: state.cluster_uuid.clone(),
        version: VersionInfo::new(),
        tagline: TAGLINE.to_string(),
        compute: Some(crate::es_types::ComputeInfo {
            backend: state.compute.backend_name().to_string(),
            device_id: state.compute.device_id,
            gpu_accelerated: state.compute.is_gpu(),
            available: state.compute.is_available(),
        }),
    };
    es_response(Ok(info))
}

/// `GET /_health` and `GET /_cluster/health` — single-node health (ES 7.10 shape).
pub async fn cluster_health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let tables = state.tables.read().await;
    let shards = tables.len() as u32;
    let health = ClusterHealth {
        cluster_name: "hypersearch".to_string(),
        status: "green".to_string(),
        timed_out: false,
        number_of_nodes: 1,
        number_of_data_nodes: 1,
        active_primary_shards: shards,
        active_shards: shards,
        relocating_shards: 0,
        initializing_shards: 0,
        unassigned_shards: 0,
        delayed_unassigned_shards: 0,
        number_of_pending_tasks: 0,
        number_of_in_flight_fetch: 0,
        task_max_waiting_in_queue_millis: 0,
        active_shards_percent_as_number: 100.0,
    };
    es_response(Ok(health))
}

/// `GET /_cat/indices` — tab-separated index summary (ES 7.10 shape).
pub async fn cat_indices(State(state): State<Arc<AppState>>) -> Response {
    let mut lines =
        vec!["health\tstatus\tindex\tpri\trep\tdocs.count\tdocs.store\tstore.size".to_string()];
    let indexes = match state.list_indexes().await {
        Ok(ix) => ix,
        Err(e) => {
            tracing::warn!(error = %e, "failed to list indexes for _cat/indices");
            return (axum::http::StatusCode::OK, lines.join("\n")).into_response();
        }
    };
    for index in indexes {
        match state.open_light(&index).await {
            Ok(table) => match table.get_table_statistics_async().await {
                Ok(stats) => {
                    lines.push(format!(
                        "green\topen\t{}\t1\t0\t{}\t{}\t{}",
                        index, stats.row_count, stats.total_size_bytes, stats.total_size_bytes
                    ));
                }
                Err(e) => {
                    tracing::warn!(index, error = %e, "failed to stat index");
                }
            },
            Err(e) => {
                tracing::warn!(index, error = %e, "failed to open index for _cat/indices");
            }
        }
    }
    (axum::http::StatusCode::OK, lines.join("\n")).into_response()
}

/// `GET /_cluster/stats` — aggregate cluster statistics (ES 7.10 shape).
pub async fn cluster_stats(State(state): State<Arc<AppState>>) -> Response {
    let indexes = state.list_indexes().await.unwrap_or_default();
    let mut total_docs: u64 = 0;
    let mut total_size: u64 = 0;
    for index in &indexes {
        if let Ok(table) = state.open_light(index).await {
            if let Ok(stats) = table.get_table_statistics_async().await {
                total_docs += stats.row_count;
                total_size += stats.total_size_bytes;
            }
        }
    }
    let body = serde_json::json!({
        "cluster_name": "hypersearch",
        "status": "green",
        "compute": {
            "backend": state.compute.backend_name(),
            "device_id": state.compute.device_id,
            "gpu_accelerated": state.compute.is_gpu(),
            "available": state.compute.is_available(),
        },
        "indices": {
            "count": indexes.len(),
            "docs": { "count": total_docs, "deleted": 0 },
            "store": { "size_in_bytes": total_size },
        },
        "nodes": {
            "count": {
                "total": 1,
                "data": 1,
                "master": 1,
            }
        },
    });
    es_response(Ok(body))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cluster_info_and_compute_metadata() {
        let tmp = tempfile::tempdir().unwrap();
        let root = format!("file://{}", tmp.path().display());
        let state = Arc::new(AppState::new(root, "test-cluster".into()));

        let resp = cluster_info(State(state)).await.into_response();
        assert_eq!(resp.status(), axum::http::StatusCode::OK);

        let body_bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();

        assert_eq!(json["cluster_name"], "hypersearch");
        assert_eq!(json["tagline"], TAGLINE);
        assert!(
            json.get("compute").is_some(),
            "expected compute block in cluster_info"
        );
        let compute = &json["compute"];
        assert_eq!(compute["backend"], "cpu");
        assert_eq!(compute["gpu_accelerated"], false);
    }

    #[tokio::test]
    async fn test_cluster_health() {
        let tmp = tempfile::tempdir().unwrap();
        let root = format!("file://{}", tmp.path().display());
        let state = Arc::new(AppState::new(root, "test-cluster".into()));

        let resp = cluster_health(State(state)).await.into_response();
        assert_eq!(resp.status(), axum::http::StatusCode::OK);

        let body_bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(json["status"], "green");
        assert_eq!(json["number_of_nodes"], 1);
    }

    #[tokio::test]
    async fn test_cluster_stats_and_cat_indices() {
        let tmp = tempfile::tempdir().unwrap();
        let root = format!("file://{}", tmp.path().display());
        let state = Arc::new(AppState::new(root, "test-cluster".into()));

        let resp = cluster_stats(State(state.clone())).await.into_response();
        assert_eq!(resp.status(), axum::http::StatusCode::OK);

        let body_bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(json["cluster_name"], "hypersearch");
        assert_eq!(json["compute"]["backend"], "cpu");

        let cat_resp = cat_indices(State(state)).await.into_response();
        assert_eq!(cat_resp.status(), axum::http::StatusCode::OK);
        let cat_bytes = axum::body::to_bytes(cat_resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8(cat_bytes.to_vec()).unwrap();
        assert!(text.starts_with("health\tstatus\tindex"));
    }
}
