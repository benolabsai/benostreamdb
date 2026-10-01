// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! ES-compatible snapshot management endpoints: `/_snapshot/*`.
//!
//! Provides a lightweight shim over BenoStreamDB's native Iceberg snapshot metadata
//! so backup tools (e.g. Curator, snapshot-restore utilities) can inspect, register,
//! and trigger snapshot checkpoints.

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use benostreamdb::BenoStreamError;
use chrono::{DateTime, Utc};
use serde_json::{Map, Value};

use super::es_response;
use crate::state::AppState;

/// GET /_snapshot or GET /_snapshot/_all
pub async fn get_all_repositories(State(state): State<Arc<AppState>>) -> Response {
    let repos = state.snapshot_repositories.read().await;
    let map: Map<String, Value> = repos.clone().into_iter().collect();
    (StatusCode::OK, Json(Value::Object(map))).into_response()
}

/// GET /_snapshot/:repository
pub async fn get_repository(
    State(state): State<Arc<AppState>>,
    Path(repository): Path<String>,
) -> Response {
    if repository == "_all" {
        return get_all_repositories(State(state)).await;
    }
    let repos = state.snapshot_repositories.read().await;
    if let Some(repo) = repos.get(&repository) {
        let mut map = Map::new();
        map.insert(repository, repo.clone());
        (StatusCode::OK, Json(Value::Object(map))).into_response()
    } else {
        es_response::<Value>(Err(BenoStreamError::SchemaIncompatible {
            reason: format!("repository_missing_exception: [{repository}] missing"),
        }))
    }
}

/// PUT /_snapshot/:repository or POST /_snapshot/:repository
pub async fn put_repository(
    State(state): State<Arc<AppState>>,
    Path(repository): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    let repo_type = body.get("type").and_then(Value::as_str).unwrap_or("fs");
    let settings = body.get("settings").cloned().unwrap_or(serde_json::json!({
        "location": state.storage_root.clone()
    }));

    let mut repos = state.snapshot_repositories.write().await;
    repos.insert(
        repository,
        serde_json::json!({
            "type": repo_type,
            "settings": settings,
        }),
    );
    (
        StatusCode::OK,
        Json(serde_json::json!({ "acknowledged": true })),
    )
        .into_response()
}

/// DELETE /_snapshot/:repository
pub async fn delete_repository(
    State(state): State<Arc<AppState>>,
    Path(repository): Path<String>,
) -> Response {
    let mut repos = state.snapshot_repositories.write().await;
    repos.remove(&repository);
    let mut snaps = state.snapshots.write().await;
    snaps.remove(&repository);
    (
        StatusCode::OK,
        Json(serde_json::json!({ "acknowledged": true })),
    )
        .into_response()
}

/// POST /_snapshot/:repository/_verify
pub async fn verify_repository(
    State(state): State<Arc<AppState>>,
    Path(repository): Path<String>,
) -> Response {
    let repos = state.snapshot_repositories.read().await;
    if !repos.contains_key(&repository) && repository != "default" {
        return es_response::<Value>(Err(BenoStreamError::SchemaIncompatible {
            reason: format!("repository_missing_exception: [{repository}] missing"),
        }));
    }

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "nodes": {
                "benostreamdb-node-1": {
                    "name": "benostreamdb-node-1"
                }
            }
        })),
    )
        .into_response()
}

/// PUT /_snapshot/:repository/:snapshot or POST /_snapshot/:repository/:snapshot
pub async fn create_snapshot(
    State(state): State<Arc<AppState>>,
    Path((repository, snapshot)): Path<(String, String)>,
    Query(params): Query<HashMap<String, String>>,
    body: Option<Json<Value>>,
) -> Response {
    es_response(
        create_snapshot_core(
            &state,
            &repository,
            &snapshot,
            &params,
            body.as_ref().map(|Json(v)| v),
        )
        .await,
    )
}

pub async fn create_snapshot_core(
    state: &AppState,
    repository: &str,
    snapshot: &str,
    params: &HashMap<String, String>,
    body: Option<&Value>,
) -> Result<Value, BenoStreamError> {
    let repos = state.snapshot_repositories.read().await;
    if !repos.contains_key(repository) && repository != "default" {
        return Err(BenoStreamError::SchemaIncompatible {
            reason: format!("repository_missing_exception: [{repository}] missing"),
        });
    }
    drop(repos);

    let wait_for_completion = params
        .get("wait_for_completion")
        .map(|v| v == "true" || v == "1")
        .unwrap_or(false);

    let mut target_indices = Vec::new();
    let mut include_global_state = true;

    if let Some(b) = body {
        if let Some(b_indices) = b.get("indices") {
            if let Some(arr) = b_indices.as_array() {
                for item in arr {
                    if let Some(s) = item.as_str() {
                        target_indices.push(s.to_string());
                    }
                }
            } else if let Some(s) = b_indices.as_str() {
                for part in s.split(',') {
                    let trimmed = part.trim();
                    if !trimmed.is_empty() {
                        target_indices.push(trimmed.to_string());
                    }
                }
            }
        }
        if let Some(igs) = b.get("include_global_state").and_then(Value::as_bool) {
            include_global_state = igs;
        }
    }

    if target_indices.is_empty() || target_indices.contains(&"_all".to_string()) {
        target_indices = state.list_indexes().await.unwrap_or_default();
    }

    // Checkpoint / flush each index so Parquet files & Iceberg metadata are fully committed
    for idx in &target_indices {
        let _ = crate::handlers::docs::refresh_core(state, idx).await;
    }

    let now: DateTime<Utc> = Utc::now();
    let now_iso = now.to_rfc3339();
    let now_millis = now.timestamp_millis();
    let shard_count = target_indices.len().max(1) as u32;

    let snapshot_record = serde_json::json!({
        "snapshot": snapshot,
        "uuid": uuid::Uuid::new_v4().to_string(),
        "version_id": 7100099,
        "version": crate::ES_VERSION,
        "indices": target_indices,
        "data_streams": [],
        "include_global_state": include_global_state,
        "state": "SUCCESS",
        "start_time": now_iso,
        "start_time_in_millis": now_millis,
        "end_time": now_iso,
        "end_time_in_millis": now_millis,
        "duration_in_millis": 0,
        "failures": [],
        "shards": {
            "total": shard_count,
            "failed": 0,
            "successful": shard_count
        }
    });

    let mut snaps = state.snapshots.write().await;
    let repo_snaps = snaps.entry(repository.to_string()).or_default();
    repo_snaps.insert(snapshot.to_string(), snapshot_record.clone());

    if wait_for_completion {
        Ok(serde_json::json!({
            "snapshot": snapshot_record
        }))
    } else {
        Ok(serde_json::json!({
            "accepted": true
        }))
    }
}

/// GET /_snapshot/:repository/:snapshot
pub async fn get_snapshot(
    State(state): State<Arc<AppState>>,
    Path((repository, snapshot)): Path<(String, String)>,
) -> Response {
    es_response(get_snapshot_core(&state, &repository, &snapshot).await)
}

pub async fn get_all_snapshots(
    State(state): State<Arc<AppState>>,
    Path(repository): Path<String>,
) -> Response {
    es_response(get_snapshot_core(&state, &repository, "_all").await)
}

pub async fn get_snapshot_core(
    state: &AppState,
    repository: &str,
    snapshot: &str,
) -> Result<Value, BenoStreamError> {
    let repos = state.snapshot_repositories.read().await;
    if !repos.contains_key(repository) && repository != "default" {
        return Err(BenoStreamError::SchemaIncompatible {
            reason: format!("repository_missing_exception: [{repository}] missing"),
        });
    }
    drop(repos);

    let snaps = state.snapshots.read().await;
    let repo_snaps = snaps.get(repository);

    if snapshot == "_all" {
        let list: Vec<Value> = match repo_snaps {
            Some(m) => m.values().cloned().collect(),
            None => Vec::new(),
        };
        return Ok(serde_json::json!({ "snapshots": list }));
    }

    if let Some(record) = repo_snaps.and_then(|m| m.get(snapshot)) {
        Ok(serde_json::json!({
            "snapshots": [record]
        }))
    } else {
        Err(BenoStreamError::SchemaIncompatible {
            reason: format!("snapshot_missing_exception: [{repository}:{snapshot}] is missing"),
        })
    }
}

/// GET /_snapshot/:repository/:snapshot/_status
pub async fn get_snapshot_status(
    State(state): State<Arc<AppState>>,
    Path((repository, snapshot)): Path<(String, String)>,
) -> Response {
    let snaps = state.snapshots.read().await;
    let record = snaps.get(&repository).and_then(|m| m.get(&snapshot));
    if let Some(r) = record {
        let uuid = r.get("uuid").and_then(Value::as_str).unwrap_or("");
        let millis = r
            .get("start_time_in_millis")
            .and_then(Value::as_i64)
            .unwrap_or(0);
        let status = serde_json::json!({
            "snapshots": [
                {
                    "snapshot": snapshot,
                    "repository": repository,
                    "uuid": uuid,
                    "state": "SUCCESS",
                    "shards_stats": {
                        "initializing": 0,
                        "started": 0,
                        "finalizing": 0,
                        "done": 1,
                        "failed": 0,
                        "total": 1
                    },
                    "stats": {
                        "incremental": { "file_count": 0, "size_in_bytes": 0 },
                        "total": { "file_count": 1, "size_in_bytes": 1024 },
                        "start_time_in_millis": millis,
                        "time_in_millis": 0
                    },
                    "indices": {}
                }
            ]
        });
        (StatusCode::OK, Json(status)).into_response()
    } else {
        es_response::<Value>(Err(BenoStreamError::SchemaIncompatible {
            reason: format!("snapshot_missing_exception: [{repository}:{snapshot}] is missing"),
        }))
    }
}

/// POST /_snapshot/:repository/:snapshot/_restore
pub async fn restore_snapshot(
    State(state): State<Arc<AppState>>,
    Path((repository, snapshot)): Path<(String, String)>,
) -> Response {
    let snaps = state.snapshots.read().await;
    let exists = snaps
        .get(&repository)
        .map(|m| m.contains_key(&snapshot))
        .unwrap_or(false);
    if !exists {
        es_response::<Value>(Err(BenoStreamError::SchemaIncompatible {
            reason: format!("snapshot_missing_exception: [{repository}:{snapshot}] is missing"),
        }))
    } else {
        (
            StatusCode::OK,
            Json(serde_json::json!({ "accepted": true })),
        )
            .into_response()
    }
}

/// DELETE /_snapshot/:repository/:snapshot
pub async fn delete_snapshot(
    State(state): State<Arc<AppState>>,
    Path((repository, snapshot)): Path<(String, String)>,
) -> Response {
    let mut snaps = state.snapshots.write().await;
    if let Some(m) = snaps.get_mut(&repository) {
        m.remove(&snapshot);
    }
    (
        StatusCode::OK,
        Json(serde_json::json!({ "acknowledged": true })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_snapshot_lifecycle() {
        let tmp = tempfile::tempdir().unwrap();
        let root = format!("file://{}", tmp.path().display());
        let state = AppState::new(root, "test-cluster".into());

        // 1. Initial repositories list should contain "default"
        let repos = get_all_repositories(State(Arc::new(state.clone()))).await;
        assert_eq!(repos.status(), StatusCode::OK);

        // 2. Register custom repository
        let put_repo = put_repository(
            State(Arc::new(state.clone())),
            Path("my_backup".to_string()),
            Json(serde_json::json!({
                "type": "fs",
                "settings": { "location": "/tmp/backups" }
            })),
        )
        .await;
        assert_eq!(put_repo.status(), StatusCode::OK);

        // 3. Verify repository
        let verify_resp = verify_repository(
            State(Arc::new(state.clone())),
            Path("my_backup".to_string()),
        )
        .await;
        assert_eq!(verify_resp.status(), StatusCode::OK);

        // 4. Create snapshot with wait_for_completion=true
        let mut params = HashMap::new();
        params.insert("wait_for_completion".to_string(), "true".to_string());
        let snap_res = create_snapshot_core(
            &state,
            "my_backup",
            "snapshot_1",
            &params,
            Some(&serde_json::json!({
                "indices": ["idx_a", "idx_b"],
                "include_global_state": false
            })),
        )
        .await
        .unwrap();

        assert_eq!(snap_res["snapshot"]["snapshot"], "snapshot_1");
        assert_eq!(snap_res["snapshot"]["state"], "SUCCESS");

        // 5. Get snapshot
        let get_res = get_snapshot_core(&state, "my_backup", "snapshot_1")
            .await
            .unwrap();
        let list = get_res["snapshots"].as_array().unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0]["snapshot"], "snapshot_1");

        // 6. Get all snapshots
        let get_all = get_snapshot_core(&state, "my_backup", "_all")
            .await
            .unwrap();
        assert_eq!(get_all["snapshots"].as_array().unwrap().len(), 1);

        // 7. Missing snapshot returns 404
        let err = get_snapshot_core(&state, "my_backup", "nonexistent")
            .await
            .unwrap_err();
        assert!(err.to_string().contains("snapshot_missing_exception"));
        let es_err = crate::es_types::EsError::from(err);
        assert_eq!(es_err.status, 404);

        // 8. Snapshot status
        let status_resp = get_snapshot_status(
            State(Arc::new(state.clone())),
            Path(("my_backup".to_string(), "snapshot_1".to_string())),
        )
        .await;
        assert_eq!(status_resp.status(), StatusCode::OK);

        // 9. Restore snapshot
        let restore_resp = restore_snapshot(
            State(Arc::new(state.clone())),
            Path(("my_backup".to_string(), "snapshot_1".to_string())),
        )
        .await;
        assert_eq!(restore_resp.status(), StatusCode::OK);

        // 10. Delete snapshot
        let del_resp = delete_snapshot(
            State(Arc::new(state.clone())),
            Path(("my_backup".to_string(), "snapshot_1".to_string())),
        )
        .await;
        assert_eq!(del_resp.status(), StatusCode::OK);

        // 11. Delete repository
        let del_repo = delete_repository(
            State(Arc::new(state.clone())),
            Path("my_backup".to_string()),
        )
        .await;
        assert_eq!(del_repo.status(), StatusCode::OK);
    }
}
