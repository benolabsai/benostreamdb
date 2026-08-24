// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Cluster-level endpoints: `GET /`, `GET /_health`, `GET /_cluster/health`.

use axum::extract::State;
use axum::response::IntoResponse;
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
