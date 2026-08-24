// Copyright (c) 2026 Richard Albright. All rights reserved.

//! ES-style JSON types shared across handlers.

use hyperstreamdb::HyperstreamError;
use serde::Serialize;

/// ES-style error envelope: `{"error": {"type": ..., "reason": ...}, "status": ...}`
#[derive(Debug, Serialize)]
pub struct EsError {
    pub error: EsErrorBody,
    pub status: u16,
}

#[derive(Debug, Serialize)]
pub struct EsErrorBody {
    #[serde(rename = "type")]
    pub error_type: String,
    pub reason: String,
}

impl From<HyperstreamError> for EsError {
    fn from(err: HyperstreamError) -> Self {
        match &err {
            HyperstreamError::PrimaryKeyViolation { key } => EsError {
                error: EsErrorBody {
                    error_type: "resource_already_exists_exception".into(),
                    reason: format!("ID conflict: document with id '{key}' already exists"),
                },
                status: 400,
            },
            HyperstreamError::TableNotFound { .. } => EsError {
                error: EsErrorBody {
                    error_type: "index_not_found_exception".into(),
                    reason: err.to_string(),
                },
                status: 404,
            },
            HyperstreamError::InvalidUri { .. } => EsError {
                error: EsErrorBody {
                    error_type: "illegal_argument_exception".into(),
                    reason: err.to_string(),
                },
                status: 400,
            },
            _ => EsError {
                error: EsErrorBody {
                    error_type: "internal_error".into(),
                    reason: err.to_string(),
                },
                status: 500,
            },
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ClusterInfo {
    pub name: String,
    pub cluster_name: String,
    pub cluster_uuid: String,
    pub version: VersionInfo,
    pub tagline: String,
}

#[derive(Debug, Serialize)]
pub struct VersionInfo {
    pub number: String,
    pub build_flavor: String,
    pub build_type: String,
    pub build_hash: String,
    pub build_date: String,
    pub build_snapshot: bool,
    pub lucene_version: String,
    pub minimum_wire_compatibility_version: String,
    pub minimum_index_compatibility_version: String,
}

impl Default for VersionInfo {
    fn default() -> Self {
        Self::new()
    }
}

impl VersionInfo {
    pub fn new() -> Self {
        Self {
            number: crate::ES_VERSION.to_string(),
            build_flavor: "default".into(),
            build_type: "tar".into(),
            build_hash: "unknown".into(),
            build_date: "2020-10-16T01:14:24.050548Z".into(),
            build_snapshot: false,
            lucene_version: "8.7.0".into(),
            minimum_wire_compatibility_version: "6.8.0".into(),
            minimum_index_compatibility_version: "6.0.0-beta1".into(),
        }
    }
}

/// ES 7.10 cluster health (single node, one primary shard per index).
#[derive(Debug, Serialize)]
pub struct ClusterHealth {
    pub cluster_name: String,
    pub status: String,
    pub timed_out: bool,
    pub number_of_nodes: u32,
    pub number_of_data_nodes: u32,
    pub active_primary_shards: u32,
    pub active_shards: u32,
    pub relocating_shards: u32,
    pub initializing_shards: u32,
    pub unassigned_shards: u32,
    pub delayed_unassigned_shards: u32,
    pub number_of_pending_tasks: u32,
    pub number_of_in_flight_fetch: u32,
    pub task_max_waiting_in_queue_millis: u64,
    pub active_shards_percent_as_number: f64,
}
