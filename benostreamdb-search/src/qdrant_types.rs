// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Qdrant REST API JSON Types.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Standard Qdrant JSON response envelope.
#[derive(Debug, Serialize)]
pub struct QdrantResponse<T> {
    pub result: T,
    pub status: String,
    pub time: f64,
}

impl<T> QdrantResponse<T> {
    pub fn ok(result: T, time: f64) -> Self {
        Self {
            result,
            status: "ok".to_string(),
            time,
        }
    }
}

/// A standard Qdrant error response.
#[derive(Debug, Serialize)]
pub struct QdrantErrorResponse {
    pub status: QdrantErrorStatus,
    pub time: f64,
}

#[derive(Debug, Serialize)]
pub struct QdrantErrorStatus {
    pub error: String,
}

// ==========================================
// Collections API Types
// ==========================================

#[derive(Debug, Deserialize)]
pub struct CreateCollectionRequest {
    pub vectors: VectorsConfig,
    pub hnsw_config: Option<HnswConfig>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VectorsConfig {
    pub size: usize,
    pub distance: String,
    pub on_disk: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HnswConfig {
    pub m: Option<usize>,
    pub ef_construct: Option<usize>,
    pub on_disk: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct CollectionInfoResult {
    pub status: String,
    pub vectors_count: usize,
    pub points_count: usize,
    pub config: CollectionConfig,
}

#[derive(Debug, Serialize)]
pub struct CollectionConfig {
    pub params: CollectionParams,
}

#[derive(Debug, Serialize)]
pub struct CollectionParams {
    pub vectors: VectorsConfig,
}

#[derive(Debug, Deserialize)]
pub struct CreatePayloadIndexRequest {
    pub field_name: String,
    pub field_schema: String,
}

// ==========================================
// Points API Types
// ==========================================

#[derive(Debug, Deserialize)]
pub struct UpsertPointsRequest {
    pub points: Vec<PointStruct>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PointStruct {
    pub id: PointId,
    pub vector: Vec<f32>,
    pub payload: HashMap<String, Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum PointId {
    Num(u64),
    Uuid(String),
}

impl PointId {
    pub fn as_string(&self) -> String {
        match self {
            PointId::Num(n) => n.to_string(),
            PointId::Uuid(s) => s.clone(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct RetrievePointsRequest {
    pub ids: Vec<PointId>,
}

#[derive(Debug, Serialize)]
pub struct RetrievedPoint {
    pub id: PointId,
    pub payload: HashMap<String, Value>,
}

#[derive(Debug, Deserialize)]
pub struct QueryPointsRequest {
    pub query: Vec<f32>,
    pub filter: Option<Filter>,
    pub score_threshold: Option<f32>,
    pub limit: Option<usize>,
    pub params: Option<SearchParams>,
    pub with_payload: Option<WithPayload>,
}

#[derive(Debug, Deserialize)]
pub struct SearchParams {
    pub hnsw_ef: Option<usize>,
    pub exact: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum WithPayload {
    Bool(bool),
    Include { include: Vec<String> },
}

#[derive(Debug, Serialize)]
pub struct ScoredPoint {
    pub id: PointId,
    pub version: usize,
    pub score: f32,
    pub payload: Option<HashMap<String, Value>>,
}

// ==========================================
// Filter Types
// ==========================================

#[derive(Debug, Deserialize)]
pub struct Filter {
    pub must: Option<Vec<Condition>>,
    pub must_not: Option<Vec<Condition>>,
    pub should: Option<Vec<Condition>>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Condition {
    Field(FieldCondition),
    Filter(Filter),
}

#[derive(Debug, Deserialize)]
pub struct FieldCondition {
    pub key: String,
    #[serde(rename = "match")]
    pub match_: Option<MatchCondition>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum MatchCondition {
    Value { value: Value },
    // E.g. match: { any: [...] } could be added here if needed by Zoo Code
}

#[derive(Debug, Deserialize)]
pub struct DeletePointsRequest {
    pub filter: Option<Filter>,
    // we ignore 'wait' in this simple emulation
}

#[derive(Debug, Serialize)]
pub struct UpdateResult {
    pub operation_id: usize,
    pub status: String,
}
