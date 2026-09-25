// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Qdrant REST API JSON types.
//!
//! These mirror the Qdrant v1.x REST wire format closely enough for the
//! official clients and LangChain/Zoo Code integrations to talk to
//! `bsdb-search` without modification. Only the fields the server actually
//! honours are modelled; unknown fields are ignored by serde.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

// ==========================================
// Response envelopes
// ==========================================

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

/// Generic acknowledgement returned by mutating endpoints.
#[derive(Debug, Serialize)]
pub struct UpdateResult {
    pub operation_id: u64,
    pub status: String,
}

impl UpdateResult {
    pub fn completed() -> Self {
        Self {
            operation_id: 0,
            status: "completed".to_string(),
        }
    }
}

// ==========================================
// Collections API types
// ==========================================

#[derive(Debug, Deserialize)]
pub struct CreateCollectionRequest {
    pub vectors: VectorsConfig,
    #[serde(default)]
    pub hnsw_config: Option<HnswConfig>,
    #[serde(default)]
    pub on_disk_payload: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct VectorsConfig {
    pub size: usize,
    pub distance: String,
    #[serde(default)]
    pub on_disk: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct HnswConfig {
    #[serde(default)]
    pub m: Option<usize>,
    #[serde(default)]
    pub ef_construct: Option<usize>,
    #[serde(default)]
    pub on_disk: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct CollectionInfoResult {
    pub status: String,
    pub optimizer_status: String,
    pub vectors_count: usize,
    pub indexed_vectors_count: usize,
    pub points_count: usize,
    pub segments_count: usize,
    pub config: CollectionConfig,
    pub payload_schema: HashMap<String, Value>,
}

#[derive(Debug, Serialize)]
pub struct CollectionConfig {
    pub params: CollectionParams,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hnsw_config: Option<HnswConfig>,
}

#[derive(Debug, Serialize)]
pub struct CollectionParams {
    pub vectors: VectorsConfig,
}

#[derive(Debug, Serialize)]
pub struct CollectionsListResult {
    pub collections: Vec<CollectionDescription>,
}

#[derive(Debug, Serialize)]
pub struct CollectionDescription {
    pub name: String,
}

#[derive(Debug, Serialize)]
pub struct CollectionExistsResult {
    pub exists: bool,
}

/// `PATCH /collections/:name` — update collection parameters. Fields the
/// server cannot honour are accepted and ignored (documented in
/// `QDRANT_COMPATIBILITY.md`).
#[derive(Debug, Deserialize)]
pub struct UpdateCollectionRequest {
    #[serde(default)]
    pub vectors: Option<Value>,
    #[serde(default)]
    pub hnsw_config: Option<HnswConfig>,
    #[serde(default)]
    pub optimizers_config: Option<Value>,
    #[serde(default)]
    pub params: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct CreatePayloadIndexRequest {
    pub field_name: String,
    #[serde(default)]
    pub field_schema: Option<Value>,
}

// ==========================================
// Points API types
// ==========================================

#[derive(Debug, Deserialize)]
pub struct UpsertPointsRequest {
    pub points: Vec<PointStruct>,
    #[serde(default)]
    pub wait: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PointStruct {
    pub id: PointId,
    #[serde(default)]
    pub vector: Option<VectorInput>,
    #[serde(default)]
    pub payload: Option<HashMap<String, Value>>,
}

/// A point vector: either a single unnamed vector or a named-vector map.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum VectorInput {
    Single(Vec<f32>),
    Named(HashMap<String, Vec<f32>>),
}

impl VectorInput {
    /// The vector the engine stores in its single `vector` column. For named
    /// vectors the entry keyed `vector` (or the first entry) is used.
    pub fn primary(&self) -> Option<&Vec<f32>> {
        match self {
            VectorInput::Single(v) => Some(v),
            VectorInput::Named(m) => m.get("vector").or_else(|| m.values().next()),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Hash)]
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
    #[serde(default)]
    pub with_payload: Option<WithPayload>,
    #[serde(default)]
    pub with_vector: Option<WithVector>,
}

/// `with_payload` accepts a bool or an include/exclude selector.
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum WithPayload {
    Bool(bool),
    Selector(PayloadSelector),
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct PayloadSelector {
    #[serde(default)]
    pub include: Option<Vec<String>>,
    #[serde(default)]
    pub exclude: Option<Vec<String>>,
}

impl WithPayload {
    pub fn enabled(&self) -> bool {
        match self {
            WithPayload::Bool(b) => *b,
            WithPayload::Selector(_) => true,
        }
    }

    /// Whether a payload field survives the selector.
    pub fn keep(&self, field: &str) -> bool {
        match self {
            WithPayload::Bool(_) => true,
            WithPayload::Selector(sel) => {
                if let Some(include) = &sel.include {
                    if !include.is_empty() {
                        return include.iter().any(|i| i == field);
                    }
                }
                if let Some(exclude) = &sel.exclude {
                    return !exclude.iter().any(|e| e == field);
                }
                true
            }
        }
    }
}

/// `with_vector` accepts a bool or a list of named vectors.
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum WithVector {
    Bool(bool),
    Names(Vec<String>),
}

impl WithVector {
    pub fn enabled(&self) -> bool {
        match self {
            WithVector::Bool(b) => *b,
            WithVector::Names(_) => true,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct RetrievedPoint {
    pub id: PointId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<HashMap<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
}

#[derive(Debug, Serialize)]
pub struct ScoredPoint {
    pub id: PointId,
    pub version: u64,
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<HashMap<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
}

/// `POST /collections/:name/points/search` (legacy) request.
#[derive(Debug, Deserialize)]
pub struct SearchPointsRequest {
    pub vector: Vec<f32>,
    #[serde(default)]
    pub filter: Option<Filter>,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub offset: Option<usize>,
    #[serde(default)]
    pub with_payload: Option<WithPayload>,
    #[serde(default)]
    pub with_vector: Option<WithVector>,
    #[serde(default)]
    pub score_threshold: Option<f32>,
    #[serde(default)]
    pub params: Option<Value>,
}

/// `POST /collections/:name/points/query` (universal query API) request.
#[derive(Debug, Deserialize)]
pub struct QueryPointsRequest {
    #[serde(default)]
    pub query: Option<QueryInput>,
    #[serde(default)]
    pub filter: Option<Filter>,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub offset: Option<usize>,
    #[serde(default)]
    pub with_payload: Option<WithPayload>,
    #[serde(default)]
    pub with_vector: Option<WithVector>,
    #[serde(default)]
    pub using: Option<String>,
    #[serde(default)]
    pub score_threshold: Option<f32>,
    #[serde(default)]
    pub params: Option<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum QueryInput {
    Nearest(Vec<f32>),
    NearestObj { nearest: Vec<f32> },
}

impl QueryInput {
    pub fn vector(&self) -> &Vec<f32> {
        match self {
            QueryInput::Nearest(v) => v,
            QueryInput::NearestObj { nearest } => nearest,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ScrollRequest {
    #[serde(default)]
    pub filter: Option<Filter>,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub offset: Option<PointId>,
    #[serde(default)]
    pub with_payload: Option<WithPayload>,
    #[serde(default)]
    pub with_vector: Option<WithVector>,
}

#[derive(Debug, Serialize)]
pub struct ScrollResult {
    pub points: Vec<RetrievedPoint>,
    pub next_page_offset: Option<PointId>,
}

#[derive(Debug, Deserialize)]
pub struct CountRequest {
    #[serde(default)]
    pub filter: Option<Filter>,
    #[serde(default)]
    pub exact: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct CountResult {
    pub count: usize,
}

#[derive(Debug, Deserialize)]
pub struct RecommendRequest {
    pub positive: Vec<Vec<f32>>,
    #[serde(default)]
    pub negative: Option<Vec<Vec<f32>>>,
    #[serde(default)]
    pub strategy: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub filter: Option<Filter>,
    #[serde(default)]
    pub with_payload: Option<WithPayload>,
    #[serde(default)]
    pub with_vector: Option<WithVector>,
    #[serde(default)]
    pub using: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct DiscoverRequest {
    pub target: Vec<f32>,
    #[serde(default)]
    pub context: Option<Vec<ContextPair>>,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub filter: Option<Filter>,
    #[serde(default)]
    pub with_payload: Option<WithPayload>,
    #[serde(default)]
    pub with_vector: Option<WithVector>,
    #[serde(default)]
    pub using: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ContextPair {
    pub positive: Vec<f32>,
    pub negative: Vec<f32>,
}

#[derive(Debug, Deserialize)]
pub struct BatchRequest {
    pub operations: Vec<BatchOperation>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum BatchOperation {
    Upsert {
        upsert: PointsBatch,
    },
    Delete {
        delete: DeletePointsRequest,
    },
    SetPayload {
        set_payload: SetPayloadRequest,
    },
    OverwritePayload {
        overwrite_payload: SetPayloadRequest,
    },
    DeletePayload {
        delete_payload: DeletePayloadRequest,
    },
    ClearPayload {
        clear_payload: ClearPayloadRequest,
    },
    UpdateVectors {
        update_vectors: UpdateVectorsRequest,
    },
}

#[derive(Debug, Deserialize)]
pub struct PointsBatch {
    pub points: Vec<PointStruct>,
}

#[derive(Debug, Deserialize)]
pub struct DeletePointsRequest {
    #[serde(default)]
    pub points: Option<Vec<PointId>>,
    #[serde(default)]
    pub filter: Option<Filter>,
}

#[derive(Debug, Deserialize)]
pub struct SetPayloadRequest {
    pub payload: HashMap<String, Value>,
    #[serde(default)]
    pub points: Option<Vec<PointId>>,
    #[serde(default)]
    pub filter: Option<Filter>,
}

#[derive(Debug, Deserialize)]
pub struct DeletePayloadRequest {
    pub keys: Vec<String>,
    #[serde(default)]
    pub points: Option<Vec<PointId>>,
    #[serde(default)]
    pub filter: Option<Filter>,
}

#[derive(Debug, Deserialize)]
pub struct ClearPayloadRequest {
    #[serde(default)]
    pub points: Option<Vec<PointId>>,
    #[serde(default)]
    pub filter: Option<Filter>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateVectorsRequest {
    pub points: Vec<PointVectors>,
}

#[derive(Debug, Deserialize)]
pub struct PointVectors {
    pub id: PointId,
    pub vector: VectorInput,
}

// ==========================================
// Filter types
// ==========================================

#[derive(Debug, Deserialize, Clone)]
pub struct Filter {
    #[serde(default)]
    pub must: Option<ConditionList>,
    #[serde(default)]
    pub should: Option<ConditionList>,
    #[serde(default)]
    pub must_not: Option<ConditionList>,
}

/// Qdrant accepts a single condition or an array of conditions for each of
/// `must` / `should` / `must_not`.
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum ConditionList {
    One(Box<Condition>),
    Many(Vec<Condition>),
}

impl ConditionList {
    pub fn into_vec(self) -> Vec<Condition> {
        match self {
            ConditionList::One(c) => vec![*c],
            ConditionList::Many(v) => v,
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum Condition {
    Field(FieldCondition),
    HasId { has_id: Vec<PointId> },
    Filter(Box<Filter>),
}

#[derive(Debug, Deserialize, Clone)]
pub struct FieldCondition {
    pub key: String,
    #[serde(default, rename = "match")]
    pub match_: Option<MatchCondition>,
    #[serde(default)]
    pub range: Option<RangeCondition>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum MatchCondition {
    Value { value: Value },
    Any { any: Vec<Value> },
    Text { text: String },
}

#[derive(Debug, Deserialize, Clone)]
pub struct RangeCondition {
    #[serde(default)]
    pub lt: Option<f64>,
    #[serde(default)]
    pub gt: Option<f64>,
    #[serde(default)]
    pub gte: Option<f64>,
    #[serde(default)]
    pub lte: Option<f64>,
}

// ==========================================
// Aliases API types
// ==========================================

#[derive(Debug, Serialize)]
pub struct AliasesListResult {
    pub aliases: Vec<AliasDescription>,
}

#[derive(Debug, Serialize)]
pub struct AliasDescription {
    pub alias_name: String,
    pub collection_name: String,
}

#[derive(Debug, Deserialize)]
pub struct AliasOperationsRequest {
    pub actions: Vec<AliasAction>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum AliasAction {
    Create { create_alias: CreateAlias },
    Delete { delete_alias: DeleteAlias },
    Rename { rename_alias: RenameAlias },
}

#[derive(Debug, Deserialize)]
pub struct CreateAlias {
    pub collection_name: String,
    pub alias_name: String,
}

#[derive(Debug, Deserialize)]
pub struct DeleteAlias {
    pub alias_name: String,
}

#[derive(Debug, Deserialize)]
pub struct RenameAlias {
    pub old_alias_name: String,
    pub new_alias_name: String,
}

// ==========================================
// Service API types
// ==========================================

#[derive(Debug, Serialize)]
pub struct ServiceInfo {
    pub title: String,
    pub version: String,
    pub commit: String,
}

#[derive(Debug, Serialize)]
pub struct TelemetryResult {
    pub title: String,
    pub version: String,
    pub commit: String,
    pub collections: usize,
    pub uptime_seconds: f64,
}
