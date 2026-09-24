// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Fuzz the Qdrant-compatible request-body deserialization. These types use
//! `#[serde(untagged)]` enums (`PointId`, `Condition`), which are a classic
//! source of pathological backtracking and error-path panics on hostile input.

#![no_main]

use libfuzzer_sys::fuzz_target;
use serde_json::Value;

use benostreamdb_search::qdrant_types::{
    CreateCollectionRequest, DeletePointsRequest, QueryPointsRequest, RetrievePointsRequest,
    UpsertPointsRequest,
};

fuzz_target!(|data: &[u8]| {
    let Ok(v) = serde_json::from_slice::<Value>(data) else {
        return;
    };
    let _ = serde_json::from_value::<UpsertPointsRequest>(v.clone());
    let _ = serde_json::from_value::<QueryPointsRequest>(v.clone());
    let _ = serde_json::from_value::<DeletePointsRequest>(v.clone());
    let _ = serde_json::from_value::<RetrievePointsRequest>(v.clone());
    let _ = serde_json::from_value::<CreateCollectionRequest>(v);
});
