// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! End-to-end conformance tests for the Qdrant-compatible REST API.
//!
//! The router is exercised in-process with `tower::ServiceExt::oneshot`, so
//! the suite runs under plain `cargo test` with no network or spawned binary.
//! `AppState::new` uses the CPU compute context, which keeps the per-commit
//! index builds fast.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use benostreamdb_search::handlers::qdrant;
use benostreamdb_search::state::AppState;
use serde_json::{json, Value};
use tower::ServiceExt;

struct Harness {
    app: axum::Router,
    _tmp: tempfile::TempDir,
}

impl Harness {
    fn new() -> Self {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = format!("file://{}", tmp.path().display());
        let state = Arc::new(AppState::new(root, "qdrant-test".to_string()));
        Self {
            app: qdrant::router(state),
            _tmp: tmp,
        }
    }

    async fn call(&self, method: &str, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
        let mut builder = Request::builder().method(method).uri(uri);
        let req = match body {
            Some(b) => {
                builder = builder.header("content-type", "application/json");
                builder.body(Body::from(b.to_string())).expect("request")
            }
            None => builder.body(Body::empty()).expect("request"),
        };
        let resp = self.app.clone().oneshot(req).await.expect("oneshot");
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("body");
        let val: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        (status, val)
    }

    async fn ok(&self, method: &str, uri: &str, body: Option<Value>) -> Value {
        let (status, val) = self.call(method, uri, body).await;
        assert!(status.is_success(), "{method} {uri} -> {status}: {val}");
        assert_eq!(val["status"], "ok", "{method} {uri}: {val}");
        val["result"].clone()
    }
}

#[tokio::test]
async fn service_endpoints() {
    let h = Harness::new();
    let info = h.ok("GET", "/", None).await;
    assert_eq!(info["title"], "benostreamdb");
    assert!(info["version"].is_string());

    for path in ["/healthz", "/livez", "/readyz"] {
        let (status, _) = h.call("GET", path, None).await;
        assert_eq!(status, StatusCode::OK, "{path}");
    }

    let telemetry = h.ok("GET", "/telemetry", None).await;
    assert_eq!(telemetry["title"], "benostreamdb");
    assert!(telemetry["collections"].is_number());
}

#[tokio::test]
async fn collection_lifecycle() {
    let h = Harness::new();

    // Create.
    let created = h
        .ok(
            "PUT",
            "/collections/demo",
            Some(json!({"vectors": {"size": 4, "distance": "Euclid"}})),
        )
        .await;
    assert_eq!(created, json!(true));

    // Duplicate create is a 400.
    let (status, _) = h
        .call(
            "PUT",
            "/collections/demo",
            Some(json!({"vectors": {"size": 4, "distance": "Euclid"}})),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // Exists.
    let exists = h.ok("GET", "/collections/demo/exists", None).await;
    assert_eq!(exists["exists"], json!(true));
    let missing = h.ok("GET", "/collections/nope/exists", None).await;
    assert_eq!(missing["exists"], json!(false));

    // List.
    let list = h.ok("GET", "/collections", None).await;
    let names: Vec<&str> = list["collections"]
        .as_array()
        .expect("collections")
        .iter()
        .filter_map(|c| c["name"].as_str())
        .collect();
    assert!(names.contains(&"demo"), "{list}");

    // Get: real (zero) point count and the configured distance.
    let info = h.ok("GET", "/collections/demo", None).await;
    assert_eq!(info["points_count"], json!(0));
    assert_eq!(info["config"]["params"]["vectors"]["size"], json!(4));
    assert_eq!(info["config"]["params"]["vectors"]["distance"], "Euclid");

    // Update (PATCH) is accepted.
    let patched = h
        .ok(
            "PATCH",
            "/collections/demo",
            Some(json!({"hnsw_config": {"m": 16}})),
        )
        .await;
    assert_eq!(patched, json!(true));

    // Payload index create + delete.
    let idx = h
        .ok(
            "PUT",
            "/collections/demo/index",
            Some(json!({"field_name": "city", "field_schema": "keyword"})),
        )
        .await;
    assert_eq!(idx["status"], "completed");
    let del_idx = h.ok("DELETE", "/collections/demo/index/city", None).await;
    assert_eq!(del_idx["status"], "completed");

    // Delete.
    let deleted = h.ok("DELETE", "/collections/demo", None).await;
    assert_eq!(deleted, json!(true));
    let (status, _) = h.call("GET", "/collections/demo", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn points_upsert_retrieve_and_count() {
    let h = Harness::new();
    h.ok(
        "PUT",
        "/collections/demo",
        Some(json!({"vectors": {"size": 4, "distance": "Euclid"}})),
    )
    .await;

    h.ok(
        "PUT",
        "/collections/demo/points",
        Some(json!({"points": [
            {"id": 1, "vector": [1, 0, 0, 0], "payload": {"city": "London", "n": 1}},
            {"id": 2, "vector": [0, 1, 0, 0], "payload": {"city": "Paris", "n": 2}},
            {"id": 3, "vector": [0.9, 0.1, 0, 0], "payload": {"city": "London", "n": 3}}
        ]})),
    )
    .await;

    // points_count reflects the real row count.
    let info = h.ok("GET", "/collections/demo", None).await;
    assert_eq!(info["points_count"], json!(3));

    // Retrieve via POST.
    let got = h
        .ok(
            "POST",
            "/collections/demo/points",
            Some(json!({"ids": [1, 2]})),
        )
        .await;
    assert_eq!(got.as_array().expect("array").len(), 2);

    // Retrieve via GET (with a body).
    let got = h
        .ok("GET", "/collections/demo/points", Some(json!({"ids": [3]})))
        .await;
    assert_eq!(got[0]["id"], json!(3));

    // Get by id.
    let point = h.ok("GET", "/collections/demo/points/2", None).await;
    assert_eq!(point["id"], json!(2));
    assert_eq!(point["payload"]["city"], "Paris");

    // Missing point -> 404.
    let (status, _) = h.call("GET", "/collections/demo/points/999", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    // Count (all + filtered).
    let count = h
        .ok("POST", "/collections/demo/points/count", Some(json!({})))
        .await;
    assert_eq!(count["count"], json!(3));
    let count = h
        .ok(
            "POST",
            "/collections/demo/points/count",
            Some(json!({"filter": {"must": [{"key": "city", "match": {"value": "London"}}]}})),
        )
        .await;
    assert_eq!(count["count"], json!(2));

    // Scroll.
    let scroll = h
        .ok(
            "POST",
            "/collections/demo/points/scroll",
            Some(json!({"limit": 2})),
        )
        .await;
    assert_eq!(scroll["points"].as_array().expect("points").len(), 2);
    assert!(scroll["next_page_offset"].is_number());
}

#[tokio::test]
async fn search_score_semantics_follow_distance() {
    let h = Harness::new();
    h.ok(
        "PUT",
        "/collections/demo",
        Some(json!({"vectors": {"size": 4, "distance": "Euclid"}})),
    )
    .await;
    h.ok(
        "PUT",
        "/collections/demo/points",
        Some(json!({"points": [
            {"id": 1, "vector": [1, 0, 0, 0]},
            {"id": 2, "vector": [0, 1, 0, 0]},
            {"id": 3, "vector": [0.9, 0.1, 0, 0]}
        ]})),
    )
    .await;

    // Euclid: score is the true Euclidean distance (lower is better), and
    // results are ordered best-first.
    let hits = h
        .ok(
            "POST",
            "/collections/demo/points/search",
            Some(json!({"vector": [0.9, 0.1, 0, 0], "limit": 3})),
        )
        .await;
    let hits = hits.as_array().expect("hits");
    assert_eq!(hits[0]["id"], json!(3));
    let s0 = hits[0]["score"].as_f64().expect("score");
    assert!(s0.abs() < 1e-5, "exact match should score ~0, got {s0}");
    let s1 = hits[1]["score"].as_f64().expect("score");
    assert!(
        (s1 - 0.14142138).abs() < 1e-4,
        "expected sqrt(0.02), got {s1}"
    );
    let s2 = hits[2]["score"].as_f64().expect("score");
    assert!(
        (s2 - 1.2727922).abs() < 1e-4,
        "expected sqrt(1.62), got {s2}"
    );
    assert!(s0 <= s1 && s1 <= s2, "not best-first: {hits:?}");

    // Cosine: score is a similarity (higher is better).
    h.ok(
        "PUT",
        "/collections/cos",
        Some(json!({"vectors": {"size": 4, "distance": "Cosine"}})),
    )
    .await;
    h.ok(
        "PUT",
        "/collections/cos/points",
        Some(json!({"points": [
            {"id": 1, "vector": [1, 0, 0, 0]},
            {"id": 2, "vector": [0, 1, 0, 0]}
        ]})),
    )
    .await;
    let hits = h
        .ok(
            "POST",
            "/collections/cos/points/search",
            Some(json!({"vector": [1, 0, 0, 0], "limit": 2})),
        )
        .await;
    let hits = hits.as_array().expect("hits");
    assert_eq!(hits[0]["id"], json!(1));
    let top = hits[0]["score"].as_f64().expect("score");
    assert!(
        (top - 1.0).abs() < 1e-4,
        "cosine similarity should be ~1, got {top}"
    );
    assert!(top >= hits[1]["score"].as_f64().expect("score"));
}

#[tokio::test]
async fn query_recommend_discover() {
    let h = Harness::new();
    h.ok(
        "PUT",
        "/collections/demo",
        Some(json!({"vectors": {"size": 4, "distance": "Euclid"}})),
    )
    .await;
    h.ok(
        "PUT",
        "/collections/demo/points",
        Some(json!({"points": [
            {"id": 1, "vector": [1, 0, 0, 0]},
            {"id": 2, "vector": [0, 1, 0, 0]},
            {"id": 3, "vector": [0.9, 0.1, 0, 0]}
        ]})),
    )
    .await;

    // Universal query API (bare vector and {nearest}).
    let q = h
        .ok(
            "POST",
            "/collections/demo/points/query",
            Some(json!({"query": [0.9, 0.1, 0, 0], "limit": 2})),
        )
        .await;
    assert_eq!(q["points"].as_array().expect("points").len(), 2);
    let q = h
        .ok(
            "POST",
            "/collections/demo/points/query",
            Some(json!({"query": {"nearest": [0.9, 0.1, 0, 0]}, "limit": 2})),
        )
        .await;
    assert_eq!(q["points"][0]["id"], json!(3));

    // Query with no vector behaves like a scroll.
    let q = h
        .ok(
            "POST",
            "/collections/demo/points/query",
            Some(json!({"limit": 2})),
        )
        .await;
    assert_eq!(q["points"].as_array().expect("points").len(), 2);

    // Recommend.
    let rec = h
        .ok(
            "POST",
            "/collections/demo/points/recommend",
            Some(json!({"positive": [[1, 0, 0, 0]], "limit": 2})),
        )
        .await;
    assert_eq!(rec.as_array().expect("rec").len(), 2);

    // Discover.
    let disc = h
        .ok(
            "POST",
            "/collections/demo/points/discover",
            Some(json!({
                "target": [1, 0, 0, 0],
                "context": [{"positive": [1, 0, 0, 0], "negative": [0, 1, 0, 0]}],
                "limit": 2
            })),
        )
        .await;
    assert!(!disc.as_array().expect("disc").is_empty());
}

#[tokio::test]
async fn payload_and_vector_writes_take_effect() {
    let h = Harness::new();
    h.ok(
        "PUT",
        "/collections/demo",
        Some(json!({"vectors": {"size": 4, "distance": "Euclid"}})),
    )
    .await;
    h.ok(
        "PUT",
        "/collections/demo/points",
        Some(json!({"points": [
            {"id": 1, "vector": [1, 0, 0, 0], "payload": {"city": "London"}}
        ]})),
    )
    .await;

    // Set (merge) payload.
    h.ok(
        "POST",
        "/collections/demo/points/payload",
        Some(json!({"payload": {"tag": "x"}, "points": [1]})),
    )
    .await;
    let p = h.ok("GET", "/collections/demo/points/1", None).await;
    assert_eq!(p["payload"]["city"], "London");
    assert_eq!(p["payload"]["tag"], "x");

    // Overwrite payload.
    h.ok(
        "PUT",
        "/collections/demo/points/payload",
        Some(json!({"payload": {"only": "y"}, "points": [1]})),
    )
    .await;
    let p = h.ok("GET", "/collections/demo/points/1", None).await;
    assert_eq!(p["payload"]["only"], "y");
    assert!(p["payload"].get("city").is_none(), "{p}");

    // Delete payload keys.
    h.ok(
        "POST",
        "/collections/demo/points/payload/delete",
        Some(json!({"keys": ["only"], "points": [1]})),
    )
    .await;
    let p = h.ok("GET", "/collections/demo/points/1", None).await;
    assert!(p["payload"].get("only").is_none(), "{p}");

    // Clear payload.
    h.ok(
        "POST",
        "/collections/demo/points/payload/clear",
        Some(json!({"points": [1]})),
    )
    .await;
    let p = h.ok("GET", "/collections/demo/points/1", None).await;
    assert_eq!(p["payload"], json!({}));

    // Update vectors.
    h.ok(
        "PUT",
        "/collections/demo/points/vectors",
        Some(json!({"points": [{"id": 1, "vector": [0, 0, 1, 0]}]})),
    )
    .await;
    let p = h.ok("GET", "/collections/demo/points/1", None).await;
    assert_eq!(p["vector"], json!([0.0, 0.0, 1.0, 0.0]));
}

#[tokio::test]
async fn batch_and_deletes() {
    let h = Harness::new();
    h.ok(
        "PUT",
        "/collections/demo",
        Some(json!({"vectors": {"size": 4, "distance": "Euclid"}})),
    )
    .await;
    h.ok(
        "PUT",
        "/collections/demo/points",
        Some(json!({"points": [
            {"id": 1, "vector": [1, 0, 0, 0], "payload": {"city": "London"}},
            {"id": 2, "vector": [0, 1, 0, 0], "payload": {"city": "Paris"}}
        ]})),
    )
    .await;

    // Batch: upsert + set_payload.
    let batch = h
        .ok(
            "POST",
            "/collections/demo/points/batch",
            Some(json!({"operations": [
                {"upsert": {"points": [{"id": 3, "vector": [0, 0, 1, 0], "payload": {"city": "Rome"}}]}},
                {"set_payload": {"payload": {"z": 1}, "points": [3]}}
            ]})),
        )
        .await;
    assert_eq!(batch.as_array().expect("batch").len(), 2);
    let p = h.ok("GET", "/collections/demo/points/3", None).await;
    assert_eq!(p["payload"]["z"], json!(1));

    // Delete by ids.
    h.ok(
        "POST",
        "/collections/demo/points/delete",
        Some(json!({"points": [3]})),
    )
    .await;
    let (status, _) = h.call("GET", "/collections/demo/points/3", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    // Delete by filter.
    h.ok(
        "POST",
        "/collections/demo/points/delete",
        Some(json!({"filter": {"must": [{"key": "city", "match": {"value": "Paris"}}]}})),
    )
    .await;
    let (status, _) = h.call("GET", "/collections/demo/points/2", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    let count = h
        .ok("POST", "/collections/demo/points/count", Some(json!({})))
        .await;
    assert_eq!(count["count"], json!(1));
}

#[tokio::test]
async fn aliases() {
    let h = Harness::new();
    h.ok(
        "PUT",
        "/collections/demo",
        Some(json!({"vectors": {"size": 4, "distance": "Euclid"}})),
    )
    .await;
    h.ok(
        "PUT",
        "/collections/demo/points",
        Some(json!({"points": [{"id": 1, "vector": [1, 0, 0, 0]}]})),
    )
    .await;

    h.ok(
        "POST",
        "/collections/aliases",
        Some(json!({"actions": [
            {"create_alias": {"collection_name": "demo", "alias_name": "demo_alias"}}
        ]})),
    )
    .await;

    let list = h.ok("GET", "/collections/aliases", None).await;
    assert_eq!(list["aliases"][0]["alias_name"], "demo_alias");
    assert_eq!(list["aliases"][0]["collection_name"], "demo");

    // The alias resolves to the collection.
    let info = h.ok("GET", "/collections/demo_alias", None).await;
    assert_eq!(info["points_count"], json!(1));

    // Rename then delete.
    h.ok(
        "POST",
        "/collections/aliases",
        Some(json!({"actions": [
            {"rename_alias": {"old_alias_name": "demo_alias", "new_alias_name": "demo_alias2"}}
        ]})),
    )
    .await;
    let list = h.ok("GET", "/collections/aliases", None).await;
    assert_eq!(list["aliases"][0]["alias_name"], "demo_alias2");

    h.ok(
        "POST",
        "/collections/aliases",
        Some(json!({"actions": [{"delete_alias": {"alias_name": "demo_alias2"}}]})),
    )
    .await;
    let list = h.ok("GET", "/collections/aliases", None).await;
    assert_eq!(list["aliases"].as_array().expect("aliases").len(), 0);
}

#[tokio::test]
async fn error_envelope_shape() {
    let h = Harness::new();
    let (status, val) = h.call("GET", "/collections/missing", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(val["status"]["error"], "Collection not found");
    assert!(val["time"].is_number());
}

#[tokio::test]
async fn snapshots_lifecycle() {
    let h = Harness::new();
    h.ok(
        "PUT",
        "/collections/demo",
        Some(json!({"vectors": {"size": 4, "distance": "Cosine"}})),
    )
    .await;
    h.ok(
        "PUT",
        "/collections/demo/points",
        Some(json!({"points": [{"id": 1, "vector": [1, 0, 0, 0]}]})),
    )
    .await;

    // Create snapshot
    let snap = h.ok("POST", "/collections/demo/snapshots", None).await;
    let snap_name = snap["name"].as_str().expect("snapshot name");
    assert!(snap_name.starts_with("demo-"));
    assert!(snap["creation_time"].is_string());
    assert!(snap["size"].is_number());

    // List snapshots
    let snaps = h.ok("GET", "/collections/demo/snapshots", None).await;
    let list = snaps.as_array().expect("snapshot list");
    assert_eq!(list.len(), 1);
    assert_eq!(list[0]["name"], snap_name);

    // Recover snapshot
    let recovered = h
        .ok(
            "PUT",
            "/collections/demo/snapshots/recover",
            Some(json!({"location": snap_name})),
        )
        .await;
    assert_eq!(recovered, json!(true));

    // Delete snapshot
    let deleted = h
        .ok(
            "DELETE",
            &format!("/collections/demo/snapshots/{snap_name}"),
            None,
        )
        .await;
    assert_eq!(deleted, json!(true));

    // List snapshots is now empty
    let snaps = h.ok("GET", "/collections/demo/snapshots", None).await;
    assert_eq!(snaps.as_array().expect("snapshot list").len(), 0);
}

#[tokio::test]
async fn payload_indexing() {
    let h = Harness::new();
    h.ok(
        "PUT",
        "/collections/demo",
        Some(json!({"vectors": {"size": 4, "distance": "Cosine"}})),
    )
    .await;
    h.ok(
        "PUT",
        "/collections/demo/points",
        Some(json!({"points": [{"id": 1, "vector": [1, 0, 0, 0], "payload": {"tag": "alpha", "city": "London"}}]})),
    )
    .await;

    // Create payload index for keyword and text
    h.ok(
        "PUT",
        "/collections/demo/index",
        Some(json!({"field_name": "tag", "field_schema": "keyword"})),
    )
    .await;
    h.ok(
        "PUT",
        "/collections/demo/index",
        Some(json!({"field_name": "city", "field_schema": "text"})),
    )
    .await;

    // Verify payload_schema in GET collection info
    let info = h.ok("GET", "/collections/demo", None).await;
    assert!(info["payload_schema"]["tag"].is_object());
    assert!(info["payload_schema"]["city"].is_object());

    // Delete index
    h.ok("DELETE", "/collections/demo/index/tag", None).await;
    let info = h.ok("GET", "/collections/demo", None).await;
    assert!(info["payload_schema"]["tag"].is_null());
    assert!(info["payload_schema"]["city"].is_object());
}

#[tokio::test]
async fn recommend_best_score_and_query_prefetch_fusion() {
    let h = Harness::new();
    h.ok(
        "PUT",
        "/collections/demo",
        Some(json!({"vectors": {"size": 4, "distance": "Cosine"}})),
    )
    .await;
    h.ok(
        "PUT",
        "/collections/demo/points",
        Some(json!({"points": [
            {"id": 1, "vector": [1, 0, 0, 0], "payload": {"city": "Berlin"}},
            {"id": 2, "vector": [0, 1, 0, 0], "payload": {"city": "Paris"}},
            {"id": 3, "vector": [0.8, 0.2, 0, 0], "payload": {"city": "London"}}
        ]})),
    )
    .await;

    // Recommend with best_score
    let rec = h
        .ok(
            "POST",
            "/collections/demo/points/recommend",
            Some(json!({
                "positive": [[1, 0, 0, 0]],
                "negative": [[0, 1, 0, 0]],
                "strategy": "best_score",
                "limit": 2
            })),
        )
        .await;
    let rec_points = rec.as_array().expect("rec points");
    assert_eq!(rec_points.len(), 2);
    assert_eq!(rec_points[0]["id"], json!(1));

    // Query points with prefetch and RRF fusion
    let q = h
        .ok(
            "POST",
            "/collections/demo/points/query",
            Some(json!({
                "prefetch": [
                    {"query": [1, 0, 0, 0], "limit": 2},
                    {"query": [0.8, 0.2, 0, 0], "limit": 2}
                ],
                "query": {"fusion": "rrf"},
                "limit": 2
            })),
        )
        .await;
    let points = q["points"].as_array().expect("points");
    assert_eq!(points.len(), 2);

    // Fallback unmapped router test
    let (status, val) = h
        .call("GET", "/collections/demo/unmapped_route", None)
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(val["status"]["error"]
        .as_str()
        .expect("error")
        .contains("No route for URI"));
    assert!(val["time"].is_number());
}

#[tokio::test]
async fn sparse_vector_lifecycle_and_search() {
    let h = Harness::new();

    // 1. Create collection with sparse vector configuration
    let created = h
        .ok(
            "PUT",
            "/collections/sparse_docs",
            Some(json!({
                "sparse_vectors": {
                    "text": {}
                }
            })),
        )
        .await;
    assert_eq!(created, json!(true));

    // 2. Verify collection metadata exposes sparse_vectors
    let info = h.ok("GET", "/collections/sparse_docs", None).await;
    assert_eq!(info["status"], "green");
    assert!(info["config"]["params"]["sparse_vectors"]["text"].is_object());

    // 3. Upsert points with sparse vectors
    let upsert = h
        .ok(
            "PUT",
            "/collections/sparse_docs/points",
            Some(json!({
                "points": [
                    {
                        "id": 1,
                        "vector": {
                            "indices": [1, 5, 10],
                            "values": [0.5, 1.0, 0.2]
                        },
                        "payload": {"title": "doc 1", "category": "tech"}
                    },
                    {
                        "id": 2,
                        "vector": {
                            "indices": [2, 5, 20],
                            "values": [0.8, 0.5, 0.1]
                        },
                        "payload": {"title": "doc 2", "category": "science"}
                    },
                    {
                        "id": 3,
                        "vector": {
                            "indices": [10, 30],
                            "values": [0.9, 0.4]
                        },
                        "payload": {"title": "doc 3", "category": "tech"}
                    }
                ]
            })),
        )
        .await;
    assert_eq!(upsert["status"], "completed");

    // 4. Retrieve point 1 and check sparse vector wire format
    let retrieved = h
        .ok(
            "POST",
            "/collections/sparse_docs/points",
            Some(json!({
                "ids": [1],
                "with_vector": true,
                "with_payload": true
            })),
        )
        .await;
    let ret_pts = retrieved.as_array().expect("retrieved array");
    assert_eq!(ret_pts.len(), 1);
    assert_eq!(ret_pts[0]["id"], json!(1));
    assert_eq!(ret_pts[0]["payload"]["title"], json!("doc 1"));
    assert_eq!(ret_pts[0]["vector"]["indices"], json!([1, 5, 10]));
    assert_eq!(ret_pts[0]["vector"]["values"], json!([0.5, 1.0, 0.2]));

    // 5. Search using sparse vector dot product
    // Query with indices [5, 10] with weights [1.0, 1.0]
    // Doc 1 dot: (idx 5: 1.0*1.0) + (idx 10: 0.2*1.0) = 1.2
    // Doc 2 dot: (idx 5: 0.5*1.0) = 0.5
    // Doc 3 dot: (idx 10: 0.9*1.0) = 0.9
    // Expected order: Doc 1 (1.2), Doc 3 (0.9), Doc 2 (0.5)
    let search = h
        .ok(
            "POST",
            "/collections/sparse_docs/points/search",
            Some(json!({
                "vector": {
                    "indices": [5, 10],
                    "values": [1.0, 1.0]
                },
                "with_payload": true,
                "with_vector": true,
                "limit": 3
            })),
        )
        .await;
    let pts = search.as_array().expect("search points");
    assert_eq!(pts.len(), 3);
    assert_eq!(pts[0]["id"], json!(1));
    assert!((pts[0]["score"].as_f64().unwrap() - 1.2).abs() < 1e-4);
    assert_eq!(pts[1]["id"], json!(3));
    assert!((pts[1]["score"].as_f64().unwrap() - 0.9).abs() < 1e-4);
    assert_eq!(pts[2]["id"], json!(2));
    assert!((pts[2]["score"].as_f64().unwrap() - 0.5).abs() < 1e-4);

    // 6. Search with score_threshold
    let search_filtered = h
        .ok(
            "POST",
            "/collections/sparse_docs/points/search",
            Some(json!({
                "vector": {
                    "indices": [5, 10],
                    "values": [1.0, 1.0]
                },
                "score_threshold": 0.8,
                "limit": 3
            })),
        )
        .await;
    let filtered_pts = search_filtered.as_array().expect("filtered points");
    assert_eq!(filtered_pts.len(), 2);
    assert_eq!(filtered_pts[0]["id"], json!(1));
    assert_eq!(filtered_pts[1]["id"], json!(3));

    // 7. Universal query API with sparse prefetch and RRF fusion
    let q = h
        .ok(
            "POST",
            "/collections/sparse_docs/points/query",
            Some(json!({
                "prefetch": [
                    {
                        "query": {
                            "indices": [5],
                            "values": [1.0]
                        },
                        "limit": 2
                    },
                    {
                        "query": {
                            "indices": [10],
                            "values": [1.0]
                        },
                        "limit": 2
                    }
                ],
                "query": {"fusion": "rrf"},
                "limit": 3
            })),
        )
        .await;
    let query_pts = q["points"].as_array().expect("query points");
    assert!(!query_pts.is_empty());
}

#[tokio::test]
async fn hybrid_collection_lifecycle() {
    let h = Harness::new();

    // Create a collection supporting both dense and sparse vectors
    let created = h
        .ok(
            "PUT",
            "/collections/hybrid_demo",
            Some(json!({
                "vectors": {
                    "size": 4,
                    "distance": "Cosine"
                },
                "sparse_vectors": {
                    "splade": {}
                }
            })),
        )
        .await;
    assert_eq!(created, json!(true));

    let info = h.ok("GET", "/collections/hybrid_demo", None).await;
    assert_eq!(info["status"], "green");
    assert_eq!(info["config"]["params"]["vectors"]["size"], 4);
    assert!(info["config"]["params"]["sparse_vectors"]["splade"].is_object());
}
