// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Cluster-level endpoints: `GET /`, `GET /_health`, `GET /_cluster/health`,
//! `GET /_cluster/stats`, and `GET /_cat/indices`.

use axum::extract::State;
use axum::response::IntoResponse;
use axum::response::Response;
use axum::Json;
use std::sync::Arc;

use super::es_response;
use crate::es_types::{ClusterHealth, ClusterInfo, VersionInfo};
use crate::state::AppState;
use crate::TAGLINE;

/// `GET /` — cluster root info (ES 7.10 shape).
pub async fn cluster_info(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let info = ClusterInfo {
        name: "bsdb-search-1".to_string(),
        cluster_name: "bsdb-search".to_string(),
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
        cluster_name: "bsdb-search".to_string(),
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
        "cluster_name": "bsdb-search",
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

/// `GET /_cat/master` and `GET /_cat/cluster_manager`
pub async fn cat_cluster_manager() -> Response {
    let body = "id\thost\tip\tnode\nbsdb-node-1\t127.0.0.1\t127.0.0.1\tbsdb-search-1";
    let mut resp = (axum::http::StatusCode::OK, body).into_response();
    resp.headers_mut().insert(
        axum::http::header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static("text/plain; charset=UTF-8"),
    );
    resp
}

/// `GET /_cat/nodes`
pub async fn cat_nodes() -> Response {
    let body = "ip\theap.percent\tram.percent\tcpu\tload_1m\tload_5m\tload_15m\tnode.role\tmaster\tname\n127.0.0.1\t0\t0\t0\t0.00\t0.00\t0.00\tcdfhimrst\t*\tbsdb-search-1";
    let mut resp = (axum::http::StatusCode::OK, body).into_response();
    resp.headers_mut().insert(
        axum::http::header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static("text/plain; charset=UTF-8"),
    );
    resp
}

/// `GET /_cat/shards`
pub async fn cat_shards(State(state): State<Arc<AppState>>) -> Response {
    let mut lines = vec!["index\tshard\tprirep\tstate\tdocs\tstore\tip\tnode".to_string()];
    if let Ok(indexes) = state.list_indexes().await {
        for index in indexes {
            lines.push(format!(
                "{index}\t0\tp\tSTARTED\t0\t0b\t127.0.0.1\tbsdb-search-1"
            ));
        }
    }
    let mut resp = (axum::http::StatusCode::OK, lines.join("\n")).into_response();
    resp.headers_mut().insert(
        axum::http::header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static("text/plain; charset=UTF-8"),
    );
    resp
}

/// `GET /_cat/health`
pub async fn cat_health() -> Response {
    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let body = format!(
        "epoch\ttimestamp\tcluster\tstatus\tnode.total\tnode.data\tshards\tpri\trelo\tinit\tunassign\tpending_tasks\tmax_task_wait_time\tactive_shards_percent\n\
         {now_secs}\t00:00:00\tbsdb-search\tgreen\t1\t1\t1\t1\t0\t0\t0\t0\t-\t100.0%"
    );
    let mut resp = (axum::http::StatusCode::OK, body).into_response();
    resp.headers_mut().insert(
        axum::http::header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static("text/plain; charset=UTF-8"),
    );
    resp
}

/// `GET /_nodes`, `GET /_nodes/_all`, `GET /_nodes/http`, `GET /_nodes/stats`
pub async fn nodes_info(State(_state): State<Arc<AppState>>) -> Response {
    let info = serde_json::json!({
        "_nodes": {
            "total": 1,
            "successful": 1,
            "failed": 0
        },
        "cluster_name": "bsdb-search",
        "nodes": {
            "bsdb-node-1": {
                "name": "bsdb-search-1",
                "transport_address": "127.0.0.1:9300",
                "host": "127.0.0.1",
                "ip": "127.0.0.1",
                "version": crate::ES_VERSION,
                "build_flavor": "default",
                "build_type": "tar",
                "build_hash": "unknown",
                "roles": ["cluster_manager", "master", "data", "ingest"],
                "http": {
                    "bound_address": ["127.0.0.1:9200"],
                    "publish_address": "127.0.0.1:9200",
                    "max_content_length_in_bytes": 104857600
                }
            }
        }
    });
    es_response(Ok(info))
}

/// `GET /_cluster/settings` and `PUT /_cluster/settings`
pub async fn get_cluster_settings() -> Response {
    es_response(Ok(serde_json::json!({
        "persistent": {},
        "transient": {}
    })))
}

pub async fn put_cluster_settings() -> Response {
    es_response(Ok(serde_json::json!({
        "acknowledged": true,
        "persistent": {},
        "transient": {}
    })))
}

/// `GET /_xpack`
pub async fn xpack_info() -> Response {
    es_response(Ok(serde_json::json!({
        "build": {
            "hash": "unknown",
            "date": "2020-10-16T01:14:24.050548Z"
        },
        "license": {
            "uid": "benostreamdb-license",
            "type": "basic",
            "mode": "basic",
            "status": "active"
        },
        "features": {}
    })))
}

/// `GET /_license`
pub async fn license_info() -> Response {
    es_response(Ok(serde_json::json!({
        "license": {
            "status": "active",
            "uid": "benostreamdb-license",
            "type": "basic",
            "issue_date": "2020-10-16T01:14:24.050548Z",
            "issue_date_in_millis": 1602810864050u64,
            "expiry_date": "2099-12-31T23:59:59.999Z",
            "expiry_date_in_millis": 4102444799999u64,
            "max_nodes": 1000,
            "issued_to": "BenoStreamDB",
            "issuer": "BenoStreamDB"
        }
    })))
}

/// `GET /_ingest/pipeline`
pub async fn get_ingest_pipeline() -> Response {
    es_response(Ok(serde_json::json!({})))
}

pub async fn put_ingest_pipeline() -> Response {
    es_response(Ok(serde_json::json!({ "acknowledged": true })))
}

/// Global fallback handler for any unmapped route under the search daemon.
/// Returns a graceful ES-shaped 400 error envelope so client JSON parsers do not crash.
pub async fn fallback_unimplemented(req: axum::extract::Request) -> Response {
    let path = req.uri().path().to_string();
    let method = req.method().to_string();
    let es = crate::es_types::EsError {
        error: crate::es_types::EsErrorBody {
            error_type: "illegal_argument_exception".into(),
            reason: format!("No handler found for uri [{path}] and method [{method}]"),
        },
        status: 400,
    };
    let mut resp = (axum::http::StatusCode::BAD_REQUEST, Json(es)).into_response();
    resp.headers_mut().insert(
        axum::http::HeaderName::from_static("x-elastic-product"),
        axum::http::HeaderValue::from_static("Elasticsearch"),
    );
    resp.headers_mut().insert(
        axum::http::HeaderName::from_static("x-opensearch-version"),
        axum::http::HeaderValue::from_static("3.0.0"),
    );
    resp
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

        assert_eq!(json["cluster_name"], "bsdb-search");
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
        assert_eq!(json["cluster_name"], "bsdb-search");
        assert_eq!(json["compute"]["backend"], "cpu");

        let cat_resp = cat_indices(State(state)).await.into_response();
        assert_eq!(cat_resp.status(), axum::http::StatusCode::OK);
        let cat_bytes = axum::body::to_bytes(cat_resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8(cat_bytes.to_vec()).unwrap();
        assert!(text.starts_with("health\tstatus\tindex"));
    }

    #[tokio::test]
    async fn test_cluster_stubs_and_fallback() {
        let tmp = tempfile::tempdir().unwrap();
        let root = format!("file://{}", tmp.path().display());
        let state = Arc::new(AppState::new(root, "test-cluster".into()));

        // 1. Cat master / cluster manager
        let cm_resp = cat_cluster_manager().await;
        assert_eq!(cm_resp.status(), axum::http::StatusCode::OK);

        // 2. Cat nodes & shards & health
        let nodes_resp = cat_nodes().await;
        assert_eq!(nodes_resp.status(), axum::http::StatusCode::OK);
        let shards_resp = cat_shards(State(state.clone())).await;
        assert_eq!(shards_resp.status(), axum::http::StatusCode::OK);
        let health_resp = cat_health().await;
        assert_eq!(health_resp.status(), axum::http::StatusCode::OK);

        // 3. Nodes info
        let n_info = nodes_info(State(state.clone())).await;
        assert_eq!(n_info.status(), axum::http::StatusCode::OK);

        // 4. Cluster settings
        let cs_get = get_cluster_settings().await;
        assert_eq!(cs_get.status(), axum::http::StatusCode::OK);
        let cs_put = put_cluster_settings().await;
        assert_eq!(cs_put.status(), axum::http::StatusCode::OK);

        // 5. XPack & License
        let xp = xpack_info().await;
        assert_eq!(xp.status(), axum::http::StatusCode::OK);
        let lic = license_info().await;
        assert_eq!(lic.status(), axum::http::StatusCode::OK);

        // 6. Ingest pipeline stubs
        let ip_get = get_ingest_pipeline().await;
        assert_eq!(ip_get.status(), axum::http::StatusCode::OK);
        let ip_put = put_ingest_pipeline().await;
        assert_eq!(ip_put.status(), axum::http::StatusCode::OK);

        // 7. Fallback for unmapped route
        let dummy_req = axum::extract::Request::builder()
            .method("POST")
            .uri("/_plugins/_custom_unknown")
            .body(axum::body::Body::empty())
            .unwrap();
        let fb_resp = fallback_unimplemented(dummy_req).await;
        assert_eq!(fb_resp.status(), axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(
            fb_resp.headers().get("x-elastic-product").unwrap(),
            "Elasticsearch"
        );
        assert_eq!(
            fb_resp.headers().get("x-opensearch-version").unwrap(),
            "3.0.0"
        );

        let fb_bytes = axum::body::to_bytes(fb_resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let fb_json: serde_json::Value = serde_json::from_slice(&fb_bytes).unwrap();
        assert_eq!(fb_json["status"], 400);
        assert_eq!(fb_json["error"]["type"], "illegal_argument_exception");
        assert!(fb_json["error"]["reason"]
            .as_str()
            .unwrap()
            .contains("No handler found for uri [/_plugins/_custom_unknown] and method [POST]"));
    }
}
