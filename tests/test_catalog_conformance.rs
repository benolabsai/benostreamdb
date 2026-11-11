// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Cross-catalog Iceberg contract & CAS conformance suite (`Workstream C4`).
//!
//! Exercises an identical lifecycle + multi-writer Compare-and-Swap (CAS)
//! convergence contract across:
//! - [`CatalogType::Memory`] (always-on in CI)
//! - [`CatalogType::Jdbc`] over SQLite `:memory:` & file-backed SQLite (always-on in CI),
//!   plus PostgreSQL when `BSDB_TEST_POSTGRES_URL` or `DATABASE_URL` is provided
//! - [`CatalogType::Rest`] against an embedded Axum Iceberg REST mock server
//!   (OAuth2 `/v1/oauth/tokens` refresh + `requirements` CAS 409 retry)
//! - [`CatalogType::Nessie`] when `NESSIE_URI` is provided
//! - SQL DDL (`CREATE SCHEMA` / `CREATE TABLE` / `INSERT` / `DROP TABLE` / `DROP SCHEMA`)
//!   and `bsdb-search`-style storage-first registration (`Table::create_async` before
//!   `catalog.create_table`).

use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow::array::{Array, Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use axum::extract::{Path as AxumPath, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use benostreamdb::core::catalog::common::{
    apply_table_updates, catalog_conflict_error, catalog_not_found_error,
    ensure_storage_table_initialized, extract_metadata_location_from_updates,
    load_table_metadata_from_location, validate_requirements_against_metadata,
};
use benostreamdb::core::catalog::jdbc::JdbcCatalogClient;
use benostreamdb::core::catalog::memory::MemoryCatalogClient;
use benostreamdb::core::catalog::rest::RestCatalogClient;
use benostreamdb::core::catalog::{create_catalog_async, Catalog, CatalogType};
use benostreamdb::core::error::BenoStreamError;
use benostreamdb::core::metadata::TableMetadata;
use benostreamdb::core::sql::session::BenoStreamSession;
use benostreamdb::core::table::builder::TableBuilder;
use benostreamdb::core::table::WalDurability;
use benostreamdb::Table;
use parking_lot::Mutex;
use tempfile::tempdir;

fn canon_dir_uri(dir: &Path) -> String {
    let canon = std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf());
    format!("file://{}", canon.display())
}

fn events_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("payload", DataType::Utf8, false),
    ]))
}

fn make_events_batch(start_id: i32, count: i32) -> Result<RecordBatch> {
    let ids = Int32Array::from_iter_values(start_id..start_id + count);
    let payloads =
        StringArray::from_iter_values((start_id..start_id + count).map(|i| format!("row_{i}")));
    Ok(RecordBatch::try_new(
        events_schema(),
        vec![Arc::new(ids), Arc::new(payloads)],
    )?)
}

fn is_catalog_conflict(err: &anyhow::Error) -> bool {
    err.downcast_ref::<BenoStreamError>().is_some_and(|e| {
        matches!(
            e,
            BenoStreamError::CatalogRequest {
                status: Some(409),
                ..
            }
        )
    }) || err.to_string().contains("409")
}

async fn read_sorted_ids(table: &Table) -> Result<Vec<i32>> {
    let batches = table.read_async(None, None, Some(&["id"])).await?;
    let mut ids = Vec::new();
    for b in batches {
        let col = b
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .context("expected Int32 id column")?;
        for i in 0..col.len() {
            if !col.is_null(i) {
                ids.push(col.value(i));
            }
        }
    }
    ids.sort_unstable();
    Ok(ids)
}

#[derive(Clone, Copy)]
struct ContractCapabilities {
    tracks_empty_namespaces: bool,
    enforces_nonempty_drop_namespace: bool,
    rejects_duplicate_create: bool,
}

async fn run_catalog_contract(
    catalog: Arc<dyn Catalog>,
    warehouse_uri: &str,
    caps: ContractCapabilities,
) -> Result<()> {
    let suffix = uuid::Uuid::new_v4().simple().to_string();
    let ns = format!("conf_ns_{suffix}");
    let tbl = "events";
    let table_loc = format!("{}/{}/{}", warehouse_uri.trim_end_matches('/'), ns, tbl);

    catalog.create_namespace(&ns).await?;
    if caps.tracks_empty_namespaces {
        assert!(
            catalog.namespace_exists(&ns).await?,
            "namespace_exists should return true right after create_namespace"
        );
    }
    assert!(
        catalog.list_tables(&ns).await?.is_empty(),
        "new namespace must start empty"
    );

    catalog
        .create_table(&ns, tbl, events_schema(), Some(&table_loc))
        .await?;
    assert!(catalog.table_exists(&ns, tbl).await?);
    assert_eq!(catalog.list_tables(&ns).await?, vec![tbl.to_string()]);

    if caps.rejects_duplicate_create {
        let dup_err = catalog
            .create_table(&ns, tbl, events_schema(), Some(&table_loc))
            .await
            .expect_err("duplicate create_table must error");
        assert!(
            is_catalog_conflict(&dup_err),
            "expected 409 conflict on duplicate create_table, got: {dup_err:#}"
        );
    }

    let meta1 = catalog.load_table(&ns, tbl).await?;
    assert!(
        !meta1.location.ends_with(".metadata.json") && !meta1.location.ends_with("/manifest.json"),
        "loaded TableMetadata.location must be table root, got: {}",
        meta1.location
    );

    let wal_dir = tempdir()?;
    let t = TableBuilder::new(&meta1.location)
        .with_catalog(catalog.clone(), &ns, tbl)
        .with_wal_dir(wal_dir.path().to_path_buf())
        .with_durability(WalDurability::Sync)
        .build_async()
        .await?;

    t.write_async(vec![make_events_batch(1, 5)?]).await?;
    t.commit_async().await?;

    let meta2 = catalog.load_table(&ns, tbl).await?;
    assert!(
        meta2.current_snapshot_id.is_some(),
        "catalog metadata must advance current_snapshot_id after commit"
    );
    assert!(
        !meta2.snapshots.is_empty(),
        "catalog metadata must record committed snapshot"
    );

    t.write_async(vec![make_events_batch(6, 5)?]).await?;
    t.commit_async().await?;

    let wal_dir_reopen = tempdir()?;
    let reopened = TableBuilder::new(&catalog.load_table(&ns, tbl).await?.location)
        .with_catalog(catalog.clone(), &ns, tbl)
        .with_wal_dir(wal_dir_reopen.path().to_path_buf())
        .build_async()
        .await?;
    assert_eq!(
        read_sorted_ids(&reopened).await?,
        (1..=10).collect::<Vec<_>>()
    );

    // Direct CAS negative check: stale snapshot-id requirement must return 409.
    let stale_err = catalog
        .commit_table_with_requirements(
            &ns,
            tbl,
            vec![serde_json::json!({
                "action": "set-properties",
                "updates": { "conf.probe": "1" }
            })],
            vec![serde_json::json!({
                "type": "assert-ref-snapshot-id",
                "ref": "main",
                "snapshot-id": 9_999_999_i64
            })],
        )
        .await
        .expect_err("stale assert-ref-snapshot-id must fail CAS");
    assert!(
        is_catalog_conflict(&stale_err),
        "expected HTTP-409 equivalent error on stale CAS requirement, got: {stale_err:#}"
    );

    // Valid property commit with matching snapshot-id must succeed.
    let current_meta = catalog.load_table(&ns, tbl).await?;
    catalog
        .commit_table_with_requirements(
            &ns,
            tbl,
            vec![serde_json::json!({
                "action": "set-properties",
                "updates": { "conf.probe": "ok" }
            })],
            vec![serde_json::json!({
                "type": "assert-ref-snapshot-id",
                "ref": "main",
                "snapshot-id": current_meta.current_snapshot_id
            })],
        )
        .await?;

    if caps.enforces_nonempty_drop_namespace {
        let non_empty_err = catalog
            .drop_namespace(&ns)
            .await
            .expect_err("drop_namespace must reject non-empty namespace");
        assert!(
            is_catalog_conflict(&non_empty_err),
            "expected 409 on non-empty drop_namespace, got: {non_empty_err:#}"
        );
    }

    catalog.drop_table(&ns, tbl).await?;
    assert!(!catalog.table_exists(&ns, tbl).await?);
    assert!(catalog.list_tables(&ns).await?.is_empty());
    assert!(
        catalog.load_table(&ns, tbl).await.is_err(),
        "load_table after drop_table must fail"
    );

    catalog.drop_namespace(&ns).await?;
    if caps.tracks_empty_namespaces {
        assert!(
            !catalog.namespace_exists(&ns).await?,
            "namespace_exists should return false after drop_namespace"
        );
    }

    Ok(())
}

// ----------------------------------------------------------------------------
// 1. Memory Catalog (always-on in CI)
// ----------------------------------------------------------------------------

#[tokio::test]
async fn memory_catalog_conformance() -> Result<()> {
    let wh_dir = tempdir()?;
    let wh_uri = canon_dir_uri(wh_dir.path());
    let mut cfg = HashMap::new();
    cfg.insert(
        "uri".to_string(),
        format!("memory://conf_{}", uuid::Uuid::new_v4()),
    );
    cfg.insert("warehouse".to_string(), wh_uri.clone());

    let catalog: Arc<dyn Catalog> =
        Arc::from(create_catalog_async(CatalogType::Memory, cfg).await?);
    run_catalog_contract(
        catalog,
        &wh_uri,
        ContractCapabilities {
            tracks_empty_namespaces: true,
            enforces_nonempty_drop_namespace: true,
            rejects_duplicate_create: true,
        },
    )
    .await
}

#[tokio::test]
async fn memory_catalog_concurrent_writers_409_convergence() -> Result<()> {
    let wh_dir = tempdir()?;
    let wh_uri = canon_dir_uri(wh_dir.path());
    let mem_uri = format!("memory://concurrent_{}", uuid::Uuid::new_v4());
    let catalog_seed = Arc::new(MemoryCatalogClient::for_uri(&mem_uri, Some(wh_uri.clone())));
    catalog_seed.create_namespace("race").await?;
    let table_loc = format!("{wh_uri}/race/events");
    catalog_seed
        .create_table("race", "events", events_schema(), Some(&table_loc))
        .await?;

    let wal_root = tempdir()?;
    let writers = 4usize;
    let rounds = 3usize;
    let rows_per_round = 5i32;

    let mut handles = Vec::new();
    for w in 0..writers {
        // Distinct client handle pointing at the same named memory:// catalog URI
        let cat: Arc<dyn Catalog> =
            Arc::new(MemoryCatalogClient::for_uri(&mem_uri, Some(wh_uri.clone())));
        let t = TableBuilder::new(&table_loc)
            .with_catalog(cat, "race", "events")
            .with_wal_dir(wal_root.path().join(format!("w_{w}")))
            .with_durability(WalDurability::Sync)
            .build_async()
            .await?;

        handles.push(tokio::spawn(async move {
            for r in 0..rounds {
                let start = 100 + (w as i32) * 100 + (r as i32) * rows_per_round;
                t.write_async(vec![make_events_batch(start, rows_per_round)?])
                    .await?;
                t.commit_async().await?;
            }
            Ok::<_, anyhow::Error>(())
        }));
    }

    for h in handles {
        h.await??;
    }

    let check_wal = tempdir()?;
    let final_table = TableBuilder::new(&table_loc)
        .with_wal_dir(check_wal.path().to_path_buf())
        .build_async()
        .await?;
    let actual_ids = read_sorted_ids(&final_table).await?;

    let mut expected_ids = BTreeSet::new();
    for w in 0..writers {
        for r in 0..rounds {
            let start = 100 + (w as i32) * 100 + (r as i32) * rows_per_round;
            for i in start..start + rows_per_round {
                expected_ids.insert(i);
            }
        }
    }
    assert_eq!(
        actual_ids.into_iter().collect::<BTreeSet<_>>(),
        expected_ids,
        "all concurrent MemoryCatalog writers must converge with zero lost rows"
    );

    let meta = catalog_seed.load_table("race", "events").await?;
    assert_eq!(meta.snapshots.len(), writers * rounds);
    assert!(catalog_seed
        .metadata_location("race", "events")
        .is_some_and(|loc| loc.contains("/metadata/v")));

    Ok(())
}

// ----------------------------------------------------------------------------
// 2. JDBC Catalog: SQLite :memory:, SQLite file CAS, & optional Postgres
// ----------------------------------------------------------------------------

#[tokio::test]
async fn jdbc_sqlite_memory_conformance() -> Result<()> {
    let wh_dir = tempdir()?;
    let wh_uri = canon_dir_uri(wh_dir.path());
    let mut cfg = HashMap::new();
    cfg.insert("uri".to_string(), "sqlite::memory:".to_string());
    cfg.insert("warehouse".to_string(), wh_uri.clone());
    cfg.insert("catalog_name".to_string(), "test_mem".to_string());

    let catalog: Arc<dyn Catalog> = Arc::from(create_catalog_async(CatalogType::Jdbc, cfg).await?);
    run_catalog_contract(
        catalog,
        &wh_uri,
        ContractCapabilities {
            tracks_empty_namespaces: true,
            enforces_nonempty_drop_namespace: true,
            rejects_duplicate_create: true,
        },
    )
    .await
}

#[tokio::test]
async fn jdbc_sqlite_file_conformance_and_concurrent_cas() -> Result<()> {
    let dir = tempdir()?;
    let wh_dir = tempdir()?;
    let wh_uri = canon_dir_uri(wh_dir.path());
    let db_path = dir.path().join("catalog.sqlite");
    let jdbc_uri = format!("sqlite://{}", db_path.display());

    let mut cfg = HashMap::new();
    cfg.insert("uri".to_string(), jdbc_uri.clone());
    cfg.insert("warehouse".to_string(), wh_uri.clone());
    cfg.insert("catalog_name".to_string(), "bench".to_string());

    let catalog: Arc<dyn Catalog> = Arc::from(create_catalog_async(CatalogType::Jdbc, cfg).await?);
    run_catalog_contract(
        catalog.clone(),
        &wh_uri,
        ContractCapabilities {
            tracks_empty_namespaces: true,
            enforces_nonempty_drop_namespace: true,
            rejects_duplicate_create: true,
        },
    )
    .await?;

    // Separate JdbcCatalogClient pools pointing to the same SQLite DB must converge under race.
    let client_a = Arc::new(
        JdbcCatalogClient::new(jdbc_uri.clone(), Some(wh_uri.clone()), "bench".to_string()).await?,
    );
    client_a.create_namespace("concurrent").await?;
    let table_loc = format!("{wh_uri}/concurrent/events");
    client_a
        .create_table("concurrent", "events", events_schema(), Some(&table_loc))
        .await?;

    let wal_root = tempdir()?;
    let writers = 4usize;
    let rounds = 3usize;
    let rows_per_round = 4i32;

    let mut handles = Vec::new();
    for w in 0..writers {
        let client: Arc<dyn Catalog> = Arc::new(
            JdbcCatalogClient::new(jdbc_uri.clone(), Some(wh_uri.clone()), "bench".to_string())
                .await?,
        );
        let t = TableBuilder::new(&table_loc)
            .with_catalog(client, "concurrent", "events")
            .with_wal_dir(wal_root.path().join(format!("sqlite_w_{w}")))
            .with_durability(WalDurability::Sync)
            .build_async()
            .await?;

        handles.push(tokio::spawn(async move {
            for r in 0..rounds {
                let start = 1000 + (w as i32) * 100 + (r as i32) * rows_per_round;
                t.write_async(vec![make_events_batch(start, rows_per_round)?])
                    .await?;
                t.commit_async().await?;
            }
            Ok::<_, anyhow::Error>(())
        }));
    }

    for h in handles {
        h.await??;
    }

    let check_wal = tempdir()?;
    let final_table = TableBuilder::new(&table_loc)
        .with_wal_dir(check_wal.path().to_path_buf())
        .build_async()
        .await?;
    assert_eq!(
        read_sorted_ids(&final_table).await?.len(),
        writers * rounds * (rows_per_round as usize)
    );

    let meta = client_a.load_table("concurrent", "events").await?;
    assert_eq!(meta.snapshots.len(), writers * rounds);
    assert!(client_a
        .metadata_location("concurrent", "events")
        .await?
        .is_some_and(|loc| loc.contains("/metadata/v")));

    Ok(())
}

#[tokio::test]
async fn jdbc_postgres_conformance_if_configured() -> Result<()> {
    let Ok(pg_url) =
        std::env::var("BSDB_TEST_POSTGRES_URL").or_else(|_| std::env::var("DATABASE_URL"))
    else {
        return Ok(());
    };
    if !pg_url.starts_with("postgres://") && !pg_url.starts_with("postgresql://") {
        return Ok(());
    }

    let wh_dir = tempdir()?;
    let wh_uri = canon_dir_uri(wh_dir.path());
    let client = JdbcCatalogClient::new(
        pg_url,
        Some(wh_uri.clone()),
        format!("ci_{}", uuid::Uuid::new_v4().simple()),
    )
    .await?;

    run_catalog_contract(
        Arc::new(client),
        &wh_uri,
        ContractCapabilities {
            tracks_empty_namespaces: true,
            enforces_nonempty_drop_namespace: true,
            rejects_duplicate_create: true,
        },
    )
    .await
}

// ----------------------------------------------------------------------------
// 3. Embedded Iceberg REST Catalog Mock (OAuth2 Refresh + 409 CAS Retry)
// ----------------------------------------------------------------------------

#[derive(Default)]
struct MockRestState {
    tables: Mutex<HashMap<(String, String), (String, TableMetadata)>>,
    token_calls: AtomicUsize,
}

#[derive(serde::Deserialize)]
struct MockCreateTablePayload {
    name: String,
    location: Option<String>,
    schema: benostreamdb::core::manifest::Schema,
}

#[derive(serde::Deserialize)]
struct MockCommitPayload {
    updates: Vec<serde_json::Value>,
    #[serde(default)]
    requirements: Vec<serde_json::Value>,
}

async fn mock_oauth_token(State(st): State<Arc<MockRestState>>) -> impl IntoResponse {
    let call_num = st.token_calls.fetch_add(1, Ordering::SeqCst) + 1;
    // First token expires immediately (`expires_in = 0`) to force a live refresh on call #2;
    // subsequent tokens stay valid (`3600s`) so caching is also verified.
    let expires_in = if call_num == 1 { 0 } else { 3600 };
    Json(serde_json::json!({
        "access_token": format!("mock-token-{call_num}"),
        "token_type": "Bearer",
        "expires_in": expires_in,
    }))
}

fn require_bearer(headers: &HeaderMap) -> bool {
    headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("Bearer mock-token-"))
}

async fn mock_create_ns(headers: HeaderMap) -> impl IntoResponse {
    if !require_bearer(&headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    (StatusCode::OK, Json(serde_json::json!({}))).into_response()
}

async fn mock_create_table(
    State(st): State<Arc<MockRestState>>,
    headers: HeaderMap,
    AxumPath((_prefix, ns)): AxumPath<(String, String)>,
    Json(body): Json<MockCreateTablePayload>,
) -> impl IntoResponse {
    if !require_bearer(&headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let key = (ns.clone(), body.name.clone());
    if st.tables.lock().contains_key(&key) {
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "error": { "message": "already exists", "type": "AlreadyExistsException", "code": 409 }
            })),
        )
            .into_response();
    }
    let loc = body
        .location
        .unwrap_or_else(|| format!("memory://mock-rest/{ns}/{}", body.name));
    let arrow_schema = Arc::new(body.schema.to_arrow());
    let meta_loc = match ensure_storage_table_initialized(&loc, &arrow_schema).await {
        Ok(m) => m,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    };
    let mut meta = match load_table_metadata_from_location(&meta_loc).await {
        Ok(m) => m,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    };
    if meta.location.is_empty() {
        meta.location = loc;
    }
    st.tables.lock().insert(key, (meta_loc, meta));
    (StatusCode::OK, Json(serde_json::json!({}))).into_response()
}

async fn mock_list_tables(
    State(st): State<Arc<MockRestState>>,
    headers: HeaderMap,
    AxumPath((_prefix, ns)): AxumPath<(String, String)>,
) -> impl IntoResponse {
    if !require_bearer(&headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let mut names: Vec<String> = st
        .tables
        .lock()
        .keys()
        .filter(|(n, _)| n == &ns)
        .map(|(_, t)| t.clone())
        .collect();
    names.sort();
    let identifiers: Vec<_> = names
        .into_iter()
        .map(|name| serde_json::json!({ "namespace": [ns], "name": name }))
        .collect();
    (
        StatusCode::OK,
        Json(serde_json::json!({ "identifiers": identifiers })),
    )
        .into_response()
}

async fn mock_load_table(
    State(st): State<Arc<MockRestState>>,
    headers: HeaderMap,
    AxumPath((_prefix, ns, table)): AxumPath<(String, String, String)>,
) -> impl IntoResponse {
    if !require_bearer(&headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let guard = st.tables.lock();
    let Some((meta_loc, meta)) = guard.get(&(ns, table)) else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": { "message": "not found", "type": "NoSuchTableException", "code": 404 }
            })),
        )
            .into_response();
    };
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "metadata-location": meta_loc,
            "metadata": meta,
        })),
    )
        .into_response()
}

async fn mock_commit_table(
    State(st): State<Arc<MockRestState>>,
    headers: HeaderMap,
    AxumPath((_prefix, ns, table)): AxumPath<(String, String, String)>,
    Json(body): Json<MockCommitPayload>,
) -> impl IntoResponse {
    if !require_bearer(&headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let mut guard = st.tables.lock();
    let Some((meta_loc, meta)) = guard.get_mut(&(ns.clone(), table.clone())) else {
        let err = catalog_not_found_error("rest", "not found");
        return (StatusCode::NOT_FOUND, err.to_string()).into_response();
    };
    if let Err(err) = validate_requirements_against_metadata(&body.requirements, Some(meta), "rest")
    {
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "error": {
                    "message": err.to_string(),
                    "type": "CommitFailedException",
                    "code": 409
                }
            })),
        )
            .into_response();
    }
    if let Err(err) = apply_table_updates(meta, &body.updates) {
        return (StatusCode::BAD_REQUEST, err.to_string()).into_response();
    }
    if let Some(new_loc) = extract_metadata_location_from_updates(&body.updates) {
        *meta_loc = new_loc;
    }
    (StatusCode::OK, Json(serde_json::json!({}))).into_response()
}

async fn mock_drop_table(
    State(st): State<Arc<MockRestState>>,
    headers: HeaderMap,
    AxumPath((_prefix, ns, table)): AxumPath<(String, String, String)>,
) -> impl IntoResponse {
    if !require_bearer(&headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let removed = st.tables.lock().remove(&(ns, table));
    if removed.is_none() {
        let _ = catalog_conflict_error("rest", "unused");
        return StatusCode::NOT_FOUND.into_response();
    }
    StatusCode::NO_CONTENT.into_response()
}

#[tokio::test]
async fn rest_catalog_mock_oauth2_and_conformance() -> Result<()> {
    let state = Arc::new(MockRestState::default());
    let app = Router::new()
        .route("/v1/oauth/tokens", post(mock_oauth_token))
        .route("/v1/:prefix/namespaces", post(mock_create_ns))
        .route(
            "/v1/:prefix/namespaces/:ns/tables",
            get(mock_list_tables).post(mock_create_table),
        )
        .route(
            "/v1/:prefix/namespaces/:ns/tables/:table",
            get(mock_load_table)
                .post(mock_commit_table)
                .delete(mock_drop_table),
        )
        .with_state(state.clone());

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let server = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    let wh_dir = tempdir()?;
    let wh_uri = canon_dir_uri(wh_dir.path());
    let client = Arc::new(RestCatalogClient::with_oauth2(
        format!("http://{addr}"),
        Some("wh".to_string()),
        None,
        "client".to_string(),
        "secret".to_string(),
        Some("PRINCIPAL_ROLE:ALL".to_string()),
    ));

    run_catalog_contract(
        client,
        &wh_uri,
        ContractCapabilities {
            tracks_empty_namespaces: false,
            enforces_nonempty_drop_namespace: false,
            rejects_duplicate_create: true,
        },
    )
    .await?;

    // First token returned expires_in=0, so the second call refreshed cleanly and cached thereafter.
    assert_eq!(
        state.token_calls.load(Ordering::SeqCst),
        2,
        "expected OAuth2 token endpoint to be called twice (initial + refresh, then cached)"
    );

    server.abort();
    Ok(())
}

// ----------------------------------------------------------------------------
// 4. SQL DDL & Search-Style Pre-Created Storage Compatibility
// ----------------------------------------------------------------------------

#[tokio::test]
async fn sql_ddl_external_catalog_lifecycle_memory_and_jdbc() -> Result<()> {
    // Exercise SQL CREATE SCHEMA / CREATE TABLE / INSERT / SELECT / DROP TABLE / DROP SCHEMA
    // against both Memory and SQLite JDBC catalogs (validating no double-create storage error).
    for kind in ["memory", "jdbc"] {
        let wh_dir = tempdir()?;
        let wh_uri = canon_dir_uri(wh_dir.path());
        let catalog: Arc<dyn Catalog> = match kind {
            "memory" => Arc::new(MemoryCatalogClient::with_warehouse(wh_uri.clone())),
            _ => Arc::new(
                JdbcCatalogClient::new(
                    "sqlite::memory:".to_string(),
                    Some(wh_uri.clone()),
                    "sql_test".to_string(),
                )
                .await?,
            ),
        };

        let mut session = BenoStreamSession::new(None);
        session.set_warehouse(Some(wh_uri.clone()));
        session
            .register_catalog("datafusion", catalog.clone())
            .await;

        session.sql("CREATE SCHEMA analytics").await?;
        assert!(catalog.namespace_exists("analytics").await?);

        session
            .sql("CREATE TABLE analytics.metrics (id INT, val VARCHAR)")
            .await?;
        session
            .sql("CREATE TABLE IF NOT EXISTS analytics.metrics (id INT, val VARCHAR)")
            .await?;
        assert!(catalog.table_exists("analytics", "metrics").await?);

        session
            .sql("INSERT INTO analytics.metrics VALUES (1, 'a'), (2, 'b')")
            .await?;
        let (batches, _) = session
            .sql("SELECT COUNT(*) FROM analytics.metrics")
            .await?;
        let cnt = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .context("expected COUNT(*) Int64Array")?
            .value(0);
        assert_eq!(cnt, 2);

        let meta = catalog.load_table("analytics", "metrics").await?;
        assert!(
            meta.current_snapshot_id.is_some(),
            "SQL INSERT into {kind}-backed table must advance catalog snapshot"
        );

        session.sql("DROP TABLE analytics.metrics").await?;
        assert!(!catalog.table_exists("analytics", "metrics").await?);

        session.sql("DROP SCHEMA analytics").await?;
        assert!(!catalog.namespace_exists("analytics").await?);
    }

    Ok(())
}

#[tokio::test]
async fn search_gateway_precreated_storage_registration_is_idempotent() -> Result<()> {
    // Reproduces `benostreamdb-search`'s exact sequence: `Table::create_async` first, then
    // `catalog.create_table(ns, tbl, schema, Some(&uri))`.
    let wh_dir = tempdir()?;
    let mem_loc = format!("{}/pre/mem_idx", canon_dir_uri(wh_dir.path()));
    let jdbc_loc = format!("{}/pre/jdbc_idx", canon_dir_uri(wh_dir.path()));

    drop(Table::create_async(mem_loc.clone(), events_schema()).await?);
    let mem_cat = MemoryCatalogClient::new();
    mem_cat
        .create_table("pre", "mem_idx", events_schema(), Some(&mem_loc))
        .await?;
    assert!(mem_cat.table_exists("pre", "mem_idx").await?);

    drop(Table::create_async(jdbc_loc.clone(), events_schema()).await?);
    let jdbc_cat = JdbcCatalogClient::new("sqlite::memory:".to_string(), None, "s".into()).await?;
    jdbc_cat
        .create_table("pre", "jdbc_idx", events_schema(), Some(&jdbc_loc))
        .await?;
    assert!(jdbc_cat.table_exists("pre", "jdbc_idx").await?);

    Ok(())
}

#[tokio::test]
async fn nessie_catalog_conformance_if_configured() -> Result<()> {
    let Ok(nessie_uri) = std::env::var("NESSIE_URI") else {
        return Ok(());
    };
    let wh_dir = tempdir()?;
    let wh_uri = canon_dir_uri(wh_dir.path());
    let mut cfg = HashMap::new();
    cfg.insert("url".to_string(), nessie_uri);
    let catalog: Arc<dyn Catalog> =
        Arc::from(create_catalog_async(CatalogType::Nessie, cfg).await?);

    run_catalog_contract(
        catalog,
        &wh_uri,
        ContractCapabilities {
            tracks_empty_namespaces: false,
            enforces_nonempty_drop_namespace: false,
            rejects_duplicate_create: false,
        },
    )
    .await
}
