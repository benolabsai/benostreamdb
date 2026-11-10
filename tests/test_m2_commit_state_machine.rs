// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! M2 PendingCommit transaction state machine certification suite (Pillars 1–9).
//!
//! Validates every crash boundary, store conflict, catalog retry, WAL watermark,
//! V3 row-lineage OCC retry, metadata hint non-regression, and reader/flush
//! linearizability guarantee in `Table`'s explicit `PendingCommit` state machine.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs::File;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use arrow::array::{Array, FixedSizeListArray, Float32Array, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::{Json, Router};
use benostreamdb::core::catalog::rest::RestCatalogClient;
use benostreamdb::core::catalog::Catalog;
use benostreamdb::core::error::BenoStreamError;
use benostreamdb::core::fault_injection::{arm, disarm, CrashPoint};
use benostreamdb::core::index::{VectorMetric, VectorValue};
use benostreamdb::core::manifest::{
    IndexAlgorithm, ManifestEntry, ManifestManager, PartitionSpec, SortOrder,
};
use benostreamdb::core::metadata::{Snapshot, TableMetadata};
use benostreamdb::core::storage::create_object_store;
use benostreamdb::core::table::builder::TableBuilder;
use benostreamdb::core::table::{TableEvent, VectorSearchParams, WalDurability};
use benostreamdb::Table;
use bytes::Bytes;
use futures::stream::BoxStream;
use object_store::local::LocalFileSystem;
use object_store::path::Path as ObjPath;
use object_store::{
    GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore, PutMode,
    PutMultipartOptions, PutOptions, PutPayload, PutResult,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use tempfile::tempdir;
use tokio::io::{AsyncBufReadExt, BufReader as TokioBufReader};
use tokio::sync::Notify;

#[cfg(unix)]
use std::os::unix::process::ExitStatusExt;

// ============================================================================
// Process-global fault-injection lock & RAII guard
// ============================================================================

static FAULT_GATE: tokio::sync::RwLock<()> = tokio::sync::RwLock::const_new(());

async fn shared_no_fault_guard() -> tokio::sync::RwLockReadGuard<'static, ()> {
    FAULT_GATE.read().await
}

async fn exclusive_fault_guard() -> tokio::sync::RwLockWriteGuard<'static, ()> {
    FAULT_GATE.write().await
}

struct CrashGuard;

impl CrashGuard {
    fn arm(point: CrashPoint) -> Self {
        arm(point);
        Self
    }
}

impl Drop for CrashGuard {
    fn drop(&mut self) {
        disarm();
    }
}

// ============================================================================
// Generic schema, batch, URI, and Parquet inspection helpers
// ============================================================================

fn canon_dir_uri(dir: &Path) -> String {
    let canon = std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf());
    format!("file://{}", canon.display())
}

fn scalar_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Utf8, false),
    ]))
}

fn make_scalar_batch(start_id: i32, count: i32) -> Result<RecordBatch> {
    let ids = Int32Array::from_iter_values(start_id..start_id + count);
    let vals =
        StringArray::from_iter_values((start_id..start_id + count).map(|i| format!("v_{i}")));
    Ok(RecordBatch::try_new(
        scalar_schema(),
        vec![Arc::new(ids), Arc::new(vals)],
    )?)
}

fn make_padded_scalar_batch(start_id: i32, count: i32, pad_len: usize) -> Result<RecordBatch> {
    let pad: String = "x".repeat(pad_len);
    let ids = Int32Array::from_iter_values(start_id..start_id + count);
    let vals =
        StringArray::from_iter_values((start_id..start_id + count).map(|i| format!("v_{i}_{pad}")));
    Ok(RecordBatch::try_new(
        scalar_schema(),
        vec![Arc::new(ids), Arc::new(vals)],
    )?)
}

fn vector_schema(dim: i32) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
            false,
        ),
    ]))
}

fn make_vector_batch(dim: i32, rows: &[(i32, Vec<f32>)]) -> Result<RecordBatch> {
    let ids = Int32Array::from_iter_values(rows.iter().map(|(id, _)| *id));
    let mut flat = Vec::with_capacity(rows.len() * (dim as usize));
    for (_, vec) in rows {
        anyhow::ensure!(vec.len() == dim as usize, "vector dim mismatch");
        flat.extend_from_slice(vec);
    }
    let values = Arc::new(Float32Array::from(flat));
    let field = Arc::new(Field::new("item", DataType::Float32, true));
    let fsl = FixedSizeListArray::try_new(field, dim, values, None)?;
    Ok(RecordBatch::try_new(
        vector_schema(dim),
        vec![Arc::new(ids), Arc::new(fsl)],
    )?)
}

async fn read_ids_raw(table: &Table) -> Result<Vec<i32>> {
    let batches = table.read_async(None, None, None).await?;
    let mut out = Vec::new();
    for b in &batches {
        if let Some(col) = b.column_by_name("id") {
            let arr = col
                .as_any()
                .downcast_ref::<Int32Array>()
                .context("expected Int32 id column")?;
            out.extend(arr.iter().flatten());
        }
    }
    Ok(out)
}

async fn read_ids_sorted_unique(table: &Table) -> Result<Vec<i32>> {
    let mut ids = read_ids_raw(table).await?;
    let total = ids.len();
    ids.sort_unstable();
    ids.dedup();
    anyhow::ensure!(
        ids.len() == total,
        "duplicate id detected in table read: total={total}, unique={}",
        ids.len()
    );
    Ok(ids)
}

async fn vector_search_ids(
    table: &Table,
    col: &str,
    query: Vec<f32>,
    k: usize,
) -> Result<Vec<i32>> {
    let params =
        VectorSearchParams::new(col, VectorValue::Float32(query), k).with_metric(VectorMetric::L2);
    let batches = table.read_async(None, Some(vec![params]), None).await?;
    let mut out = Vec::new();
    for b in &batches {
        if let Some(col) = b.column_by_name("id") {
            let arr = col
                .as_any()
                .downcast_ref::<Int32Array>()
                .context("expected Int32 id column in vector search result")?;
            out.extend(arr.iter().flatten());
        }
    }
    Ok(out)
}

fn read_parquet_lineage_rows(
    table_dir: &Path,
    entry: &ManifestEntry,
) -> Result<Vec<(i32, i64, i64)>> {
    let full = table_dir.join(&entry.file_path);
    let file = File::open(&full)
        .with_context(|| format!("failed to open Parquet segment {}", full.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;
    let mut out = Vec::new();
    for batch_res in reader {
        let batch = batch_res?;
        let ids = batch
            .column_by_name("id")
            .context("missing id column")?
            .as_any()
            .downcast_ref::<Int32Array>()
            .context("id not Int32")?;
        let row_ids = batch
            .column_by_name("_row_id")
            .context("missing _row_id column")?
            .as_any()
            .downcast_ref::<Int64Array>()
            .context("_row_id not Int64")?;
        let seqs = batch
            .column_by_name("_last_updated_sequence_number")
            .context("missing _last_updated_sequence_number column")?
            .as_any()
            .downcast_ref::<Int64Array>()
            .context("_last_updated_sequence_number not Int64")?;
        for i in 0..batch.num_rows() {
            out.push((ids.value(i), row_ids.value(i), seqs.value(i)));
        }
    }
    Ok(out)
}

fn list_wal_arrow_files(wal_dir: &Path) -> Result<Vec<PathBuf>> {
    if !wal_dir.exists() {
        return Ok(Vec::new());
    }
    let mut files = Vec::new();
    for entry in std::fs::read_dir(wal_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("arrow") {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

fn collect_parquet_rel_paths(dir: &Path) -> Result<HashSet<String>> {
    fn walk(root: &Path, cur: &Path, out: &mut HashSet<String>) -> Result<()> {
        if !cur.exists() {
            return Ok(());
        }
        for entry in std::fs::read_dir(cur)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
                if name == "_wal" || name == "_staging" || name == "_manifest" || name == "metadata"
                {
                    continue;
                }
                walk(root, &path, out)?;
            } else if path.extension().and_then(|s| s.to_str()) == Some("parquet") {
                if let Ok(rel) = path.strip_prefix(root) {
                    out.insert(rel.to_string_lossy().to_string());
                }
            }
        }
        Ok(())
    }
    let mut out = HashSet::new();
    walk(dir, dir, &mut out)?;
    Ok(out)
}

// ============================================================================
// ScriptedStore: deterministic ObjectStore fault, CAS, & barrier wrapper
// ============================================================================

#[derive(Clone, Debug)]
enum PutAction {
    AmbiguousAckAfterInnerPut,
    FailBeforePutGeneric,
    #[allow(dead_code)]
    FailBeforePutAlreadyExists,
    #[allow(dead_code)]
    FailBeforePutPrecondition,
    AbortProcessAfterInnerPut,
}

#[derive(Clone, Debug)]
struct PutRule {
    path_contains: String,
    payload_contains: Option<String>,
    remaining: usize,
    action: PutAction,
}

struct PauseHandle {
    entered_flag: Arc<AtomicBool>,
    entered: Arc<Notify>,
    released_flag: Arc<AtomicBool>,
    release: Arc<Notify>,
}

impl PauseHandle {
    async fn wait_entered(&self) {
        loop {
            let notified = self.entered.notified();
            if self.entered_flag.load(Ordering::Acquire) {
                return;
            }
            notified.await;
        }
    }

    fn release(&self) {
        self.released_flag.store(true, Ordering::Release);
        self.release.notify_waiters();
    }
}

#[derive(Debug)]
struct PauseRule {
    path_contains: String,
    payload_contains: Option<String>,
    remaining: usize,
    entered_flag: Arc<AtomicBool>,
    entered: Arc<Notify>,
    released_flag: Arc<AtomicBool>,
    release: Arc<Notify>,
}

#[derive(Debug)]
struct ScriptedStore {
    inner: Arc<dyn ObjectStore>,
    emulate_cas: bool,
    rules: parking_lot::Mutex<Vec<PutRule>>,
    barriers: parking_lot::Mutex<Vec<PauseRule>>,
    cas_lock: tokio::sync::Mutex<()>,
    etags: parking_lot::Mutex<HashMap<String, String>>,
    next_etag: AtomicU64,
    put_log: parking_lot::Mutex<Vec<String>>,
}

impl std::fmt::Display for ScriptedStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ScriptedStore({})", self.inner)
    }
}

impl ScriptedStore {
    fn new(inner: Arc<dyn ObjectStore>, emulate_cas: bool) -> Arc<Self> {
        Arc::new(Self {
            inner,
            emulate_cas,
            rules: parking_lot::Mutex::new(Vec::new()),
            barriers: parking_lot::Mutex::new(Vec::new()),
            cas_lock: tokio::sync::Mutex::new(()),
            etags: parking_lot::Mutex::new(HashMap::new()),
            next_etag: AtomicU64::new(1),
            put_log: parking_lot::Mutex::new(Vec::new()),
        })
    }

    fn add_put_rule(&self, path_contains: &str, count: usize, action: PutAction) {
        self.rules.lock().push(PutRule {
            path_contains: path_contains.to_string(),
            payload_contains: None,
            remaining: count,
            action,
        });
    }

    fn add_pause_barrier(
        &self,
        path_contains: &str,
        payload_contains: Option<&str>,
    ) -> PauseHandle {
        let entered_flag = Arc::new(AtomicBool::new(false));
        let entered = Arc::new(Notify::new());
        let released_flag = Arc::new(AtomicBool::new(false));
        let release = Arc::new(Notify::new());

        self.barriers.lock().push(PauseRule {
            path_contains: path_contains.to_string(),
            payload_contains: payload_contains.map(str::to_string),
            remaining: 1,
            entered_flag: entered_flag.clone(),
            entered: entered.clone(),
            released_flag: released_flag.clone(),
            release: release.clone(),
        });

        PauseHandle {
            entered_flag,
            entered,
            released_flag,
            release,
        }
    }

    fn bump_etag(&self, key: &str) -> String {
        let seq = self.next_etag.fetch_add(1, Ordering::SeqCst);
        let tag = format!("scripted-{seq}");
        self.etags.lock().insert(key.to_string(), tag.clone());
        tag
    }

    fn ensure_etag_for_existing(&self, key: &str) -> String {
        let mut guard = self.etags.lock();
        if let Some(existing) = guard.get(key) {
            return existing.clone();
        }
        let seq = self.next_etag.fetch_add(1, Ordering::SeqCst);
        let tag = format!("scripted-{seq}");
        guard.insert(key.to_string(), tag.clone());
        tag
    }

    fn pop_matching_barrier(&self, path: &str, payload_utf8: Option<&str>) -> Option<PauseHandle> {
        let mut barriers = self.barriers.lock();
        let idx = barriers.iter().position(|b| {
            b.remaining > 0
                && path.contains(&b.path_contains)
                && match &b.payload_contains {
                    Some(needle) => payload_utf8.is_some_and(|text| text.contains(needle)),
                    None => true,
                }
        })?;
        let b = &mut barriers[idx];
        b.remaining -= 1;
        let handle = PauseHandle {
            entered_flag: b.entered_flag.clone(),
            entered: b.entered.clone(),
            released_flag: b.released_flag.clone(),
            release: b.release.clone(),
        };
        if b.remaining == 0 {
            barriers.remove(idx);
        }
        Some(handle)
    }

    fn pop_matching_rule(&self, path: &str, payload_utf8: Option<&str>) -> Option<PutAction> {
        let mut rules = self.rules.lock();
        let idx = rules.iter().position(|r| {
            r.remaining > 0
                && path.contains(&r.path_contains)
                && match &r.payload_contains {
                    Some(needle) => payload_utf8.is_some_and(|text| text.contains(needle)),
                    None => true,
                }
        })?;
        let r = &mut rules[idx];
        r.remaining -= 1;
        let action = r.action.clone();
        if r.remaining == 0 {
            rules.remove(idx);
        }
        Some(action)
    }

    async fn wait_barrier_if_matched(&self, path: &str, payload: &PutPayload) {
        let payload_bytes: Vec<u8> = payload
            .as_ref()
            .iter()
            .flat_map(|b| b.as_ref().iter().copied())
            .collect();
        let payload_utf8 = std::str::from_utf8(&payload_bytes).ok();
        if let Some(handle) = self.pop_matching_barrier(path, payload_utf8) {
            handle.entered_flag.store(true, Ordering::Release);
            handle.entered.notify_waiters();
            loop {
                let notified = handle.release.notified();
                if handle.released_flag.load(Ordering::Acquire) {
                    break;
                }
                notified.await;
            }
        }
    }

    async fn execute_put_with_cas(
        &self,
        location: &ObjPath,
        payload: PutPayload,
        opts: Option<PutOptions>,
    ) -> object_store::Result<PutResult> {
        let key = location.to_string();
        if !self.emulate_cas {
            return match opts {
                Some(o) => self.inner.put_opts(location, payload, o).await,
                None => self.inner.put(location, payload).await,
            };
        }

        let _guard = self.cas_lock.lock().await;
        match opts {
            None => {
                let res = self.inner.put(location, payload).await?;
                let tag = self.bump_etag(&key);
                Ok(PutResult {
                    e_tag: res.e_tag.or(Some(tag)),
                    version: res.version,
                })
            }
            Some(o) => match o.mode {
                PutMode::Overwrite => {
                    let res = self.inner.put_opts(location, payload, o).await?;
                    let tag = self.bump_etag(&key);
                    Ok(PutResult {
                        e_tag: res.e_tag.or(Some(tag)),
                        version: res.version,
                    })
                }
                PutMode::Create => {
                    if self.inner.head(location).await.is_ok() {
                        return Err(object_store::Error::AlreadyExists {
                            path: key,
                            source: "ScriptedStore CAS create already exists".into(),
                        });
                    }
                    let res = self.inner.put_opts(location, payload, o).await?;
                    let tag = self.bump_etag(&key);
                    Ok(PutResult {
                        e_tag: res.e_tag.or(Some(tag)),
                        version: res.version,
                    })
                }
                PutMode::Update(expected) => {
                    if self.inner.head(location).await.is_err() {
                        return Err(object_store::Error::Precondition {
                            path: key,
                            source: "ScriptedStore CAS update target missing".into(),
                        });
                    }
                    let current = self.ensure_etag_for_existing(&key);
                    if expected.e_tag.as_deref() != Some(current.as_str()) {
                        return Err(object_store::Error::Precondition {
                            path: key,
                            source: format!(
                                "ScriptedStore CAS ETag mismatch: expected {:?}, current {current}",
                                expected.e_tag
                            )
                            .into(),
                        });
                    }
                    self.inner.put(location, payload).await?;
                    let tag = self.bump_etag(&key);
                    Ok(PutResult {
                        e_tag: Some(tag),
                        version: None,
                    })
                }
            },
        }
    }

    async fn handle_put(
        &self,
        location: &ObjPath,
        payload: PutPayload,
        opts: Option<PutOptions>,
    ) -> object_store::Result<PutResult> {
        let key = location.to_string();
        self.put_log.lock().push(key.clone());

        self.wait_barrier_if_matched(&key, &payload).await;

        let payload_bytes: Vec<u8> = payload
            .as_ref()
            .iter()
            .flat_map(|b| b.as_ref().iter().copied())
            .collect();
        let payload_utf8 = std::str::from_utf8(&payload_bytes).ok();

        if let Some(action) = self.pop_matching_rule(&key, payload_utf8) {
            match action {
                PutAction::FailBeforePutGeneric => {
                    return Err(object_store::Error::Generic {
                        store: "ScriptedStore",
                        source: format!("injected generic put error on {key}").into(),
                    });
                }
                PutAction::FailBeforePutAlreadyExists => {
                    return Err(object_store::Error::AlreadyExists {
                        path: key,
                        source: "injected AlreadyExists".into(),
                    });
                }
                PutAction::FailBeforePutPrecondition => {
                    return Err(object_store::Error::Precondition {
                        path: key,
                        source: "injected Precondition".into(),
                    });
                }
                PutAction::AmbiguousAckAfterInnerPut => {
                    self.execute_put_with_cas(location, payload, opts).await?;
                    return Err(object_store::Error::Generic {
                        store: "ScriptedStore",
                        source: format!("simulated ambiguous timeout after commit on {key}").into(),
                    });
                }
                PutAction::AbortProcessAfterInnerPut => {
                    self.execute_put_with_cas(location, payload, opts).await?;
                    std::process::abort();
                }
            }
        }

        self.execute_put_with_cas(location, payload, opts).await
    }
}

#[async_trait]
impl ObjectStore for ScriptedStore {
    async fn put(
        &self,
        location: &ObjPath,
        payload: PutPayload,
    ) -> object_store::Result<PutResult> {
        self.handle_put(location, payload, None).await
    }

    async fn put_opts(
        &self,
        location: &ObjPath,
        payload: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        self.handle_put(location, payload, Some(opts)).await
    }

    async fn get_opts(
        &self,
        location: &ObjPath,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        let mut res = self.inner.get_opts(location, options).await?;
        if self.emulate_cas {
            let key = location.to_string();
            let tag = self.ensure_etag_for_existing(&key);
            res.meta.e_tag = Some(tag);
        }
        Ok(res)
    }

    async fn put_multipart_opts(
        &self,
        location: &ObjPath,
        opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_range(
        &self,
        location: &ObjPath,
        range: Range<u64>,
    ) -> object_store::Result<Bytes> {
        self.inner.get_range(location, range).await
    }

    async fn head(&self, location: &ObjPath) -> object_store::Result<ObjectMeta> {
        let mut meta = self.inner.head(location).await?;
        if self.emulate_cas {
            let key = location.to_string();
            let tag = self.ensure_etag_for_existing(&key);
            meta.e_tag = Some(tag);
        }
        Ok(meta)
    }

    async fn delete(&self, location: &ObjPath) -> object_store::Result<()> {
        let res = self.inner.delete(location).await;
        if res.is_ok() {
            self.etags.lock().remove(&location.to_string());
        }
        res
    }

    fn list(
        &self,
        prefix: Option<&ObjPath>,
    ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&ObjPath>,
    ) -> object_store::Result<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy(&self, from: &ObjPath, to: &ObjPath) -> object_store::Result<()> {
        self.inner.copy(from, to).await
    }

    async fn copy_if_not_exists(&self, from: &ObjPath, to: &ObjPath) -> object_store::Result<()> {
        self.inner.copy_if_not_exists(from, to).await
    }
}

// ============================================================================
// In-process MockCatalog for Pillar 5
// ============================================================================

struct MockCatalogState {
    meta: TableMetadata,
    metadata_location: String,
    commits: usize,
    fail_next_commits: usize,
    requirements_log: Vec<Vec<serde_json::Value>>,
    updates_log: Vec<Vec<serde_json::Value>>,
}

struct MockCatalog {
    state: parking_lot::Mutex<MockCatalogState>,
}

impl MockCatalog {
    fn new(meta: TableMetadata, metadata_location: String) -> Arc<Self> {
        Arc::new(Self {
            state: parking_lot::Mutex::new(MockCatalogState {
                meta,
                metadata_location,
                commits: 0,
                fail_next_commits: 0,
                requirements_log: Vec::new(),
                updates_log: Vec::new(),
            }),
        })
    }

    fn commits(&self) -> usize {
        self.state.lock().commits
    }

    fn set_fail_next_commits(&self, n: usize) {
        self.state.lock().fail_next_commits = n;
    }

    fn update_count(&self) -> usize {
        self.state.lock().updates_log.len()
    }
}

#[async_trait]
impl Catalog for MockCatalog {
    async fn create_table(
        &self,
        _namespace: &str,
        _table_name: &str,
        _schema: SchemaRef,
        _location: Option<&str>,
    ) -> Result<()> {
        Ok(())
    }

    async fn load_table(&self, _namespace: &str, _table_name: &str) -> Result<TableMetadata> {
        Ok(self.state.lock().meta.clone())
    }

    async fn create_branch(&self, _branch_name: &str, _source_ref: Option<&str>) -> Result<()> {
        Ok(())
    }

    async fn table_exists(&self, _namespace: &str, _table_name: &str) -> Result<bool> {
        Ok(true)
    }

    async fn commit_table(
        &self,
        namespace: &str,
        table_name: &str,
        updates: Vec<serde_json::Value>,
    ) -> Result<()> {
        self.commit_table_with_requirements(namespace, table_name, updates, Vec::new())
            .await
    }

    async fn commit_table_with_requirements(
        &self,
        _namespace: &str,
        _table_name: &str,
        updates: Vec<serde_json::Value>,
        requirements: Vec<serde_json::Value>,
    ) -> Result<()> {
        let mut st = self.state.lock();
        st.requirements_log.push(requirements);
        st.updates_log.push(updates.clone());

        if st.fail_next_commits > 0 {
            st.fail_next_commits -= 1;
            return Err(anyhow::Error::new(BenoStreamError::CatalogRequest {
                catalog: "mock".to_string(),
                status: Some(500),
                reason: "simulated HTTP 500 error from mock catalog".to_string(),
            }));
        }

        for u in &updates {
            let action = u.get("action").and_then(|v| v.as_str()).unwrap_or("");
            match action {
                "set-metadata-location" => {
                    if let Some(loc) = u.get("metadata-location").and_then(|v| v.as_str()) {
                        st.metadata_location = loc.to_string();
                    }
                }
                "add-snapshot" => {
                    if let Some(snap_val) = u.get("snapshot") {
                        let snap: Snapshot = serde_json::from_value(snap_val.clone())?;
                        st.meta.add_snapshot(snap);
                    }
                }
                "set-current-snapshot" => {
                    if let Some(id) = u.get("snapshot-id").and_then(|v| v.as_i64()) {
                        st.meta.current_snapshot_id = Some(id);
                    }
                }
                _ => {}
            }
        }
        st.commits += 1;
        Ok(())
    }
}

// ============================================================================
// Minimal Axum Iceberg REST Catalog Server for Pillar 1B
// ============================================================================

#[derive(Clone, serde::Deserialize)]
struct RestCommitBody {
    updates: Vec<serde_json::Value>,
    #[serde(default)]
    requirements: Vec<serde_json::Value>,
}

struct RestServerState {
    meta: parking_lot::Mutex<TableMetadata>,
    metadata_location: parking_lot::Mutex<String>,
    post_count: AtomicUsize,
    recorded_posts: parking_lot::Mutex<Vec<(Vec<serde_json::Value>, Vec<serde_json::Value>)>>,
    concurrent_snap_id: i64,
}

async fn rest_get_table(
    State(state): State<Arc<RestServerState>>,
    axum::extract::Path((_ns, _table)): axum::extract::Path<(String, String)>,
) -> impl IntoResponse {
    let loc = state.metadata_location.lock().clone();
    let meta = state.meta.lock().clone();
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "metadata-location": loc,
            "metadata": meta,
        })),
    )
}

async fn rest_commit_table(
    State(state): State<Arc<RestServerState>>,
    axum::extract::Path((_ns, _table)): axum::extract::Path<(String, String)>,
    Json(body): Json<RestCommitBody>,
) -> impl IntoResponse {
    let call_idx = state.post_count.fetch_add(1, Ordering::SeqCst);
    state
        .recorded_posts
        .lock()
        .push((body.requirements.clone(), body.updates.clone()));

    if call_idx == 0 {
        let mut meta = state.meta.lock();
        let prev = meta.current_snapshot_id;
        let competitor = Snapshot {
            snapshot_id: state.concurrent_snap_id,
            parent_snapshot_id: prev,
            timestamp_ms: chrono::Utc::now().timestamp_millis(),
            sequence_number: Some(meta.last_sequence_number + 1),
            summary: HashMap::from([("operation".to_string(), "append".to_string())]),
            manifest_list: format!("{}/_manifest/snap-concurrent.avro", meta.location),
            schema_id: Some(meta.current_schema_id),
            first_row_id: None,
            added_rows: None,
        };
        meta.add_snapshot(competitor);
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "error": {
                    "message": format!("CommitFailedException: ref main updated to {}", state.concurrent_snap_id),
                    "type": "CommitFailedException",
                    "code": 409
                }
            })),
        )
            .into_response();
    }

    let mut meta = state.meta.lock();
    for req in &body.requirements {
        let rtype = req.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match rtype {
            "assert-table-uuid" => {
                let uuid = req.get("uuid").and_then(|v| v.as_str()).unwrap_or("");
                if uuid != meta.table_uuid {
                    return (
                        StatusCode::CONFLICT,
                        Json(serde_json::json!({
                            "error": { "message": "uuid mismatch", "type": "CommitFailedException", "code": 409 }
                        })),
                    )
                        .into_response();
                }
            }
            "assert-ref-snapshot-id" => {
                let expected = req.get("snapshot-id").and_then(|v| v.as_i64());
                if expected != meta.current_snapshot_id {
                    return (
                        StatusCode::CONFLICT,
                        Json(serde_json::json!({
                            "error": { "message": "ref snapshot mismatch", "type": "CommitFailedException", "code": 409 }
                        })),
                    )
                        .into_response();
                }
            }
            _ => {}
        }
    }

    for u in &body.updates {
        let action = u.get("action").and_then(|v| v.as_str()).unwrap_or("");
        match action {
            "set-metadata-location" => {
                if let Some(loc) = u.get("metadata-location").and_then(|v| v.as_str()) {
                    *state.metadata_location.lock() = loc.to_string();
                }
            }
            "add-snapshot" => {
                if let Some(snap_val) = u.get("snapshot") {
                    if let Ok(snap) = serde_json::from_value::<Snapshot>(snap_val.clone()) {
                        meta.add_snapshot(snap);
                    }
                }
            }
            "set-current-snapshot" => {
                if let Some(id) = u.get("snapshot-id").and_then(|v| v.as_i64()) {
                    meta.current_snapshot_id = Some(id);
                }
            }
            _ => {}
        }
    }

    (StatusCode::OK, Json(serde_json::json!({}))).into_response()
}

// ============================================================================
// Subprocess Entrypoint (Pillar 2)
// ============================================================================

#[tokio::test]
async fn z_pillar2_subprocess_entrypoint() -> Result<()> {
    let Ok(mode) = std::env::var("BSDB_M2_CHILD_MODE") else {
        return Ok(());
    };
    let uri = std::env::var("BSDB_M2_TABLE_URI").context("missing BSDB_M2_TABLE_URI")?;
    let dir_path = uri.strip_prefix("file://").unwrap_or(&uri);

    match mode.as_str() {
        "abort_after_manifest" => {
            let local = Arc::new(LocalFileSystem::new_with_prefix(dir_path)?);
            let store = ScriptedStore::new(local, true);
            store.add_put_rule("_manifest/v2.json", 1, PutAction::AbortProcessAfterInnerPut);

            let table = TableBuilder::new(&uri)
                .with_store(store)
                .with_durability(WalDurability::Sync)
                .build_async()
                .await?;
            table.write_async(vec![make_scalar_batch(1, 10)?]).await?;
            table.commit_async().await?;
        }
        "abort_before_hint" => {
            let local = Arc::new(LocalFileSystem::new_with_prefix(dir_path)?);
            let store = ScriptedStore::new(local, true);
            store.add_put_rule(
                "metadata/v2.metadata.json",
                1,
                PutAction::AbortProcessAfterInnerPut,
            );

            let table = TableBuilder::new(&uri)
                .with_store(store)
                .with_durability(WalDurability::Sync)
                .build_async()
                .await?;
            table.write_async(vec![make_scalar_batch(1, 10)?]).await?;
            table.commit_async().await?;
        }
        "sigkill_streaming_loop" => {
            let table = TableBuilder::new(&uri)
                .with_durability(WalDurability::Sync)
                .build_async()
                .await?;

            use std::io::Write;
            println!("READY");
            std::io::stdout().flush()?;

            let mut next_id = 1i32;
            let mut batch_num = 0usize;
            loop {
                let count = 200i32;
                let batch = make_padded_scalar_batch(next_id, count, 256)?;
                table.write_async(vec![batch]).await?;
                println!("ACK:{}:{}", next_id, next_id + count);
                std::io::stdout().flush()?;
                next_id += count;
                batch_num += 1;
                if batch_num.is_multiple_of(5) {
                    table.commit_async().await?;
                }
            }
        }
        other => anyhow::bail!("unknown BSDB_M2_CHILD_MODE: {other}"),
    }

    Ok(())
}

// ============================================================================
// Pillar 1A: ScriptedStore CAS, Version Hint, & Ambiguous Manifest/Metadata Ack
// ============================================================================

#[tokio::test]
async fn pillar1_scripted_store_metadata_cas_version_hint_and_ambiguous_manifest_ack() -> Result<()>
{
    let _guard = shared_no_fault_guard().await;
    let dir = tempdir()?;
    let uri = canon_dir_uri(dir.path());

    drop(Table::create_async(uri.clone(), scalar_schema()).await?);

    let local = Arc::new(LocalFileSystem::new_with_prefix(dir.path())?);
    let store = ScriptedStore::new(local, true);
    let table = TableBuilder::new(&uri)
        .with_store(store.clone())
        .with_durability(WalDurability::Sync)
        .build_async()
        .await?;

    // 1. Ambiguous timeout after `_manifest/v2.json` actually landed in the store.
    table.write_async(vec![make_scalar_batch(1, 5)?]).await?;
    store.add_put_rule("_manifest/v2.json", 1, PutAction::AmbiguousAckAfterInnerPut);

    let err1 = table.commit_async().await;
    assert!(
        err1.is_err(),
        "simulated timeout must surface as commit Err"
    );

    let mm = ManifestManager::new(store.clone(), "", &uri);
    let (m2, v2) = mm.load_latest_direct().await?;
    assert_eq!(
        v2, 2,
        "ambiguous commit must still have landed _manifest/v2.json"
    );
    let entries2 = mm.load_all_entries(&m2).await?;
    assert_eq!(entries2.len(), 1);
    assert!(
        dir.path().join(&entries2[0].file_path).exists(),
        "referenced Parquet file must NOT be cleaned up on ambiguous manifest ack"
    );
    assert_eq!(
        table.write_buffer_row_count(),
        5,
        "buffer must stay restored until retry converges"
    );

    // Retry must detect `benostream.committed_wal_tx` on v2.json and converge cleanly
    // without creating `_manifest/v3.json` or writing duplicate Parquet files.
    table.commit_async().await?;
    assert!(
        !dir.path().join("_manifest/v3.json").exists(),
        "idempotent convergence must not advance to _manifest/v3.json"
    );
    assert_eq!(table.write_buffer_row_count(), 0);
    assert_eq!(
        read_ids_sorted_unique(&table).await?,
        (1..=5).collect::<Vec<_>>()
    );

    // 2. Ambiguous timeout after `metadata/v3.metadata.json` actually landed in the store.
    table.write_async(vec![make_scalar_batch(6, 5)?]).await?;
    store.add_put_rule(
        "metadata/v3.metadata.json",
        1,
        PutAction::AmbiguousAckAfterInnerPut,
    );

    let err2 = table.commit_async().await;
    assert!(
        err2.is_err(),
        "simulated metadata timeout must surface as Err"
    );
    assert!(
        dir.path().join("metadata/v3.metadata.json").exists(),
        "ambiguous metadata put must have landed v3.metadata.json"
    );

    // Retry must see snapshot 3 already present in v3.metadata.json (`AlreadyCurrent`),
    // converge cleanly, update version-hint.text to "3", and avoid writing v4.metadata.json.
    table.commit_async().await?;
    assert!(
        !dir.path().join("metadata/v4.metadata.json").exists(),
        "idempotent metadata tail must not write duplicate v4.metadata.json"
    );
    let latest_meta = TableMetadata::load_latest(store.as_ref()).await?;
    assert_eq!(latest_meta.current_snapshot_id, Some(3));
    assert_eq!(
        std::fs::read_to_string(dir.path().join("metadata/version-hint.text"))?.trim(),
        "3"
    );
    assert_eq!(table.write_buffer_row_count(), 0);
    drop(table);

    let reopened = Table::new_async(uri).await?;
    assert_eq!(reopened.write_buffer_row_count(), 0);
    assert_eq!(
        read_ids_sorted_unique(&reopened).await?,
        (1..=10).collect::<Vec<_>>()
    );

    Ok(())
}

// ============================================================================
// Pillar 1B: Axum Iceberg REST Catalog + 409 Conflict + Diverged Location
// ============================================================================

#[tokio::test]
async fn pillar1_axum_rest_catalog_409_conflict_requirements_and_diverged_location() -> Result<()> {
    let _guard = shared_no_fault_guard().await;
    let table_dir = tempdir()?;
    let meta_dir = tempdir()?;
    let table_uri = canon_dir_uri(table_dir.path());
    let meta_uri = canon_dir_uri(meta_dir.path());

    // Seed ONLY `_manifest/v1.json` in `table_dir` (`table_dir/metadata/` never exists).
    let table_store = create_object_store(&table_uri)?;
    let mm = ManifestManager::new(table_store, "", &table_uri);
    let ms = benostreamdb::core::manifest::Schema::from_arrow(&scalar_schema(), 1);
    let max_col_id = ms.fields.iter().map(|f| f.id).max().unwrap_or(0);
    mm.update_schema(vec![ms.clone()], 1, Some(max_col_id))
        .await?;

    // Seed `TableMetadata` in `meta_dir` with snapshot 900.
    let meta_store = create_object_store(&meta_uri)?;
    let table_uuid = uuid::Uuid::new_v4().to_string();
    let mut initial_meta = TableMetadata::new(
        2,
        table_uuid.clone(),
        meta_uri.clone(),
        ms,
        PartitionSpec::default(),
        SortOrder::default(),
    );
    initial_meta.add_snapshot(Snapshot {
        snapshot_id: 900,
        parent_snapshot_id: None,
        timestamp_ms: chrono::Utc::now().timestamp_millis(),
        sequence_number: Some(1),
        summary: HashMap::from([("operation".to_string(), "append".to_string())]),
        manifest_list: format!("{table_uri}/_manifest/snap-900.avro"),
        schema_id: Some(1),
        first_row_id: None,
        added_rows: None,
    });
    let rel_meta_path = initial_meta.save_to_store(meta_store.as_ref(), 1).await?;
    let initial_meta_loc = format!("{meta_uri}/{rel_meta_path}");

    let server_state = Arc::new(RestServerState {
        meta: parking_lot::Mutex::new(initial_meta),
        metadata_location: parking_lot::Mutex::new(initial_meta_loc),
        post_count: AtomicUsize::new(0),
        recorded_posts: parking_lot::Mutex::new(Vec::new()),
        concurrent_snap_id: 901,
    });

    let app = Router::new()
        .route(
            "/v1/namespaces/:ns/tables/:table",
            get(rest_get_table).post(rest_commit_table),
        )
        .with_state(server_state.clone());

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let server_handle = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    let rest_catalog = Arc::new(RestCatalogClient::new(format!("http://{addr}"), None));
    let table = TableBuilder::new(&table_uri)
        .with_index_all(false)
        .with_durability(WalDurability::Sync)
        .with_catalog(rest_catalog, "ns", "tbl")
        .build_async()
        .await?;

    table.write_async(vec![make_scalar_batch(1, 8)?]).await?;
    table.commit_async().await?;
    server_handle.abort();

    assert_eq!(
        server_state.post_count.load(Ordering::SeqCst),
        2,
        "expected HTTP 409 on first POST followed by successful CAS retry on second POST"
    );

    let posts = server_state.recorded_posts.lock().clone();
    let (reqs1, _) = &posts[0];
    let (reqs2, updates2) = &posts[1];

    let req1_ref = reqs1
        .iter()
        .find(|r| r.get("type").and_then(|v| v.as_str()) == Some("assert-ref-snapshot-id"))
        .and_then(|r| r.get("snapshot-id"))
        .and_then(|v| v.as_i64());
    assert_eq!(req1_ref, Some(900));

    let req2_uuid = reqs2
        .iter()
        .find(|r| r.get("type").and_then(|v| v.as_str()) == Some("assert-table-uuid"))
        .and_then(|r| r.get("uuid"))
        .and_then(|v| v.as_str());
    assert_eq!(req2_uuid, Some(table_uuid.as_str()));

    let req2_ref = reqs2
        .iter()
        .find(|r| r.get("type").and_then(|v| v.as_str()) == Some("assert-ref-snapshot-id"))
        .and_then(|r| r.get("snapshot-id"))
        .and_then(|v| v.as_i64());
    assert_eq!(req2_ref, Some(901));

    let committed_meta_loc = updates2
        .iter()
        .find(|u| u.get("action").and_then(|v| v.as_str()) == Some("set-metadata-location"))
        .and_then(|u| u.get("metadata-location"))
        .and_then(|v| v.as_str())
        .context("missing set-metadata-location update")?;
    assert!(
        committed_meta_loc.starts_with(&format!("{meta_uri}/metadata/")),
        "metadata-location must resolve into diverged meta_dir, got {committed_meta_loc}"
    );

    let parent_snap = updates2
        .iter()
        .find(|u| u.get("action").and_then(|v| v.as_str()) == Some("add-snapshot"))
        .and_then(|u| u.get("snapshot"))
        .and_then(|s| s.get("parent-snapshot-id"))
        .and_then(|v| v.as_i64());
    assert_eq!(parent_snap, Some(901));

    assert!(
        !table_dir.path().join("metadata").exists(),
        "metadata/ must NEVER be created under table_dir when catalog specifies diverged location"
    );
    assert!(meta_dir.path().join("metadata/v2.metadata.json").exists());

    drop(table);
    let reopened = Table::new_async(table_uri).await?;
    assert_eq!(
        read_ids_sorted_unique(&reopened).await?,
        (1..=8).collect::<Vec<_>>()
    );

    Ok(())
}

// ============================================================================
// Pillar 2A: Real Subprocess SIGABRT Boundaries
// ============================================================================

#[tokio::test]
async fn pillar2_subprocess_sigabrt_boundaries() -> Result<()> {
    let _guard = shared_no_fault_guard().await;
    let exe = std::env::current_exe()?;

    // Sub-case A: abort immediately after `_manifest/v2.json` is written (before metadata/WAL prune).
    {
        let dir = tempdir()?;
        let uri = canon_dir_uri(dir.path());
        drop(Table::create_async(uri.clone(), scalar_schema()).await?);

        let status = std::process::Command::new(&exe)
            .args(["--exact", "z_pillar2_subprocess_entrypoint", "--nocapture"])
            .env("BSDB_M2_CHILD_MODE", "abort_after_manifest")
            .env("BSDB_M2_TABLE_URI", &uri)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()?;

        assert!(!status.success(), "child process must have aborted");
        #[cfg(unix)]
        assert_eq!(
            status.signal(),
            Some(6),
            "expected SIGABRT (signal 6), got status {status:?}"
        );

        assert!(dir.path().join("_manifest/v2.json").exists());
        assert!(!dir.path().join("metadata/v2.metadata.json").exists());
        assert!(
            !list_wal_arrow_files(&dir.path().join("_wal"))?.is_empty(),
            "unpruned WAL segment must still exist on disk after crash before WalTruncate"
        );

        let reopened = Table::new_async(uri.clone()).await?;
        assert_eq!(
            reopened.write_buffer_row_count(),
            0,
            "committed tx in v2.json must be skipped during WAL replay"
        );
        assert_eq!(
            read_ids_sorted_unique(&reopened).await?,
            (1..=10).collect::<Vec<_>>()
        );

        reopened
            .write_async(vec![make_scalar_batch(11, 5)?])
            .await?;
        reopened.commit_async().await?;
        assert_eq!(
            read_ids_sorted_unique(&reopened).await?,
            (1..=15).collect::<Vec<_>>()
        );
        let store = create_object_store(&uri)?;
        let meta = TableMetadata::load_latest(store.as_ref()).await?;
        assert_eq!(meta.current_snapshot_id, Some(3));
    }

    // Sub-case B: abort immediately after `metadata/v2.metadata.json` is written (before version-hint.text).
    {
        let dir = tempdir()?;
        let uri = canon_dir_uri(dir.path());
        drop(Table::create_async(uri.clone(), scalar_schema()).await?);

        let status = std::process::Command::new(&exe)
            .args(["--exact", "z_pillar2_subprocess_entrypoint", "--nocapture"])
            .env("BSDB_M2_CHILD_MODE", "abort_before_hint")
            .env("BSDB_M2_TABLE_URI", &uri)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()?;

        assert!(!status.success(), "child process must have aborted");
        #[cfg(unix)]
        assert_eq!(
            status.signal(),
            Some(6),
            "expected SIGABRT (signal 6), got status {status:?}"
        );

        assert!(dir.path().join("metadata/v2.metadata.json").exists());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("metadata/version-hint.text"))?.trim(),
            "1",
            "version-hint.text must still lag at 1 before advance_version_hint ran"
        );

        let store = create_object_store(&uri)?;
        let meta = TableMetadata::load_latest(store.as_ref()).await?;
        assert_eq!(
            meta.current_snapshot_id,
            Some(2),
            "TableMetadata::load_latest must discover v2 despite stale hint=1"
        );

        let reopened = Table::new_async(uri.clone()).await?;
        assert_eq!(reopened.write_buffer_row_count(), 0);
        assert_eq!(
            read_ids_sorted_unique(&reopened).await?,
            (1..=10).collect::<Vec<_>>()
        );

        reopened
            .write_async(vec![make_scalar_batch(11, 5)?])
            .await?;
        reopened.commit_async().await?;
        assert_eq!(
            std::fs::read_to_string(dir.path().join("metadata/version-hint.text"))?.trim(),
            "3"
        );
        assert_eq!(
            read_ids_sorted_unique(&reopened).await?,
            (1..=15).collect::<Vec<_>>()
        );
    }

    Ok(())
}

// ============================================================================
// Pillar 2B: Real Subprocess SIGKILL Under Sustained WalDurability::Sync Writes
// ============================================================================

#[tokio::test]
async fn pillar2_subprocess_sigkill_sustained_sync_writes() -> Result<()> {
    let _guard = shared_no_fault_guard().await;
    let exe = std::env::current_exe()?;
    let dir = tempdir()?;
    let uri = canon_dir_uri(dir.path());
    drop(Table::create_async(uri.clone(), scalar_schema()).await?);

    let mut child = tokio::process::Command::new(&exe)
        .args(["--exact", "z_pillar2_subprocess_entrypoint", "--nocapture"])
        .env("BSDB_M2_CHILD_MODE", "sigkill_streaming_loop")
        .env("BSDB_M2_TABLE_URI", &uri)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;

    let stdout = child.stdout.take().context("missing child stdout")?;
    let mut reader = TokioBufReader::new(stdout).lines();

    let mut acked_ids: BTreeSet<i32> = BTreeSet::new();
    let mut acked_batches = 0usize;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);

    while tokio::time::Instant::now() < deadline {
        let next_line = tokio::time::timeout(Duration::from_secs(10), reader.next_line()).await?;
        match next_line? {
            Some(line) if line.starts_with("ACK:") => {
                let parts: Vec<&str> = line.trim().split(':').collect();
                if parts.len() == 3 {
                    let start: i32 = parts[1].parse()?;
                    let end: i32 = parts[2].parse()?;
                    for id in start..end {
                        acked_ids.insert(id);
                    }
                    acked_batches += 1;
                    // Stop right in the middle of batch 13+ after periodic commits ran at 5 & 10,
                    // ensuring both manifest commits AND active WAL fsynced + in-flight writes exist.
                    if acked_batches >= 12 {
                        break;
                    }
                }
            }
            Some(_) => {}
            None => anyhow::bail!("child closed stdout unexpectedly"),
        }
    }

    assert!(
        acked_batches >= 12,
        "child did not acknowledge enough batches before SIGKILL"
    );

    // Let the next batch write start and potentially sit mid-IPC-write, then deliver SIGKILL.
    tokio::time::sleep(Duration::from_millis(2)).await;
    child.kill().await?;
    let status = child.wait().await?;
    #[cfg(unix)]
    assert_eq!(
        status.signal(),
        Some(9),
        "expected SIGKILL (signal 9), got {status:?}"
    );

    let reopened = Table::new_async(uri.clone()).await?;
    let recovered: BTreeSet<i32> = read_ids_sorted_unique(&reopened)
        .await?
        .into_iter()
        .collect();

    for id in &acked_ids {
        assert!(
            recovered.contains(id),
            "acked WalDurability::Sync row id={id} lost after SIGKILL"
        );
    }

    reopened.commit_async().await?;
    drop(reopened);

    let reopened2 = Table::new_async(uri).await?;
    assert_eq!(reopened2.write_buffer_row_count(), 0);
    let recovered2: BTreeSet<i32> = read_ids_sorted_unique(&reopened2)
        .await?
        .into_iter()
        .collect();
    assert_eq!(
        recovered, recovered2,
        "recovered row set must remain identical across commit and reopen"
    );

    Ok(())
}

#[tokio::test]
async fn pillar2_wal_truncated_eof_tail_preserves_acked_sync_batches() -> Result<()> {
    let _guard = shared_no_fault_guard().await;
    let dir = tempdir()?;
    let uri = canon_dir_uri(dir.path());

    {
        let table = TableBuilder::new(&uri)
            .with_durability(WalDurability::Sync)
            .build_async()
            .await?;

        // Batch 1 (1..=5) committed cleanly to manifest, pruning its sealed segment.
        table.write_async(vec![make_scalar_batch(1, 5)?]).await?;
        table.commit_async().await?;

        // Batch 2 (6..=10) acknowledged synchronously via WalDurability::Sync (uncommitted).
        table.write_async(vec![make_scalar_batch(6, 5)?]).await?;
        drop(table);
    }

    // Wait for the dropped Table's background WAL worker to write its 8-byte EOS marker
    // and close the segment fd before simulating an ungraceful truncated trailing frame.
    let wal_dir = dir.path().join("_wal");
    let eos_marker: [u8; 8] = [0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0];
    let seg_path = {
        let mut found = None;
        for _ in 0..400 {
            let files = list_wal_arrow_files(&wal_dir)?;
            if let Some(p) = files.first() {
                let bytes = std::fs::read(p)?;
                if bytes.ends_with(&eos_marker) {
                    found = Some(p.clone());
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        found.context("timed out waiting for closed WAL segment")?
    };

    // Strip EOS and append half of an encoded Arrow IPC record-batch message frame.
    let mut seg_bytes = std::fs::read(&seg_path)?;
    assert!(seg_bytes.ends_with(&eos_marker));
    seg_bytes.truncate(seg_bytes.len() - eos_marker.len());

    let extra_batch = make_scalar_batch(11, 20)?;
    let mut header_only = Vec::new();
    {
        let _w =
            arrow::ipc::writer::StreamWriter::try_new(&mut header_only, &extra_batch.schema())?;
    }
    let mut extra_stream = Vec::new();
    {
        let mut w =
            arrow::ipc::writer::StreamWriter::try_new(&mut extra_stream, &extra_batch.schema())?;
        w.write(&extra_batch)?;
    }
    let frame_bytes = &extra_stream[header_only.len()..];
    anyhow::ensure!(frame_bytes.len() > 16, "expected multi-byte IPC frame");
    seg_bytes.extend_from_slice(&frame_bytes[..frame_bytes.len() / 2]);
    std::fs::write(&seg_path, &seg_bytes)?;

    // Also leave a sibling torn-header stub segment (crashed before schema finished writing).
    std::fs::write(
        wal_dir.join("log_99999999999999999999_000001_00000001.arrow"),
        [0xFF, 0xFF, 0xFF, 0xFF, 0x30, 0x00],
    )?;

    let reopened = Table::new_async(uri.clone()).await?;
    assert_eq!(
        reopened.write_buffer_row_count(),
        5,
        "synced batch (6..=10) must survive truncated trailing batch and sibling torn header"
    );
    assert_eq!(
        read_ids_sorted_unique(&reopened).await?,
        (1..=10).collect::<Vec<_>>()
    );

    reopened.commit_async().await?;
    assert!(
        list_wal_arrow_files(&wal_dir)?.is_empty(),
        "commit_async must watermark-prune both truncated-tail and torn-header WAL segments"
    );
    drop(reopened);

    let reopened2 = Table::new_async(uri).await?;
    assert_eq!(reopened2.write_buffer_row_count(), 0);
    assert_eq!(
        read_ids_sorted_unique(&reopened2).await?,
        (1..=10).collect::<Vec<_>>()
    );

    Ok(())
}

// ============================================================================
// Pillar 3: H1 Same-Handle WAL Rotation & Reuse
// ============================================================================

#[tokio::test]
async fn pillar3_wal_rotate_same_handle_reuse() -> Result<()> {
    let _guard = shared_no_fault_guard().await;
    let dir = tempdir()?;
    let uri = canon_dir_uri(dir.path());
    let table = Table::create_async(uri.clone(), scalar_schema()).await?;
    let mut table = Some(table);
    let t = table.as_ref().context("table present")?;

    for round in 0..6i32 {
        let start = round * 10 + 1;
        t.write_async(vec![make_scalar_batch(start, 10)?]).await?;
        t.commit_async().await?;

        assert_eq!(
            t.write_buffer_row_count(),
            0,
            "write buffer must empty after commit on round {round}"
        );
        let ids = read_ids_sorted_unique(t).await?;
        let expected: Vec<i32> = (1..=(round + 1) * 10).collect();
        assert_eq!(ids, expected, "unexpected ids after round {round}");
    }

    let wal_files = list_wal_arrow_files(&dir.path().join("_wal"))?;
    assert!(
        wal_files.is_empty(),
        "committed WAL segments must be pruned on same-handle commits, found {wal_files:?}"
    );

    drop(table.take());
    let reopened = Table::new_async(uri).await?;
    assert_eq!(reopened.write_buffer_row_count(), 0);
    assert_eq!(
        read_ids_sorted_unique(&reopened).await?,
        (1..=60).collect::<Vec<_>>()
    );

    Ok(())
}

// ============================================================================
// Pillar 4: Deterministic WAL Watermark Barrier & Concurrent Stress
// ============================================================================

#[tokio::test]
async fn pillar4_deterministic_rotate_watermark_barrier() -> Result<()> {
    let _guard = shared_no_fault_guard().await;
    let dir = tempdir()?;
    let uri = canon_dir_uri(dir.path());
    drop(Table::create_async(uri.clone(), scalar_schema()).await?);

    let local = Arc::new(LocalFileSystem::new_with_prefix(dir.path())?);
    let store = ScriptedStore::new(local, true);
    let table = TableBuilder::new(&uri)
        .with_store(store.clone())
        .with_durability(WalDurability::Sync)
        .build_async()
        .await?;

    // Write Batch A (ids 1..=10).
    table.write_async(vec![make_scalar_batch(1, 10)?]).await?;

    // Pause commit *after* PendingCommit::Staging + wal.rotate_async(), right before manifest snap write.
    let barrier = store.add_pause_barrier("_manifest/snap-", None);
    let t_commit = table.clone();
    let commit_task = tokio::spawn(async move { t_commit.commit_async().await });

    barrier.wait_entered().await;

    // Concurrent write of Batch B (ids 11..=20) lands in the rotated active WAL + pending_writes.
    table.write_async(vec![make_scalar_batch(11, 10)?]).await?;

    // Reader must see both Staging (1..=10) and active pending_writes (11..=20) without gaps/dups.
    assert_eq!(
        read_ids_sorted_unique(&table).await?,
        (1..=20).collect::<Vec<_>>()
    );

    barrier.release();
    commit_task.await??;

    // Batch B remains buffered in memory and in an active unpruned WAL file on disk.
    assert_eq!(table.write_buffer_row_count(), 10);
    let remaining_wal = list_wal_arrow_files(&dir.path().join("_wal"))?;
    assert!(
        !remaining_wal.is_empty(),
        "Batch B active WAL segment must NOT be pruned when Batch A commits"
    );

    // Simulate abrupt close before Batch B commits.
    drop(table);

    let reopened = Table::new_async(uri).await?;
    assert_eq!(
        reopened.write_buffer_row_count(),
        10,
        "Batch B must recover into write buffer from surviving active WAL segment"
    );
    assert_eq!(
        read_ids_sorted_unique(&reopened).await?,
        (1..=20).collect::<Vec<_>>()
    );

    reopened.commit_async().await?;
    assert_eq!(reopened.write_buffer_row_count(), 0);
    assert_eq!(
        read_ids_sorted_unique(&reopened).await?,
        (1..=20).collect::<Vec<_>>()
    );

    Ok(())
}

#[tokio::test]
async fn pillar4_concurrent_writer_committer_watermark_stress() -> Result<()> {
    let _guard = shared_no_fault_guard().await;
    let dir = tempdir()?;
    let uri = canon_dir_uri(dir.path());
    let table = Arc::new(Table::create_async(uri.clone(), scalar_schema()).await?);

    let done_writers = Arc::new(AtomicBool::new(false));
    let mut writer_handles = Vec::new();
    for w in 0..4i32 {
        let t = table.clone();
        writer_handles.push(tokio::spawn(async move {
            for r in 0..25i32 {
                let start = w * 1000 + r * 5 + 1;
                t.write_async(vec![make_scalar_batch(start, 5)?]).await?;
                if r % 3 == 0 {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            }
            Ok::<_, anyhow::Error>(())
        }));
    }

    let t_committer = table.clone();
    let done_flag = done_writers.clone();
    let committer = tokio::spawn(async move {
        while !done_flag.load(Ordering::Acquire) {
            t_committer.commit_async().await?;
            tokio::time::sleep(Duration::from_millis(3)).await;
        }
        Ok::<_, anyhow::Error>(())
    });

    for h in writer_handles {
        h.await??;
    }
    done_writers.store(true, Ordering::Release);
    committer.await??;

    // Commit any final trailing writes, drop WITHOUT shutting down cleanly, then reopen.
    table.commit_async().await?;
    assert_eq!(table.write_buffer_row_count(), 0);
    let expected: Vec<i32> = (0..4i32)
        .flat_map(|w| (1..=125i32).map(move |off| w * 1000 + off))
        .collect();
    assert_eq!(read_ids_sorted_unique(&table).await?, expected);

    drop(table);
    let reopened = Table::new_async(uri).await?;
    assert_eq!(reopened.write_buffer_row_count(), 0);
    assert_eq!(read_ids_sorted_unique(&reopened).await?, expected);

    Ok(())
}

// ============================================================================
// Pillar 5: In-Process PendingCommit Tail Resume, Subscribers, & MockCatalog
// ============================================================================

#[tokio::test]
async fn pillar5_in_process_tail_resume_subscriber_and_catalog() -> Result<()> {
    let _guard = exclusive_fault_guard().await;
    let dir = tempdir()?;
    let uri = canon_dir_uri(dir.path());

    // Create initial table & metadata, then seed MockCatalog from landed TableMetadata.
    {
        let dummy = Table::create_async(uri.clone(), scalar_schema()).await?;
        drop(dummy);
    }
    let store = create_object_store(&uri)?;
    let initial_meta = TableMetadata::load_latest(store.as_ref()).await?;
    let initial_loc = format!("{uri}/metadata/v1.metadata.json");
    let catalog = MockCatalog::new(initial_meta, initial_loc);

    let table = TableBuilder::new(&uri)
        .with_index_all(false)
        .with_durability(WalDurability::Sync)
        .with_catalog(catalog.clone(), "ns", "tbl")
        .build_async()
        .await?;

    let mut sub = table.subscribe();
    table.write_async(vec![make_scalar_batch(1, 10)?]).await?;

    // 1. Fail after manifest commit, before metadata/catalog/WAL prune/notify.
    {
        let _cg = CrashGuard::arm(CrashPoint::ManifestVisible);
        assert!(table.commit_async().await.is_err());
    }
    assert!(dir.path().join("_manifest/v2.json").exists());
    assert!(!dir.path().join("metadata/v2.metadata.json").exists());
    assert_eq!(catalog.commits(), 0);
    assert!(
        sub.try_recv()?.is_none(),
        "subscriber must not fire prematurely"
    );
    assert_eq!(
        read_ids_sorted_unique(&table).await?,
        (1..=10).collect::<Vec<_>>(),
        "reader must not double-count ManifestCommitted rows"
    );

    // 2. Resume tail and fail at MetadataSave.
    {
        let _cg = CrashGuard::arm(CrashPoint::MetadataSave);
        assert!(table.commit_async().await.is_err());
    }
    assert!(!dir.path().join("metadata/v2.metadata.json").exists());
    assert_eq!(catalog.commits(), 0);
    assert!(sub.try_recv()?.is_none());

    // 3. Resume tail: MetadataSave succeeds and caches location/snapshot, then CatalogCommit fires.
    {
        let _cg = CrashGuard::arm(CrashPoint::CatalogCommit);
        assert!(table.commit_async().await.is_err());
    }
    assert!(dir.path().join("metadata/v2.metadata.json").exists());
    assert_eq!(catalog.commits(), 0);
    assert_eq!(catalog.update_count(), 0);

    // 4. Resume tail: reuse cached metadata_location (no v3.metadata.json!), hit catalog HTTP 500.
    catalog.set_fail_next_commits(1);
    let err_500 = table
        .commit_async()
        .await
        .expect_err("expected HTTP 500 error");
    assert!(err_500.to_string().contains("500") || err_500.to_string().contains("mock"));
    assert!(
        !dir.path().join("metadata/v3.metadata.json").exists(),
        "tail resume after cached MetadataSave must reuse v2.metadata.json rather than rewriting v3"
    );
    assert_eq!(catalog.commits(), 0);
    assert!(sub.try_recv()?.is_none());

    // 5. Resume tail: catalog commit succeeds (IcebergPublished), then WalTruncate aborts before WAL prune.
    {
        let _cg = CrashGuard::arm(CrashPoint::WalTruncate);
        assert!(table.commit_async().await.is_err());
    }
    assert_eq!(catalog.commits(), 1);
    assert!(
        !list_wal_arrow_files(&dir.path().join("_wal"))?.is_empty(),
        "WAL must remain intact before WalTruncate succeeds"
    );
    assert!(sub.try_recv()?.is_none());

    // 6. Final resume: skips metadata & catalog steps (`IcebergPublished`), prunes WAL, fires subscribers ONCE.
    table.commit_async().await?;
    assert_eq!(
        catalog.commits(),
        1,
        "IcebergPublished resume must NOT double-commit catalog"
    );
    assert_eq!(catalog.update_count(), 2); // failed 500 + 1 succeeded commit
    assert!(
        list_wal_arrow_files(&dir.path().join("_wal"))?.is_empty(),
        "WAL must be pruned once WalTruncate succeeds"
    );

    let mut received_rows = 0usize;
    let mut commit_markers = Vec::new();
    while let Some(ev) = sub.try_recv()? {
        match ev {
            TableEvent::Batch(b) => received_rows += b.num_rows(),
            TableEvent::Commit { rows } => commit_markers.push(rows),
        }
    }
    assert_eq!(received_rows, 10);
    assert_eq!(commit_markers, vec![10]);

    // Subsequent empty commit must NOT emit duplicate subscriber notifications.
    table.commit_async().await?;
    assert!(sub.try_recv()?.is_none());
    assert_eq!(table.write_buffer_row_count(), 0);

    drop(table);
    let reopened = Table::new_async(uri).await?;
    assert_eq!(reopened.write_buffer_row_count(), 0);
    assert_eq!(
        read_ids_sorted_unique(&reopened).await?,
        (1..=10).collect::<Vec<_>>()
    );

    Ok(())
}

// ============================================================================
// Pillar 6: Pre-Manifest Rollback Restores Buffer, Memory Vector Index, & PKs
// ============================================================================

#[tokio::test]
async fn pillar6_pre_manifest_failure_restores_buffer_index_and_pk() -> Result<()> {
    let _guard = exclusive_fault_guard().await;
    let dir = tempdir()?;
    let uri = canon_dir_uri(dir.path());

    // Create vector table, register primary key and HNSW index before attaching ScriptedStore.
    {
        let init = Table::create_async(uri.clone(), vector_schema(4)).await?;
        init.set_primary_key_async(vec!["id".to_string()]).await?;
        init.add_index("embedding".to_string(), IndexAlgorithm::hnsw())
            .await?;
        init.wait_for_background_tasks_async().await?;
        drop(init);
    }

    let local = Arc::new(LocalFileSystem::new_with_prefix(dir.path())?);
    let store = ScriptedStore::new(local, true);
    let table = TableBuilder::new(&uri)
        .with_store(store.clone())
        .with_durability(WalDurability::Sync)
        .build_async()
        .await?;

    let initial_batch = make_vector_batch(
        4,
        &[
            (1, vec![1.0, 0.0, 0.0, 0.0]),
            (2, vec![0.0, 1.0, 0.0, 0.0]),
            (3, vec![0.0, 0.0, 1.0, 0.0]),
        ],
    )?;
    table.write_async(vec![initial_batch]).await?;

    // Check WalRotate, DataUpload, and ManifestCommit rollbacks.
    for cp in [
        CrashPoint::WalRotate,
        CrashPoint::DataUpload,
        CrashPoint::ManifestCommit,
    ] {
        {
            let _cg = CrashGuard::arm(cp);
            assert!(
                table.commit_async().await.is_err(),
                "expected {cp} crash point to abort commit"
            );
        }
        assert!(
            collect_parquet_rel_paths(dir.path())?.is_empty(),
            "orphan Parquet segment left behind after {cp}"
        );
        assert_eq!(table.write_buffer_row_count(), 3);
        assert!(table.has_memory_index());
        let top = vector_search_ids(&table, "embedding", vec![0.0, 0.0, 1.0, 0.0], 1).await?;
        assert_eq!(top, vec![3], "memory vector search broken after {cp}");

        let dup_batch = make_vector_batch(4, &[(3, vec![0.5, 0.5, 0.0, 0.0])])?;
        let dup_err = table
            .write_async(vec![dup_batch])
            .await
            .expect_err("duplicate PK must be rejected after rollback");
        assert!(
            matches!(
                dup_err.downcast_ref::<BenoStreamError>(),
                Some(BenoStreamError::PrimaryKeyViolation { .. })
            ),
            "expected PrimaryKeyViolation after {cp}, got {dup_err:?}"
        );
    }

    // Concurrent write during `Staging` + store failure on the next manifest version.
    let next_ver = table.snapshot_version().await? + 1;
    let barrier = store.add_pause_barrier("_manifest/snap-", None);
    store.add_put_rule(
        &format!("_manifest/v{next_ver}.json"),
        1,
        PutAction::FailBeforePutGeneric,
    );

    let t_commit = table.clone();
    let commit_handle = tokio::spawn(async move { t_commit.commit_async().await });
    barrier.wait_entered().await;

    // Concurrent write arrives while `[1, 2, 3]` is in `PendingCommit::Staging`.
    let concurrent_batch = make_vector_batch(4, &[(6, vec![0.0, 0.0, 0.0, 1.0])])?;
    table.write_async(vec![concurrent_batch]).await?;

    barrier.release();
    assert!(
        commit_handle.await?.is_err(),
        "manifest failure must unwind Staging back to write buffer"
    );

    assert_eq!(table.write_buffer_row_count(), 4);
    assert!(table.has_memory_index());
    assert_eq!(
        vector_search_ids(&table, "embedding", vec![1.0, 0.0, 0.0, 0.0], 1).await?,
        vec![1]
    );
    assert_eq!(
        vector_search_ids(&table, "embedding", vec![0.0, 0.0, 0.0, 1.0], 1).await?,
        vec![6]
    );

    for existing_pk in [3, 6] {
        let dup = make_vector_batch(4, &[(existing_pk, vec![0.2, 0.2, 0.2, 0.2])])?;
        let err = table
            .write_async(vec![dup])
            .await
            .expect_err("both restored and concurrent PKs must remain enforced");
        assert!(matches!(
            err.downcast_ref::<BenoStreamError>(),
            Some(BenoStreamError::PrimaryKeyViolation { .. })
        ));
    }

    // Clean retry converges everything cleanly.
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;
    assert_eq!(table.write_buffer_row_count(), 0);
    assert_eq!(read_ids_sorted_unique(&table).await?, vec![1, 2, 3, 6]);

    drop(table);
    let reopened = Table::new_async(uri).await?;
    assert_eq!(reopened.write_buffer_row_count(), 0);
    assert_eq!(read_ids_sorted_unique(&reopened).await?, vec![1, 2, 3, 6]);

    Ok(())
}

// ============================================================================
// Pillar 7: Concurrent V3 Row Lineage OCC Retries (No Gaps, No Dups, No Orphans)
// ============================================================================

#[tokio::test]
async fn pillar7_concurrent_v3_row_lineage_no_dups_no_gaps_no_orphans() -> Result<()> {
    let _guard = shared_no_fault_guard().await;
    let dir = tempdir()?;
    let uri = canon_dir_uri(dir.path());

    let local = Arc::new(LocalFileSystem::new_with_prefix(dir.path())?);
    let shared_store = ScriptedStore::new(local, true);

    // Seed v3 format table with 4 rows (`_row_id` 0..4).
    {
        let init = Table::create_async(uri.clone(), scalar_schema()).await?;
        init.set_format_version(3);
        init.write_async(vec![make_scalar_batch(1, 4)?]).await?;
        init.commit_async().await?;
    }

    let wal_root = tempdir()?;
    let writers = 6usize;
    let rounds = 4usize;
    let rows_per_round = 5i32;

    let mut handles = Vec::new();
    for w in 0..writers {
        let t = TableBuilder::new(&uri)
            .with_store(shared_store.clone())
            .with_wal_dir(wal_root.path().join(format!("wal_{w}")))
            .with_durability(WalDurability::Sync)
            .build_async()
            .await?;
        handles.push(tokio::spawn(async move {
            for r in 0..rounds {
                let start = 1000 + (w as i32) * 100 + (r as i32) * rows_per_round;
                t.write_async(vec![make_scalar_batch(start, rows_per_round)?])
                    .await?;
                t.commit_async().await?;
            }
            Ok::<_, anyhow::Error>(())
        }));
    }

    for h in handles {
        h.await??;
    }

    let expected_total_rows = 4 + (writers * rounds * rows_per_round as usize);
    let mm = ManifestManager::new(shared_store.clone(), "", &uri);
    let (manifest, entries, _ver) = mm.load_latest_full().await?;
    assert_eq!(manifest.format_version, 3);

    let mut all_row_ids = BTreeSet::new();
    let mut all_ids = BTreeSet::new();
    let mut referenced_parquet = HashSet::new();

    for entry in &entries {
        referenced_parquet.insert(entry.file_path.clone());
        let first_row_id = entry
            .first_row_id
            .context("V3 manifest entry missing first_row_id")?;
        let rows = read_parquet_lineage_rows(dir.path(), entry)?;
        assert_eq!(rows.len() as i64, entry.record_count);
        for (pos, (id, row_id, seq)) in rows.into_iter().enumerate() {
            assert_eq!(
                row_id,
                first_row_id + pos as i64,
                "Parquet _row_id mismatch in {}",
                entry.file_path
            );
            assert!(seq > 0, "_last_updated_sequence_number must be > 0");
            assert!(
                all_row_ids.insert(row_id),
                "duplicate _row_id={row_id} detected across concurrent V3 writers"
            );
            assert!(all_ids.insert(id), "duplicate data id={id}");
        }
    }

    let expected_row_ids: BTreeSet<i64> = (0..expected_total_rows as i64).collect();
    assert_eq!(
        all_row_ids, expected_row_ids,
        "V3 _row_ids must form contiguous 0..{expected_total_rows} with zero gaps or duplicates"
    );

    let on_disk_parquet = collect_parquet_rel_paths(dir.path())?;
    assert_eq!(
        on_disk_parquet, referenced_parquet,
        "concurrent V3 OCC retries must clean up stale staged Parquet files without leaving orphans"
    );

    let meta = TableMetadata::load_latest(shared_store.as_ref()).await?;
    assert_eq!(meta.format_version, 3);
    assert_eq!(meta.next_row_id, Some(expected_total_rows as i64));

    Ok(())
}

// ============================================================================
// Pillar 8: Version-Hint Non-Regression & TableMetadata CAS Stress
// ============================================================================

#[tokio::test]
async fn pillar8_version_hint_non_regression_and_metadata_cas_stress() -> Result<()> {
    let _guard = shared_no_fault_guard().await;
    let dir = tempdir()?;
    let uri = canon_dir_uri(dir.path());
    drop(Table::create_async(uri.clone(), scalar_schema()).await?);

    let local = Arc::new(LocalFileSystem::new_with_prefix(dir.path())?);
    let store = ScriptedStore::new(local, true);

    // Part A: Lagging writer paused right before `put_opts("2", Update(e_tag))` must not regress "10".
    let barrier = store.add_pause_barrier("metadata/version-hint.text", Some("2"));
    let slow_store = store.clone();
    let slow_handle =
        tokio::spawn(
            async move { TableMetadata::advance_version_hint(slow_store.as_ref(), 2).await },
        );

    barrier.wait_entered().await;
    TableMetadata::advance_version_hint(store.as_ref(), 10).await?;
    barrier.release();
    slow_handle.await??;

    let hint_after = std::fs::read_to_string(dir.path().join("metadata/version-hint.text"))?;
    assert_eq!(
        hint_after.trim(),
        "10",
        "lagging version-hint update must not overwrite newer version 10"
    );

    // Part B: 16 concurrent `commit_update_atomic` metadata writers racing on CAS `PutMode::Create`.
    let tasks = 16usize;
    let mut handles = Vec::new();
    for i in 0..tasks {
        let s = store.clone();
        handles.push(tokio::spawn(async move {
            let snap_id = 100 + i as i64;
            TableMetadata::commit_update_atomic(s.as_ref(), 2, move |existing| {
                let mut meta = existing.context("expected existing metadata")?;
                if meta.snapshots.iter().any(|sn| sn.snapshot_id == snap_id) {
                    return Ok(
                        benostreamdb::core::metadata::MetadataUpdateOutcome::AlreadyCurrent(meta),
                    );
                }
                let seq = meta.last_sequence_number + 1;
                meta.add_snapshot(Snapshot {
                    snapshot_id: snap_id,
                    parent_snapshot_id: meta.current_snapshot_id,
                    timestamp_ms: chrono::Utc::now().timestamp_millis(),
                    sequence_number: Some(seq),
                    summary: HashMap::from([("operation".to_string(), "append".to_string())]),
                    manifest_list: format!("_manifest/snap-{snap_id}.avro"),
                    schema_id: Some(meta.current_schema_id),
                    first_row_id: None,
                    added_rows: None,
                });
                Ok(benostreamdb::core::metadata::MetadataUpdateOutcome::Updated(meta))
            })
            .await
        }));
    }

    for h in handles {
        h.await??;
    }

    let max_ver = TableMetadata::latest_version_in_store(store.as_ref())
        .await?
        .context("missing latest version")?;
    assert_eq!(max_ver, 17);
    for v in 1..=max_ver {
        let path = ObjPath::from(format!("metadata/v{v}.metadata.json"));
        let _ = TableMetadata::load_from_store(store.as_ref(), &path).await?;
    }

    let final_meta = TableMetadata::load_latest(store.as_ref()).await?;
    let snap_ids: HashSet<i64> = final_meta.snapshots.iter().map(|s| s.snapshot_id).collect();
    for i in 0..tasks {
        assert!(
            snap_ids.contains(&(100 + i as i64)),
            "concurrent metadata CAS lost snapshot {}",
            100 + i
        );
    }

    let final_hint: i32 = std::fs::read_to_string(dir.path().join("metadata/version-hint.text"))?
        .trim()
        .parse()?;
    assert_eq!(final_hint, max_ver);

    Ok(())
}

// ============================================================================
// Pillar 9: Reader vs Staging / ManifestCommitted / Index Linearizability
// ============================================================================

#[tokio::test]
async fn pillar9_concurrent_reader_vs_flush_linearizability_stress() -> Result<()> {
    let _guard = shared_no_fault_guard().await;
    let dir = tempdir()?;
    let uri = canon_dir_uri(dir.path());
    let table = Arc::new(Table::create_async(uri.clone(), vector_schema(4)).await?);
    table
        .add_index("embedding".to_string(), IndexAlgorithm::hnsw())
        .await?;

    // Seed stable anchor vectors so vector searches always have committed & streaming hits.
    let anchor_rows: Vec<(i32, Vec<f32>)> =
        vec![(1, vec![1.0, 0.0, 0.0, 0.0]), (2, vec![0.0, 1.0, 0.0, 0.0])];
    table
        .write_async(vec![make_vector_batch(4, &anchor_rows)?])
        .await?;
    table.commit_async().await?;

    let committed_ids = Arc::new(parking_lot::Mutex::new(HashSet::from([1, 2])));
    let stop = Arc::new(AtomicBool::new(false));

    // Writer + committer task executing rapid write -> read-own -> commit transitions.
    let t_writer = table.clone();
    let committed_w = committed_ids.clone();
    let writer_task = tokio::spawn(async move {
        for r in 0..24i32 {
            let base = 100 + r * 4;
            let batch_rows = vec![
                (base, vec![0.9, 0.1, (r as f32) * 0.01, 0.0]),
                (base + 1, vec![0.1, 0.9, (r as f32) * 0.01, 0.0]),
                (base + 2, vec![0.0, 0.1, 0.9, (r as f32) * 0.01]),
                (base + 3, vec![0.0, 0.0, 0.1, 0.9]),
            ];
            t_writer
                .write_async(vec![make_vector_batch(4, &batch_rows)?])
                .await?;
            // Hold a brief un-flushed window half the time before committing.
            if r % 2 == 0 {
                t_writer.commit_async().await?;
                let mut g = committed_w.lock();
                g.extend([base, base + 1, base + 2, base + 3]);
            } else {
                tokio::time::sleep(Duration::from_millis(2)).await;
                t_writer.commit_async().await?;
                let mut g = committed_w.lock();
                g.extend([base, base + 1, base + 2, base + 3]);
            }
        }
        Ok::<_, anyhow::Error>(())
    });

    // Reader tasks continuously assert no-duplicates and committed-before-start visibility
    // across both full scans and vector searches while commits flip buffer/manifest epochs.
    let mut reader_handles = Vec::new();
    for reader_idx in 0..2usize {
        let t_r = table.clone();
        let committed_r = committed_ids.clone();
        let stop_r = stop.clone();
        reader_handles.push(tokio::spawn(async move {
            let mut reads = 0usize;
            while !stop_r.load(Ordering::Acquire) {
                let required: HashSet<i32> = committed_r.lock().clone();
                if reader_idx == 0 {
                    let observed_vec = read_ids_raw(&t_r).await?;
                    let observed_set: HashSet<i32> = observed_vec.iter().copied().collect();
                    anyhow::ensure!(
                        observed_set.len() == observed_vec.len(),
                        "duplicate id observed during concurrent flush scan"
                    );
                    for id in &required {
                        anyhow::ensure!(
                            observed_set.contains(id),
                            "linearizability violation: committed id={id} vanished mid-flush"
                        );
                    }
                } else {
                    let hits =
                        vector_search_ids(&t_r, "embedding", vec![1.0, 0.0, 0.0, 0.0], 4).await?;
                    anyhow::ensure!(
                        !hits.is_empty(),
                        "vector search must never return empty across flush/index transitions"
                    );
                }
                reads += 1;
            }
            anyhow::ensure!(reads > 0, "reader made zero progress");
            Ok::<_, anyhow::Error>(reads)
        }));
    }

    writer_task.await??;
    stop.store(true, Ordering::Release);
    for h in reader_handles {
        let _ = h.await??;
    }

    table.wait_for_background_tasks_async().await?;

    let final_expected: Vec<i32> = [1, 2].into_iter().chain(100..100 + 24 * 4).collect();
    assert_eq!(read_ids_sorted_unique(&table).await?, final_expected);

    Ok(())
}
