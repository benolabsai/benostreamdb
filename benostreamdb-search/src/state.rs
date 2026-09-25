// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Shared server state: storage root, the open table cache, and the
//! Prometheus collectors behind plan-5.2.2 telemetry.

use arrow::datatypes::SchemaRef;
use benostreamdb::core::table::WalDurability;
use benostreamdb::{BenoStreamError, Table};
use futures::TryStreamExt;
use object_store::ObjectStore;
use prometheus::core::Collector;
use prometheus::{
    Histogram, HistogramOpts, HistogramVec, IntCounter, IntCounterVec, IntGauge, Opts, Registry,
    TextEncoder,
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

use crate::index_cache::{IndexFileCache, IndexFileKey};

/// Root storage URI for all search indexes.
///
/// Each index `<name>` is a BenoStreamDB table at `{storage_root}/{name}`.
#[derive(Clone)]
pub struct AppState {
    pub storage_root: String,
    /// Stable cluster identifier reported by `GET /`.
    pub cluster_uuid: String,
    /// Open tables by index name. Handlers share one `Arc<Table>` per index
    /// so writes land in a single write buffer / WAL.
    pub tables: Arc<RwLock<HashMap<String, Arc<Table>>>>,
    /// Prometheus collectors behind the `/metrics` endpoint (plan 5.2.2).
    pub metrics: Metrics,
    /// On-demand, size-capped LRU cache of index files fetched from the
    /// object store (keyed by index/segment/column/file + manifest version).
    pub index_cache: IndexFileCache,
    /// Hardware acceleration compute context (CPU, CUDA, ROCm, Intel, MPS).
    pub compute: benostreamdb::core::index::gpu::ComputeContext,
    /// Optional external Iceberg catalog (AWS Glue, Nessie, REST, Hive, Unity).
    pub catalog: Option<Arc<dyn benostreamdb::core::catalog::Catalog>>,
    /// Namespace for table operations in the external catalog (default: "default").
    pub catalog_namespace: String,
    /// Qdrant collection aliases: alias name -> target collection name.
    ///
    /// Aliases are process-local (in-memory) rather than persisted: they are a
    /// routing convenience, and the underlying collection data is durable. See
    /// `docs/QDRANT_COMPATIBILITY.md`.
    pub aliases: Arc<RwLock<HashMap<String, String>>>,
    /// Serializes open/create so concurrent first-use requests for the same
    /// index share one `Table` instance (no forked write buffers / WALs).
    open_gate: Arc<Mutex<()>>,
}

impl AppState {
    pub fn new(storage_root: String, cluster_uuid: String) -> Self {
        Self::with_compute(
            storage_root,
            cluster_uuid,
            benostreamdb::core::index::gpu::ComputeContext::default(),
        )
    }

    pub fn with_compute(
        storage_root: String,
        cluster_uuid: String,
        compute: benostreamdb::core::index::gpu::ComputeContext,
    ) -> Self {
        Self::with_catalog(
            storage_root,
            cluster_uuid,
            compute,
            None,
            "default".to_string(),
        )
    }

    pub fn with_catalog(
        storage_root: String,
        cluster_uuid: String,
        compute: benostreamdb::core::index::gpu::ComputeContext,
        catalog: Option<Arc<dyn benostreamdb::core::catalog::Catalog>>,
        catalog_namespace: String,
    ) -> Self {
        Self {
            storage_root,
            cluster_uuid,
            tables: Arc::new(RwLock::new(HashMap::new())),
            metrics: Metrics::new(),
            index_cache: IndexFileCache::from_env(),
            compute,
            catalog,
            catalog_namespace,
            aliases: Arc::new(RwLock::new(HashMap::new())),
            open_gate: Arc::new(Mutex::new(())),
        }
    }

    /// Table URI for a named search index: `{storage_root}/{index}`.
    pub fn index_uri(&self, index: &str) -> String {
        format!("{}/{}", self.storage_root.trim_end_matches('/'), index)
    }

    /// Look up an already-open table, or open/create one.
    ///
    /// `schema` is only consulted when the table does not exist yet
    /// (schema-on-write auto-creation from the first document).
    pub async fn open_or_create(
        &self,
        index: &str,
        schema: &Option<SchemaRef>,
    ) -> Result<Arc<Table>, BenoStreamError> {
        self.open_or_create_with_indexing(index, schema, true).await
    }

    /// Open/create a table for the Qdrant API.
    ///
    /// Unlike [`Self::open_or_create`], only the `vector` column is indexed
    /// (no BM25 inverted index over every payload column). Qdrant payload
    /// filtering falls back to a scan, but point writes stay cheap — indexing
    /// every inferred payload column made each upsert commit rebuild a full
    /// inverted index for columns that are never lexically searched.
    pub async fn open_or_create_qdrant(
        &self,
        index: &str,
        schema: &Option<SchemaRef>,
    ) -> Result<Arc<Table>, BenoStreamError> {
        self.open_or_create_with_indexing(index, schema, false)
            .await
    }

    async fn open_or_create_with_indexing(
        &self,
        index: &str,
        schema: &Option<SchemaRef>,
        index_all: bool,
    ) -> Result<Arc<Table>, BenoStreamError> {
        // 1. Fast path: already open in this process.
        {
            let tables = self.tables.read().await;
            if let Some(t) = tables.get(index) {
                self.metrics.index_cache_hits_total.inc();
                return Ok(Arc::clone(t));
            }
        }

        // 2. Serialize the open/create decision for this index.
        let _gate = self.open_gate.lock().await;
        self.metrics.index_cache_misses_total.inc();
        let uri = self.index_uri(index);

        // Open (or create) with indexing enabled. The default builder
        // config (`index_all = false`) would leave segments without the
        // BM25/HNSW indexes that `_search` relies on; `Table::builder`
        // works for existing tables, while new ones still need
        // `create_async` for the manifest/Iceberg init.
        let mut builder = Table::builder(uri.clone())
            .with_index_all(index_all)
            .with_durability(resolve_wal_durability());

        if let Some(catalog) = &self.catalog {
            builder = builder.with_catalog(Arc::clone(catalog), &self.catalog_namespace, index);
        }

        let mut table = if table_exists(&uri).await {
            builder.build_async().await.map_err(|e| {
                BenoStreamError::internal(format!("failed to open index '{index}': {e}"))
            })?
        } else {
            let schema = schema.clone().unwrap_or_else(empty_schema);
            match Table::create_async(uri.clone(), schema.clone()).await {
                Ok(_) => {}
                // Lost a create race with another request: re-open instead.
                Err(e) if e.to_string().contains("already exists") => {}
                Err(e) => {
                    return Err(BenoStreamError::internal(format!(
                        "failed to create index '{index}': {e}"
                    )))
                }
            }

            // Register table in the external Iceberg catalog if configured
            if let Some(catalog) = &self.catalog {
                match catalog.table_exists(&self.catalog_namespace, index).await {
                    Ok(false) => {
                        if let Err(e) = catalog
                            .create_table(&self.catalog_namespace, index, schema, Some(&uri))
                            .await
                        {
                            tracing::warn!(
                                index = %index,
                                namespace = %self.catalog_namespace,
                                error = %e,
                                "Failed to register new table in external catalog; storage table created"
                            );
                        } else {
                            tracing::info!(
                                index = %index,
                                namespace = %self.catalog_namespace,
                                "Registered new search index table in external Iceberg catalog"
                            );
                        }
                    }
                    Ok(true) => {}
                    Err(e) => {
                        tracing::warn!(
                            index = %index,
                            namespace = %self.catalog_namespace,
                            error = %e,
                            "Error checking external catalog table existence"
                        );
                    }
                }
            }

            builder.build_async().await.map_err(|e| {
                BenoStreamError::internal(format!("failed to open index '{index}': {e}"))
            })?
        };

        // Backfill indexes on segments committed before this table instance
        // was opened (a no-op for fresh tables). The call also pins the
        // indexing configuration on this instance so its commits keep
        // building the indexes in the background.
        if index_all {
            table.index_all_columns_async().await.map_err(|e| {
                BenoStreamError::internal(format!(
                    "failed to build search indexes for '{index}': {e}"
                ))
            })?;
        } else {
            table
                .add_index_columns_async(vec!["vector".to_string()], None)
                .await
                .map_err(|e| {
                    BenoStreamError::internal(format!(
                        "failed to build vector index for '{index}': {e}"
                    ))
                })?;
        }

        // 3. Publish (first instance wins) and hand back the shared handle.
        let mut tables = self.tables.write().await;
        let entry = tables
            .entry(index.to_string())
            .or_insert_with(|| Arc::new(table));
        Ok(Arc::clone(entry))
    }

    /// Enumerate all index names under the storage root: the first path
    /// component of every object that has an Iceberg `metadata/version-hint.text`.
    pub async fn list_indexes(&self) -> Result<Vec<String>, BenoStreamError> {
        let store =
            benostreamdb::core::storage::create_object_store(&self.storage_root).map_err(|e| {
                BenoStreamError::InvalidUri {
                    uri: self.storage_root.clone(),
                    reason: e.to_string(),
                }
            })?;
        let mut names: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        let mut stream = store.list(None);
        while let Some(obj) = stream
            .try_next()
            .await
            .map_err(|e| BenoStreamError::internal(format!("failed to list indexes: {e}")))?
        {
            let parts: Vec<&str> = obj.location.as_ref().split('/').collect();
            // A table marker lives at `<index>/metadata/version-hint.text`.
            if parts.len() >= 3 && parts[1] == "metadata" && parts[2] == "version-hint.text" {
                names.insert(parts[0].to_string());
            }
        }
        Ok(names.into_iter().collect())
    }

    /// Open a table for read-only stats without enabling index building.
    /// Used by `_cat/indices` / `_cluster/stats` so listing indexes does not
    /// trigger background index backfills.
    pub async fn open_light(&self, index: &str) -> Result<Table, BenoStreamError> {
        let uri = self.index_uri(index);
        Table::builder(uri)
            .build_async()
            .await
            .map_err(|e| BenoStreamError::internal(format!("failed to open index '{index}': {e}")))
    }

    /// Remove the index from the in-process table cache and delete every
    /// object under its URI (manifest, metadata, data, indexes) from the
    /// object store.
    pub async fn delete_index(&self, index: &str) -> Result<(), BenoStreamError> {
        // Drop the cached handle so a stale Table isn't reused.
        self.tables.write().await.remove(index);

        let uri = self.index_uri(index);
        let store = benostreamdb::core::storage::create_object_store(&uri).map_err(|e| {
            BenoStreamError::InvalidUri {
                uri: uri.clone(),
                reason: e.to_string(),
            }
        })?;
        let mut stream = store.list(None);
        while let Some(obj) = stream
            .try_next()
            .await
            .map_err(|e| BenoStreamError::internal(format!("failed to list '{index}': {e}")))?
        {
            store.delete(&obj.location).await.map_err(|e| {
                BenoStreamError::internal(format!(
                    "failed to delete {} from '{index}': {e}",
                    obj.location
                ))
            })?;
        }
        Ok(())
    }

    /// Fetch an index file from the object store, serving it from the
    /// [`IndexFileCache`] on a hit. On a miss the file is downloaded, counted
    /// in `bsdb_search_index_fetch_bytes_total`, and cached for future reads.
    ///
    /// `object_path` is relative to the index's table root (e.g.
    /// `indexes/<segment>/<column>.inv.parquet`).
    pub async fn fetch_index_file(
        &self,
        index: &str,
        segment_id: &str,
        column: &str,
        file: &str,
        manifest_version: u64,
        object_path: &str,
    ) -> Result<crate::index_cache::CachedIndex, BenoStreamError> {
        let key = IndexFileKey::new(index, segment_id, column, file, manifest_version);
        if let Some(cached) = self.index_cache.get(&key) {
            return Ok(cached);
        }

        let uri = self.index_uri(index);
        let store: Arc<dyn ObjectStore> = benostreamdb::core::storage::create_object_store(&uri)
            .map_err(|e| BenoStreamError::InvalidUri {
                uri: uri.clone(),
                reason: e.to_string(),
            })?;
        let location = object_store::path::Path::from(object_path);
        let get = store.get(&location).await.map_err(|e| {
            BenoStreamError::internal(format!("failed to fetch index file '{object_path}': {e}"))
        })?;
        let bytes = get.bytes().await.map_err(|e| {
            BenoStreamError::internal(format!("failed to read index file '{object_path}': {e}"))
        })?;

        self.metrics
            .index_fetch_bytes_total
            .with_label_values(&[file])
            .inc_by(bytes.len() as u64);
        let cached = crate::index_cache::CachedIndex::Bytes(bytes.to_vec());
        self.index_cache.put(key, cached.clone());
        Ok(cached)
    }
}

/// Prometheus collectors for the plan-5.2.2 operational telemetry: request
/// counters, per-route latency histograms (the `_search` route class is
/// the query-latency histogram), ingestion counters, index table-cache
/// hit/miss counters (the per-process cache in [`AppState::tables`]), and
/// an in-flight-request gauge. Cloning is cheap — every field is a handle
/// that shares its underlying collector.
#[derive(Clone)]
pub struct Metrics {
    registry: Arc<Registry>,
    /// In-flight HTTP requests (plan: "active connection gauges").
    pub active_requests: IntGauge,
    /// Total HTTP requests by method and route class.
    pub http_requests_total: IntCounterVec,
    /// HTTP request latency in seconds by method and route class (plan:
    /// "query latency histograms").
    pub http_request_duration_seconds: HistogramVec,
    /// Documents indexed by result: `created` / `updated` / `error` (plan:
    /// "ingestion throughput counters").
    pub docs_indexed_total: IntCounterVec,
    /// Index table-cache hits (`open_or_create` fast path).
    pub index_cache_hits_total: IntCounter,
    /// Index table-cache misses (open/create path).
    pub index_cache_misses_total: IntCounter,
    /// Search query latency in seconds by operation class
    /// (`match` | `knn` | `hybrid` | `filter`).
    pub query_seconds: HistogramVec,
    /// Bulk items processed by outcome status (`200` / `201` / `400` / `501`).
    pub bulk_items_total: IntCounterVec,
    /// `_refresh` duration in seconds.
    pub refresh_seconds: Histogram,
    /// Bytes fetched from the object store for index files, by file kind
    /// (`inv.parquet` | `inv.meta.json` | `hnsw`).
    pub index_fetch_bytes_total: IntCounterVec,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    // Metric definitions are built from static, compile-time-constant names and
    // help strings. The prometheus constructors are only fallible on an invalid
    // name or a duplicate registration, i.e. a programming error caught by the
    // unit tests below — never a runtime condition. There is no infallible
    // constructor in the `prometheus` API, so this is one of the documented
    // residual invariants in NO_PANIC_POLICY.md. Failure here happens once at
    // startup, before the server binds, and is not reachable from a request.
    #[allow(clippy::unwrap_used, clippy::expect_used)]
    pub fn new() -> Self {
        let registry = Registry::new();
        let active_requests =
            IntGauge::new("bsdb_search_active_requests", "In-flight HTTP requests").unwrap();
        let http_requests_total = IntCounterVec::new(
            Opts::new(
                "bsdb_search_http_requests_total",
                "Total HTTP requests by method and route class",
            ),
            &["method", "route"],
        )
        .unwrap();
        let http_request_duration_seconds = HistogramVec::new(
            HistogramOpts::new(
                "bsdb_search_http_request_duration_seconds",
                "HTTP request latency in seconds by method and route class",
            ),
            &["method", "route"],
        )
        .unwrap();
        let docs_indexed_total = IntCounterVec::new(
            Opts::new(
                "bsdb_search_docs_indexed_total",
                "Documents indexed, by result",
            ),
            &["result"],
        )
        .unwrap();
        let index_cache_hits_total = IntCounter::new(
            "bsdb_search_index_cache_hits_total",
            "Index table-cache hits (open_or_create fast path)",
        )
        .unwrap();
        let index_cache_misses_total = IntCounter::new(
            "bsdb_search_index_cache_misses_total",
            "Index table-cache misses (open/create path)",
        )
        .unwrap();
        let query_seconds = HistogramVec::new(
            HistogramOpts::new(
                "bsdb_search_query_seconds",
                "Search query latency in seconds by operation class",
            ),
            &["op"],
        )
        .unwrap();
        let bulk_items_total = IntCounterVec::new(
            Opts::new(
                "bsdb_search_bulk_items_total",
                "Bulk items processed, by outcome status",
            ),
            &["status"],
        )
        .unwrap();
        let refresh_seconds = Histogram::with_opts(HistogramOpts::new(
            "bsdb_search_refresh_seconds",
            "_refresh duration in seconds",
        ))
        .unwrap();
        let index_fetch_bytes_total = IntCounterVec::new(
            Opts::new(
                "bsdb_search_index_fetch_bytes_total",
                "Bytes fetched from the object store for index files, by file kind",
            ),
            &["kind"],
        )
        .unwrap();

        let collectors: Vec<Box<dyn Collector>> = vec![
            Box::new(active_requests.clone()),
            Box::new(http_requests_total.clone()),
            Box::new(http_request_duration_seconds.clone()),
            Box::new(docs_indexed_total.clone()),
            Box::new(index_cache_hits_total.clone()),
            Box::new(index_cache_misses_total.clone()),
            Box::new(query_seconds.clone()),
            Box::new(bulk_items_total.clone()),
            Box::new(refresh_seconds.clone()),
            Box::new(index_fetch_bytes_total.clone()),
        ];
        for collector in collectors {
            registry.register(collector).unwrap();
        }

        Self {
            registry: Arc::new(registry),
            active_requests,
            http_requests_total,
            http_request_duration_seconds,
            docs_indexed_total,
            index_cache_hits_total,
            index_cache_misses_total,
            query_seconds,
            bulk_items_total,
            refresh_seconds,
            index_fetch_bytes_total,
        }
    }

    /// Prometheus text format (version 0.0.4) snapshot of every collector.
    pub fn gather_text(&self) -> String {
        let encoder = TextEncoder::new();
        let mut out = String::new();
        encoder
            .encode_utf8(&self.registry.gather(), &mut out)
            .unwrap_or_default();
        out
    }
}

/// Default storage root: `file://~/.benostreamdb/search`
pub fn default_storage_uri() -> String {
    match std::env::var("HOME") {
        Ok(home) => format!("file://{home}/.benostreamdb/search"),
        Err(_) => "file:///tmp/.benostreamdb/search".to_string(),
    }
}

/// Resolve `BENOSEARCH_STORAGE_URI` (or fallback `BENOSTREAM_STORAGE_URI`, defaulting to `file://~/.benostreamdb/search`),
/// expanding a leading `~` into `$HOME`.
pub fn resolve_storage_uri() -> String {
    let raw = std::env::var("BENOSEARCH_STORAGE_URI")
        .or_else(|_| std::env::var("BENOSTREAM_STORAGE_URI"))
        .unwrap_or_else(|_| default_storage_uri());
    let trimmed = raw.trim_end_matches('/');
    if trimmed == "~" || trimmed.starts_with("~/") {
        let home = std::env::var("HOME").unwrap_or_default();
        format!("{home}{}", &trimmed[1..])
    } else {
        trimmed.to_string()
    }
}

/// Probe for an initialized table: the Iceberg metadata version hint under
/// the table root. The object store is prefix-scoped to the table directory,
/// so the path is relative.
pub(crate) async fn table_exists(uri: &str) -> bool {
    match benostreamdb::core::storage::create_object_store(uri) {
        Ok(store) => store
            .head(&object_store::path::Path::from(
                "metadata/version-hint.text",
            ))
            .await
            .is_ok(),
        Err(_) => false,
    }
}

/// Resolve the WAL durability mode for search tables from
/// `BENOSEARCH_WAL_DURABILITY` or `BENOSTREAM_WAL_DURABILITY` (default `async`).
///
/// `async` hands writes to the background WAL worker (batched fsync) for
/// maximum ingest throughput; `sync` fsyncs every write for the strongest
/// crash guarantees. Unknown values fall back to `async`.
fn resolve_wal_durability() -> WalDurability {
    match std::env::var("BENOSEARCH_WAL_DURABILITY")
        .or_else(|_| std::env::var("BENOSTREAM_WAL_DURABILITY"))
        .ok()
        .as_deref()
        .map(str::to_ascii_lowercase)
    {
        Some(v) if v == "sync" => WalDurability::Sync,
        _ => WalDurability::Async,
    }
}

/// Resolve external Iceberg catalog configuration from environment variables or benostream.toml.
///
/// Precedence:
/// 1. `BENOSEARCH_CATALOG_TYPE` / `BENOSTREAM_CATALOG_TYPE` (explicit environment variables)
/// 2. `benostream.toml` / `BENOSTREAM_CONFIG` (via `CatalogConfig::load_default()`)
///
/// Returns `Some((Arc<dyn Catalog>, namespace))` if configured, or `None` for path-based Iceberg mode.
pub async fn resolve_catalog() -> Option<(Arc<dyn benostreamdb::core::catalog::Catalog>, String)> {
    use benostreamdb::core::catalog::{create_catalog_async, CatalogConfig, CatalogType};
    use std::str::FromStr;

    // 1. Check environment variables
    let type_env = std::env::var("BENOSEARCH_CATALOG_TYPE")
        .or_else(|_| std::env::var("BENOSTREAM_CATALOG_TYPE"))
        .ok();

    if let Some(type_str) = type_env {
        match CatalogType::from_str(&type_str) {
            Ok(cat_type) => {
                let mut config_map = HashMap::new();

                // URL / URI
                if let Ok(u) = std::env::var("BENOSEARCH_CATALOG_URL")
                    .or_else(|_| std::env::var("BENOSEARCH_CATALOG_URI"))
                    .or_else(|_| std::env::var("BENOSTREAM_CATALOG_URL"))
                    .or_else(|_| std::env::var("BENOSTREAM_CATALOG_URI"))
                {
                    config_map.insert("url".to_string(), u.clone());
                    config_map.insert("uri".to_string(), u);
                }

                // Token / Credential / Prefix / Catalog ID
                if let Ok(tok) = std::env::var("BENOSEARCH_CATALOG_TOKEN")
                    .or_else(|_| std::env::var("BENOSTREAM_CATALOG_TOKEN"))
                {
                    config_map.insert("token".to_string(), tok);
                }
                if let Ok(cred) = std::env::var("BENOSEARCH_CATALOG_CREDENTIAL")
                    .or_else(|_| std::env::var("BENOSTREAM_CATALOG_CREDENTIAL"))
                {
                    config_map.insert("credential".to_string(), cred);
                }
                if let Ok(pfx) = std::env::var("BENOSEARCH_CATALOG_PREFIX") {
                    config_map.insert("prefix".to_string(), pfx);
                }
                if let Ok(cid) = std::env::var("BENOSEARCH_CATALOG_ID") {
                    config_map.insert("catalog_id".to_string(), cid);
                }

                let namespace = std::env::var("BENOSEARCH_CATALOG_NAMESPACE")
                    .or_else(|_| std::env::var("BENOSTREAM_CATALOG_NAMESPACE"))
                    .unwrap_or_else(|_| "default".to_string());

                match create_catalog_async(cat_type, config_map).await {
                    Ok(boxed_catalog) => {
                        let arc_catalog: Arc<dyn benostreamdb::core::catalog::Catalog> =
                            Arc::from(boxed_catalog);
                        tracing::info!(
                            catalog_type = ?cat_type,
                            namespace = %namespace,
                            "Initialized external Iceberg catalog from environment"
                        );
                        return Some((arc_catalog, namespace));
                    }
                    Err(e) => {
                        tracing::warn!(
                            catalog_type = ?cat_type,
                            error = %e,
                            "Failed to initialize external catalog from environment; falling back to path-based Iceberg"
                        );
                        return None;
                    }
                }
            }
            Err(e) => {
                tracing::warn!(
                    catalog_type = %type_str,
                    error = %e,
                    "Invalid BENOSEARCH_CATALOG_TYPE; falling back to path-based Iceberg"
                );
                return None;
            }
        }
    }

    // 2. Check benostream.toml / default configuration file
    if let Ok(cat_config) = CatalogConfig::load_default() {
        let namespace = cat_config
            .config
            .get("namespace")
            .cloned()
            .unwrap_or_else(|| "default".to_string());

        match create_catalog_async(cat_config.catalog_type, cat_config.config).await {
            Ok(boxed_catalog) => {
                let arc_catalog: Arc<dyn benostreamdb::core::catalog::Catalog> =
                    Arc::from(boxed_catalog);
                tracing::info!(
                    catalog_type = ?cat_config.catalog_type,
                    namespace = %namespace,
                    "Initialized external Iceberg catalog from configuration file"
                );
                return Some((arc_catalog, namespace));
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "Failed to initialize external catalog from config file; falling back to path-based Iceberg"
                );
            }
        }
    }

    None
}

fn empty_schema() -> SchemaRef {
    Arc::new(arrow::datatypes::Schema::empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_uri_joins_root_and_name() {
        let state = AppState::new("/data/search".to_string(), "uuid".to_string());
        assert_eq!(state.index_uri("people"), "/data/search/people");
    }

    #[test]
    fn index_uri_trims_trailing_slash_on_root() {
        let state = AppState::new("/data/search/".to_string(), "uuid".to_string());
        assert_eq!(state.index_uri("people"), "/data/search/people");
    }

    #[test]
    fn resolve_storage_uri_expands_tilde() {
        std::env::set_var("BENOSEARCH_STORAGE_URI", "~/searches");
        let home = std::env::var("HOME").unwrap_or_default();
        assert_eq!(resolve_storage_uri(), format!("{home}/searches"));
        std::env::remove_var("BENOSEARCH_STORAGE_URI");
    }

    #[tokio::test]
    async fn open_or_create_reports_cache_hits_and_misses() {
        let tmp = tempfile::tempdir().unwrap();
        let root = format!("file://{}", tmp.path().display());
        let state = AppState::new(root, "test-cluster".into());
        std::fs::create_dir_all(tmp.path().join("mi")).unwrap();

        // First open creates the table (cache miss); the second reuses the
        // in-process table cache (cache hit).
        state.open_or_create("mi", &None).await.unwrap();
        state.open_or_create("mi", &None).await.unwrap();

        assert_eq!(state.metrics.index_cache_hits_total.get(), 1);
        assert_eq!(state.metrics.index_cache_misses_total.get(), 1);
    }

    #[tokio::test]
    async fn test_resolve_catalog_lifecycle() {
        // 1. Unset -> None
        std::env::remove_var("BENOSEARCH_CATALOG_TYPE");
        std::env::remove_var("BENOSTREAM_CATALOG_TYPE");
        assert!(resolve_catalog().await.is_none());

        // 2. Set REST catalog -> Some
        std::env::set_var("BENOSEARCH_CATALOG_TYPE", "rest");
        std::env::set_var(
            "BENOSEARCH_CATALOG_URL",
            "http://localhost:8181/api/catalog/v1",
        );
        std::env::set_var("BENOSEARCH_CATALOG_NAMESPACE", "analytics");

        let resolved = resolve_catalog().await;
        assert!(resolved.is_some());
        let (_, ns) = resolved.unwrap();
        assert_eq!(ns, "analytics");

        std::env::remove_var("BENOSEARCH_CATALOG_TYPE");
        std::env::remove_var("BENOSEARCH_CATALOG_URL");
        std::env::remove_var("BENOSEARCH_CATALOG_NAMESPACE");
    }
}
