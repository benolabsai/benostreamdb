// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Detached Overlay Catalog.
//!
//! A *detached overlay* accelerates an external, read-only Apache Iceberg table
//! without requiring any write permission on the upstream data lake or catalog.
//! The authoritative Parquet data stays where it is; BenoStreamDB writes only
//! its derived overlay indexes (Puffin bundles, HNSW graphs, BM25 postings) into
//! a separate, writable storage prefix.
//!
//! The [`DetachedOverlayCatalog`] is a small JSON sidecar (`detached_catalog.json`)
//! written into the overlay prefix. It pins the upstream snapshot the overlay was
//! last synchronized against, so a background worker can detect out-of-band
//! upstream mutations (appends, compaction) and re-sync incrementally.
//!
//! Correctness is preserved by the checksum lineage contract: if the upstream
//! engine rewrites a data file, the `source_data_checksum` binding fails at read
//! time and the stale index is rejected in favour of a Parquet scan. The catalog
//! only governs *when* the overlay is refreshed, never whether a stale index may
//! be trusted.

use anyhow::{Context, Result};
use chrono::Utc;
use object_store::ObjectStore;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::Table;
use crate::core::iceberg::IcebergTableMetadata;
use crate::core::manifest::ManifestManager;
use crate::core::storage::create_object_store;
use crate::core::table::builder::TableBuilder;

/// Filename of the detached overlay catalog sidecar, written at the root of the
/// overlay storage prefix.
pub const DETACHED_CATALOG_FILENAME: &str = "detached_catalog.json";

fn default_catalog_format_version() -> u32 {
    1
}

/// Persistent state of a detached overlay catalog.
///
/// Records the upstream table identity and the snapshot the overlay currently
/// reflects.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DetachedOverlayCatalog {
    /// Format version of this catalog sidecar (currently 1).
    #[serde(default = "default_catalog_format_version")]
    pub format_version: u32,

    /// Upstream Iceberg table location (e.g. `s3://enterprise-data/warehouse/`).
    pub upstream_table_uri: String,

    /// Resolved upstream metadata file URI the overlay was mounted from.
    pub upstream_metadata_uri: String,

    /// Writable storage prefix holding the overlay indexes.
    pub overlay_storage_uri: String,

    /// Upstream Iceberg snapshot ID the overlay was last synchronized against.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upstream_snapshot_id: Option<i64>,

    /// Upstream Iceberg schema ID active at the last synchronization.
    #[serde(default)]
    pub upstream_schema_id: i32,

    /// Wall-clock time (ms since epoch) of the last successful synchronization.
    #[serde(default)]
    pub last_synced_at_ms: i64,

    /// Overlay manifest version produced by the last synchronization.
    #[serde(default)]
    pub last_synced_manifest_version: u64,
}

impl DetachedOverlayCatalog {
    /// The object-store path of the catalog sidecar within the overlay prefix.
    pub fn catalog_path() -> object_store::path::Path {
        object_store::path::Path::from(DETACHED_CATALOG_FILENAME)
    }

    /// Load the catalog from the overlay store, returning `None` when absent.
    pub async fn load(store: &Arc<dyn ObjectStore>) -> Result<Option<Self>> {
        let path = Self::catalog_path();
        match store.get(&path).await {
            Ok(result) => {
                let bytes = result.bytes().await?;
                let catalog: Self = serde_json::from_slice(&bytes)
                    .context("Failed to parse detached_catalog.json")?;
                Ok(Some(catalog))
            }
            Err(object_store::Error::NotFound { .. }) => Ok(None),
            Err(e) => Err(e).context("Failed to read detached_catalog.json"),
        }
    }

    /// Persist the catalog into the overlay store.
    pub async fn save(&self, store: &Arc<dyn ObjectStore>) -> Result<()> {
        let path = Self::catalog_path();
        let bytes = serde_json::to_vec_pretty(self)?;
        store.put(&path, bytes.into()).await?;
        Ok(())
    }
}

/// Parse the numeric version out of an Iceberg metadata filename.
///
/// Handles both the BenoStreamDB `v{N}.metadata.json` convention and the
/// standard Iceberg `{N}-{uuid}.metadata.json` form. Returns `None` for names
/// that do not carry a leading version number.
fn parse_metadata_version(name: &str) -> Option<u64> {
    let stem = name.strip_suffix(".metadata.json")?;
    let stem = stem.strip_prefix('v').unwrap_or(stem);
    let digits: String = stem.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    digits.parse().ok()
}

/// Resolve an upstream Iceberg table reference to its concrete metadata file.
///
/// `table_uri` may be either a metadata file (`.../metadata/v104.metadata.json`)
/// or a table directory (`s3://enterprise-data/warehouse/`). For a directory the
/// `metadata/` prefix is listed and the highest-versioned `*.metadata.json` is
/// selected.
///
/// Returns `(metadata_file_uri, metadata_store_uri, filename)` where
/// `metadata_store_uri` is the object-store root that contains `filename`.
pub(crate) async fn resolve_iceberg_metadata_uri(
    table_uri: &str,
) -> Result<(String, String, String)> {
    let trimmed = table_uri.trim_end_matches('/');

    if trimmed.ends_with(".json") {
        let url = url::Url::parse(trimmed).context("Invalid Iceberg metadata URI")?;
        let path = std::path::Path::new(url.path());
        let parent = path
            .parent()
            .context("No parent directory for metadata file")?;
        let filename = path
            .file_name()
            .context("No filename for metadata file")?
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Non-UTF8 metadata filename"))?
            .to_string();

        let store_uri = if trimmed.starts_with("file://") {
            format!("file://{}", parent.display())
        } else {
            let mut base = url.clone();
            if let Some(parent_str) = parent.to_str() {
                base.set_path(parent_str);
            }
            base.to_string()
        };
        return Ok((trimmed.to_string(), store_uri, filename));
    }

    // Directory form: discover the latest metadata file under `metadata/`.
    let store = create_object_store(trimmed)?;
    let prefix = object_store::path::Path::from("metadata");
    use futures::StreamExt;
    let mut stream = store.list(Some(&prefix));
    let mut best: Option<(u64, String)> = None;
    while let Some(res) = stream.next().await {
        let meta = res?;
        let name = meta
            .location
            .filename()
            .map(|s| s.to_string())
            .unwrap_or_default();
        if let Some(version) = parse_metadata_version(&name) {
            if best.as_ref().map(|(v, _)| version > *v).unwrap_or(true) {
                best = Some((version, name));
            }
        }
    }

    let filename = best
        .map(|(_, name)| name)
        .unwrap_or_else(|| "v1.metadata.json".to_string());
    let metadata_file_uri = format!("{}/metadata/{}", trimmed, filename);
    let store_uri = format!("{}/metadata", trimmed);
    Ok((metadata_file_uri, store_uri, filename))
}

/// Derive the object-store root that can resolve the absolute data-file paths
/// recorded in an upstream Iceberg table's manifests.
///
/// Iceberg manifests store absolute URIs, so the data store must be rooted at
/// the bucket (or filesystem root) rather than the table directory.
pub(crate) fn upstream_data_store_uri(location: &str) -> Result<String> {
    if location.starts_with("file://") {
        Ok("file:///".to_string())
    } else if location.starts_with("s3://") {
        let url = url::Url::parse(location)?;
        Ok(format!("s3://{}/", url.host_str().unwrap_or("")))
    } else {
        Ok(location.to_string())
    }
}

impl Table {
    /// Mount an external, read-only Iceberg table with a detached overlay index
    /// catalog.
    ///
    /// * `table_uri` — upstream Iceberg table location or metadata file URI.
    /// * `overlay_storage_uri` — writable prefix where overlay indexes and the
    ///   `detached_catalog.json` sidecar are written.
    ///
    /// The upstream table is never modified. The returned [`Table`] reads data
    /// from the upstream store (`data_store`) while committing all overlay
    /// metadata into `overlay_storage_uri`.
    pub async fn mount_external_iceberg(
        table_uri: String,
        overlay_storage_uri: String,
    ) -> Result<Self> {
        let (metadata_uri, meta_store_uri, filename) =
            resolve_iceberg_metadata_uri(&table_uri).await?;
        let meta_store = create_object_store(&meta_store_uri)?;

        let bytes = meta_store
            .get(&object_store::path::Path::from(filename.as_str()))
            .await
            .context("Failed to read upstream Iceberg metadata")?
            .bytes()
            .await?;
        let iceberg_meta: IcebergTableMetadata =
            serde_json::from_slice(&bytes).context("Failed to parse upstream Iceberg metadata")?;

        // Map the upstream Iceberg schema to a BenoStreamDB schema.
        let current_schema_json = iceberg_meta
            .schemas
            .iter()
            .find(|s| {
                s.get("schema-id").and_then(|v| v.as_i64())
                    == Some(iceberg_meta.current_schema_id as i64)
            })
            .or_else(|| iceberg_meta.schemas.first())
            .context("Upstream Iceberg table has no schemas")?;
        let hdb_schema: crate::core::manifest::Schema =
            serde_json::from_value(current_schema_json.clone())
                .context("Failed to map upstream Iceberg schema")?;
        let schema_ref: arrow::datatypes::SchemaRef = Arc::new(hdb_schema.to_arrow());

        // Open an existing overlay or create a fresh one at the overlay prefix.
        let overlay_store = create_object_store(&overlay_storage_uri)?;
        let manager = ManifestManager::new(overlay_store.clone(), "", &overlay_storage_uri);
        let (_, version) = manager.load_latest().await.unwrap_or_default();

        let mut table = if version > 0 {
            TableBuilder::new(overlay_storage_uri.clone())
                .with_index_all(false)
                .build_async()
                .await?
        } else {
            Self::create_async(overlay_storage_uri.clone(), schema_ref).await?
        };

        // Read data from the upstream lake; write overlays to the overlay prefix.
        table.data_store = Some(create_object_store(&upstream_data_store_uri(
            &iceberg_meta.location,
        )?)?);

        // Import the current upstream snapshot (incremental indexing trigger).
        if let Some(snapshot_id) = iceberg_meta.current_snapshot_id {
            table
                .import_iceberg_snapshot(snapshot_id, &iceberg_meta, meta_store)
                .await?;
        }

        let catalog = DetachedOverlayCatalog {
            format_version: default_catalog_format_version(),
            upstream_table_uri: table_uri,
            upstream_metadata_uri: metadata_uri,
            overlay_storage_uri: overlay_storage_uri.clone(),
            upstream_snapshot_id: iceberg_meta.current_snapshot_id,
            upstream_schema_id: iceberg_meta.current_schema_id,
            last_synced_at_ms: Utc::now().timestamp_millis(),
            last_synced_manifest_version: table.snapshot_version().await.unwrap_or(0),
        };
        catalog.save(&overlay_store).await?;

        tracing::info!(
            upstream = %catalog.upstream_table_uri,
            overlay = %overlay_storage_uri,
            snapshot = ?catalog.upstream_snapshot_id,
            "Mounted detached overlay catalog"
        );

        Ok(table)
    }

    /// Load this table's detached overlay catalog, if it is one.
    pub async fn detached_catalog(&self) -> Result<Option<DetachedOverlayCatalog>> {
        DetachedOverlayCatalog::load(&self.store).await
    }

    /// Synchronize the detached overlay with the upstream Iceberg table.
    ///
    /// Re-reads the upstream metadata, and when the current snapshot has advanced
    /// imports the new snapshot (reconciling added/removed data files while
    /// preserving local indexes on surviving files) and updates the catalog.
    ///
    /// Returns `true` when a new snapshot was imported, `false` when the overlay
    /// was already current.
    pub async fn sync_detached_overlay_async(&self) -> Result<bool> {
        let Some(mut catalog) = DetachedOverlayCatalog::load(&self.store).await? else {
            anyhow::bail!(
                "Table '{}' is not a detached overlay (no {} in its storage prefix)",
                self.uri,
                DETACHED_CATALOG_FILENAME
            );
        };

        let (metadata_uri, meta_store_uri, filename) =
            resolve_iceberg_metadata_uri(&catalog.upstream_table_uri).await?;
        let meta_store = create_object_store(&meta_store_uri)?;
        let bytes = meta_store
            .get(&object_store::path::Path::from(filename.as_str()))
            .await
            .context("Failed to read upstream Iceberg metadata during sync")?
            .bytes()
            .await?;
        let iceberg_meta: IcebergTableMetadata = serde_json::from_slice(&bytes)
            .context("Failed to parse upstream Iceberg metadata during sync")?;

        let current_snapshot = iceberg_meta.current_snapshot_id;
        if current_snapshot == catalog.upstream_snapshot_id {
            return Ok(false);
        }

        if let Some(snapshot_id) = current_snapshot {
            self.import_iceberg_snapshot(snapshot_id, &iceberg_meta, meta_store)
                .await?;
        }

        catalog.upstream_metadata_uri = metadata_uri;
        catalog.upstream_snapshot_id = current_snapshot;
        catalog.upstream_schema_id = iceberg_meta.current_schema_id;
        catalog.last_synced_at_ms = Utc::now().timestamp_millis();
        catalog.last_synced_manifest_version = self.snapshot_version().await.unwrap_or(0);
        catalog.save(&self.store).await?;

        tracing::info!(
            upstream = %catalog.upstream_table_uri,
            snapshot = ?current_snapshot,
            "Synchronized detached overlay with upstream snapshot"
        );
        Ok(true)
    }

    /// Start a background worker that periodically synchronizes the detached
    /// overlay with the upstream Iceberg table.
    ///
    /// The worker stops when all external references to the table are dropped,
    /// mirroring [`Table::start_streaming_flush_task`].
    pub fn start_detached_sync_worker(&self, interval: std::time::Duration) {
        let bg_table = self.clone();
        let future = async move {
            let mut ticker = tokio::time::interval(interval);
            loop {
                ticker.tick().await;

                // If only this task holds the autocommit Arc, all external
                // references are gone and the worker can exit.
                if std::sync::Arc::strong_count(&bg_table.autocommit) == 1 {
                    tracing::debug!("All table references dropped, stopping detached sync worker");
                    break;
                }

                match bg_table.sync_detached_overlay_async().await {
                    Ok(true) => tracing::info!("Detached overlay sync imported a new snapshot"),
                    Ok(false) => {}
                    Err(e) => tracing::error!("Detached overlay sync failed: {}", e),
                }
            }
        };

        let handle = if let Some(rt) = &self.rt {
            rt.spawn(future)
        } else {
            tokio::spawn(future)
        };

        if let Ok(mut tasks) = self.background_tasks.try_lock() {
            tasks.push(handle);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_benostream_and_iceberg_metadata_versions() {
        assert_eq!(parse_metadata_version("v1.metadata.json"), Some(1));
        assert_eq!(parse_metadata_version("v104.metadata.json"), Some(104));
        assert_eq!(parse_metadata_version("00001-9f2c.metadata.json"), Some(1));
        assert_eq!(parse_metadata_version("metadata.json"), None);
        assert_eq!(parse_metadata_version("v.metadata.json"), None);
    }

    #[test]
    fn catalog_round_trips_through_json() {
        let catalog = DetachedOverlayCatalog {
            format_version: 1,
            upstream_table_uri: "s3://enterprise-data/warehouse/".to_string(),
            upstream_metadata_uri: "s3://enterprise-data/warehouse/metadata/v104.metadata.json"
                .to_string(),
            overlay_storage_uri: "s3://ai-team-overlays/indexes/".to_string(),
            upstream_snapshot_id: Some(104),
            upstream_schema_id: 0,
            last_synced_at_ms: 1_700_000_000_000,
            last_synced_manifest_version: 7,
        };
        let json = serde_json::to_string(&catalog).unwrap();
        let back: DetachedOverlayCatalog = serde_json::from_str(&json).unwrap();
        assert_eq!(catalog, back);
    }

    #[test]
    fn legacy_catalog_defaults_optional_fields() {
        // A catalog written before a field existed must still deserialize.
        let json = r#"{
            "upstream_table_uri": "s3://lake/wh/",
            "upstream_metadata_uri": "s3://lake/wh/metadata/v1.metadata.json",
            "overlay_storage_uri": "s3://overlays/"
        }"#;
        let catalog: DetachedOverlayCatalog = serde_json::from_str(json).unwrap();
        assert_eq!(catalog.format_version, 1);
        assert_eq!(catalog.upstream_snapshot_id, None);
        assert_eq!(catalog.last_synced_manifest_version, 0);
    }

    #[test]
    fn data_store_uri_roots_at_bucket_or_filesystem() {
        assert_eq!(
            upstream_data_store_uri("file:///tmp/warehouse").unwrap(),
            "file:///"
        );
        assert_eq!(
            upstream_data_store_uri("s3://enterprise-data/warehouse").unwrap(),
            "s3://enterprise-data/"
        );
    }
}
