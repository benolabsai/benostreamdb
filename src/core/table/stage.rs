// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Staged data-file writes for distributed connectors.
//!
//! Spark's `BatchWrite` contract splits a write in two: the **task** writes data
//! files, the **driver** commits their metadata (`SparkWrite$TaskCommit` carries
//! `DataFile[]`, not rows). That is the only shape that keeps the driver out of
//! the data path and makes the commit metadata-sized regardless of write volume.
//!
//! So this module exposes those two halves:
//!
//!  * [`Table::stage_data_files_async`] — write batches to data files (and their
//!    indexes) and return the manifest entries. Nothing is published: a reader,
//!    engine or external, cannot see the files yet.
//!  * [`Table::commit_staged_entries_async`] — commit entries produced anywhere
//!    in one snapshot. This is a metadata-only operation, so it is safe to run on
//!    the driver.
//!
//! A retried task attempt writes new files at new paths; only the entries the
//! driver accepted are ever committed, and the uncommitted files become orphans
//! for `remove_orphan_files`.
//!
//! `Table::write_mode` says whether a table supports this: `format_version = 3`
//! tables do not, because `_row_id` is assigned from the manifest head and two
//! tasks staging concurrently would hand out overlapping ids.

use anyhow::{Context, Result};
use arrow::array::Array;
use arrow::record_batch::RecordBatch;
use std::collections::HashMap;

use super::Table;
use crate::core::manifest::{IndexFile, ManifestEntry, ManifestManager, ManifestValue};
use crate::core::segment::HybridSegmentWriter;
use crate::SegmentConfig;

impl Table {
    /// How a connector should perform writes against this table.
    ///
    /// * `"files"` — tasks may write data files, the driver commits their
    ///   entries ([`Table::stage_data_files_async`] /
    ///   [`Table::commit_staged_entries_async`]).
    /// * `"buffered"` — the table needs commit-time coordination a task cannot
    ///   provide (`format_version = 3` row-id assignment), so writes must go
    ///   through `write_async` + `commit_async`.
    pub fn write_mode(&self) -> &'static str {
        if self.get_format_version() >= 3 {
            "buffered"
        } else {
            "files"
        }
    }

    /// Write `batches` as data files with their indexes and return the manifest
    /// entries. Does **not** commit anything.
    ///
    /// The returned entries are byte-for-byte what `flush` would commit (paths
    /// relative to the table root, hive-partition prefixed, stats populated,
    /// index files attached), so committing them publishes exactly the write the
    /// engine would have made itself.
    pub async fn stage_data_files_async(
        &self,
        batches: Vec<RecordBatch>,
    ) -> Result<Vec<ManifestEntry>> {
        if batches.is_empty() {
            return Ok(Vec::new());
        }
        if self.write_mode() != "files" {
            anyhow::bail!(
                "staged writes are not supported on format_version {} tables (use write_mode() to choose)",
                self.get_format_version()
            );
        }
        let _maintenance_guard = self.maintenance_lock.read().await;

        let table_schema = self.arrow_schema();
        let aligned = self.align_staged_batches(batches, &table_schema)?;
        self.validate_staged_primary_keys(&aligned).await?;

        let spec = self.partition_spec.clone();
        let base_path = self
            .uri
            .strip_prefix("file://")
            .unwrap_or(&self.uri)
            .to_string();

        // Sort + partition, coalescing batches that land in the same partition so
        // a partition becomes one file rather than one per input batch.
        let mut grouped: Vec<(HashMap<String, serde_json::Value>, Vec<RecordBatch>)> = Vec::new();
        for batch in &aligned {
            let sorted = self.apply_sort_order(batch)?;
            for (partition_values, split) in spec.partition_batch(&sorted)? {
                if split.num_rows() == 0 {
                    continue;
                }
                match grouped.iter_mut().find(|(key, _)| key == &partition_values) {
                    Some((_, list)) => list.push(split),
                    None => grouped.push((partition_values, vec![split])),
                }
            }
        }

        let index_columns = self.indexing.index_columns.read().clone();
        let index_all = self.indexing.index_all;
        let index_configs: HashMap<String, crate::core::table::ColumnIndexConfig> =
            self.indexing.index_configs.read().clone();
        let default_device = self.indexing.default_device.read().clone();
        let primary_key = self.primary_key.read().clone();
        let build_indexes = index_all || !index_columns.is_empty() || !primary_key.is_empty();

        let mut entries = Vec::with_capacity(grouped.len());
        for (partition_values, list) in grouped {
            let batch = match list.len() {
                0 => continue,
                1 => match list.into_iter().next() {
                    Some(batch) => batch,
                    None => continue,
                },
                _ => arrow::compute::concat_batches(&list[0].schema(), &list)?,
            };
            let hive_path = spec.partition_to_path(&partition_values);
            let full_base_path = if hive_path.is_empty() {
                base_path.clone()
            } else {
                format!("{}/{}", base_path, hive_path)
            };
            let segment_id = format!("seg_{}", uuid::Uuid::new_v4());

            let config = SegmentConfig::new(&full_base_path, &segment_id)
                .with_index_all(false)
                .with_columns_to_index(Vec::new())
                .with_partition_values(partition_values.clone())
                .with_default_device(default_device.clone());
            let mut writer = HybridSegmentWriter::new(config);
            writer.primary_key = primary_key.clone();
            writer.set_store(self.store.clone());

            let batch_for_write = batch.clone();
            let (mut entry, generated) = tokio::task::spawn_blocking(move || {
                writer.write_batch(&batch_for_write)?;
                Ok::<(ManifestEntry, Vec<String>), anyhow::Error>((
                    writer.to_manifest_entry(),
                    writer.get_generated_files(),
                ))
            })
            .await
            .context("staged write task panicked")??;

            if !hive_path.is_empty() {
                entry.file_path = format!("{}/{}", hive_path, entry.file_path);
                for index_file in &mut entry.index_files {
                    index_file.file_path = format!("{}/{}", hive_path, index_file.file_path);
                }
            }

            self.upload_staged_files(&generated, &hive_path).await?;

            // Build the indexes here, not later: the driver only commits
            // metadata, so there is no post-commit hook on the table that could
            // attach them afterwards.
            if build_indexes {
                let index_files = self
                    .build_staged_indexes(
                        &batch,
                        &partition_values,
                        &segment_id,
                        &hive_path,
                        &index_columns,
                        index_all,
                        &index_configs,
                        default_device.clone(),
                        &primary_key,
                    )
                    .await?;
                if !index_files.is_empty() {
                    entry.index_files = index_files;
                }
            }

            entries.push(entry);
        }

        tracing::info!(
            "Staged {} data file(s) ({} rows) for {}",
            entries.len(),
            entries.iter().map(|e| e.record_count).sum::<i64>(),
            self.uri
        );
        Ok(entries)
    }

    /// Commit entries produced by [`Table::stage_data_files_async`] (possibly in
    /// another process) as a single snapshot.
    ///
    /// Metadata-only: the data files are already durable, so this is cheap,
    /// independent of the write's size, and safe to run on a connector driver.
    pub async fn commit_staged_entries_async(&self, entries: Vec<ManifestEntry>) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        let _maintenance_guard = self.maintenance_lock.read().await;

        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let new_manifest = manifest_manager
            .commit(
                &entries,
                &[],
                crate::core::manifest::CommitMetadata::default(),
            )
            .await?;

        self.ensure_table_metadata_async(&new_manifest).await?;
        self.publish_snapshot_async(&new_manifest, "append").await?;

        tracing::info!(
            "Committed {} staged data file(s) ({} rows) to {} as snapshot v{}",
            entries.len(),
            entries.iter().map(|e| e.record_count).sum::<i64>(),
            self.uri,
            new_manifest.version
        );
        Ok(())
    }

    /// Coerce `batches` to the table schema: cast mismatched types, NULL-fill
    /// columns the batch does not carry, and reject columns the table does not
    /// declare (a staged commit cannot carry an evolved schema).
    fn align_staged_batches(
        &self,
        batches: Vec<RecordBatch>,
        table_schema: &arrow::datatypes::SchemaRef,
    ) -> Result<Vec<RecordBatch>> {
        let mut aligned = Vec::with_capacity(batches.len());
        for batch in batches {
            for field in batch.schema().fields() {
                if table_schema.field_with_name(field.name()).is_err() {
                    anyhow::bail!(
                        "staged write carries column '{}' which the table does not declare; \
                         staged writes cannot evolve the schema",
                        field.name()
                    );
                }
            }
            let mut columns = Vec::with_capacity(table_schema.fields().len());
            for field in table_schema.fields() {
                match batch.column_by_name(field.name()) {
                    Some(column) if column.data_type() == field.data_type() => {
                        columns.push(column.clone())
                    }
                    Some(column) => columns.push(arrow::compute::cast(column, field.data_type())?),
                    None => columns.push(arrow::array::new_null_array(
                        field.data_type(),
                        batch.num_rows(),
                    )),
                }
            }
            aligned.push(RecordBatch::try_new(table_schema.clone(), columns)?);
        }
        Ok(aligned)
    }

    /// Enforce the primary-key invariant the buffered write path enforces:
    /// no NULL keys, no duplicates inside the write, none already in storage.
    async fn validate_staged_primary_keys(&self, batches: &[RecordBatch]) -> Result<()> {
        let pk_columns = self.primary_key.read().clone();
        if pk_columns.is_empty() {
            return Ok(());
        }

        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        for batch in batches {
            let key_columns: Vec<_> = pk_columns
                .iter()
                .filter_map(|name| batch.column_by_name(name).cloned())
                .collect();
            if key_columns.len() != pk_columns.len() {
                continue;
            }
            for row in 0..batch.num_rows() {
                let mut key = String::new();
                for column in &key_columns {
                    let value = ManifestValue::from_array(column, row);
                    if matches!(value, ManifestValue::Null) {
                        return Err(anyhow::Error::from(
                            crate::core::error::BenoStreamError::NullConstraintViolation {
                                column: pk_columns[0].clone(),
                            },
                        ));
                    }
                    key.push_str(&value.to_string());
                    key.push('\u{1f}');
                }
                if !seen.insert(key.clone()) {
                    return Err(anyhow::Error::from(
                        crate::core::error::BenoStreamError::PrimaryKeyViolation { key },
                    ));
                }
            }
        }

        for batch in batches {
            for name in &pk_columns {
                let Some(column) = batch.column_by_name(name) else {
                    continue;
                };
                for row in 0..batch.num_rows() {
                    let value = ManifestValue::from_array(column, row);
                    let json = serde_json::to_value(&value)?;
                    if self._check_pk_in_storage_async(name, &json).await? {
                        return Err(anyhow::Error::from(
                            crate::core::error::BenoStreamError::PrimaryKeyViolation {
                                key: value.to_string(),
                            },
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Upload the local staging files produced by a segment writer.
    async fn upload_staged_files(&self, files: &[String], hive_path: &str) -> Result<()> {
        if !(self.uri.contains("://") && !self.uri.starts_with("file://")) {
            return Ok(());
        }
        for local_path in files {
            let filename = local_path
                .split('/')
                .next_back()
                .unwrap_or(local_path)
                .to_string();
            let remote = if hive_path.is_empty() {
                filename
            } else {
                format!("{}/{}", hive_path, filename)
            };
            self.upload_staged_file(local_path, &remote).await?;
        }
        Ok(())
    }

    /// Stream one local file into the object store and remove it.
    async fn upload_staged_file(&self, local_path: &str, remote_path: &str) -> Result<()> {
        use object_store::path::Path as ObjectPath;
        use tokio::io::AsyncReadExt;

        let mut file = tokio::fs::File::open(local_path)
            .await
            .with_context(|| format!("failed to open staged file {}", local_path))?;
        let remote = ObjectPath::from(remote_path);
        let mut upload = self
            .store
            .put_multipart(&remote)
            .await
            .with_context(|| format!("failed to initiate upload to {}", remote_path))?;
        let mut buffer = vec![0u8; 8 * 1024 * 1024];
        loop {
            let read = file
                .read(&mut buffer)
                .await
                .with_context(|| format!("failed to read staged file {}", local_path))?;
            if read == 0 {
                break;
            }
            upload
                .put_part(buffer[..read].to_vec().into())
                .await
                .with_context(|| format!("failed to upload a part to {}", remote_path))?;
        }
        upload
            .complete()
            .await
            .with_context(|| format!("failed to complete upload to {}", remote_path))?;
        let _ = tokio::fs::remove_file(local_path).await;
        Ok(())
    }

    /// Build one data file's indexes and return them (paths relative to the
    /// table root, hive-partition prefixed).
    #[allow(clippy::too_many_arguments)]
    async fn build_staged_indexes(
        &self,
        batch: &RecordBatch,
        partition_values: &HashMap<String, serde_json::Value>,
        segment_id: &str,
        hive_path: &str,
        index_columns: &[String],
        index_all: bool,
        index_configs: &HashMap<String, crate::core::table::ColumnIndexConfig>,
        default_device: Option<String>,
        primary_key: &[String],
    ) -> Result<Vec<IndexFile>> {
        let base_path = self
            .uri
            .strip_prefix("file://")
            .unwrap_or(&self.uri)
            .to_string();
        let full_base_path = if hive_path.is_empty() {
            base_path
        } else {
            format!("{}/{}", base_path, hive_path)
        };
        let parquet_path = if hive_path.is_empty() {
            format!("{}.parquet", segment_id)
        } else {
            format!("{}/{}.parquet", hive_path, segment_id)
        };

        let config = SegmentConfig::new(&full_base_path, segment_id)
            .with_index_all(index_all)
            .with_columns_to_index(index_columns.to_vec())
            .with_partition_values(partition_values.clone())
            .with_column_devices(
                index_configs
                    .iter()
                    .filter_map(|(column, cfg)| {
                        cfg.device.as_ref().map(|d| (column.clone(), d.clone()))
                    })
                    .collect(),
            )
            .with_default_device(default_device)
            .with_parquet_path(parquet_path);

        let mut writer = HybridSegmentWriter::new(config).with_index_configs(index_configs.clone());
        writer.primary_key = primary_key.to_vec();
        writer.set_store(self.store.clone());

        writer
            .build_indexes(batch, 0)
            .context("failed to build staged indexes")?;
        writer.finish_indexing().await?;
        writer.upload_to_store().await?;

        let mut index_files = writer.to_manifest_entry().index_files;
        if !hive_path.is_empty() {
            for index_file in &mut index_files {
                index_file.file_path = format!("{}/{}", hive_path, index_file.file_path);
            }
        }
        Ok(index_files)
    }

    /// Create the Iceberg metadata skeleton if this table has none yet, so the
    /// snapshot the caller is about to publish has somewhere to land.
    async fn ensure_table_metadata_async(
        &self,
        new_manifest: &crate::core::manifest::Manifest,
    ) -> Result<()> {
        if self.catalog_state.catalog.is_some() {
            // A catalog owns the metadata location; `publish_snapshot_async`
            // resolves it and reports a genuinely missing table.
            return Ok(());
        }
        let store = crate::core::storage::create_object_store(&self.uri)?;
        if crate::core::metadata::TableMetadata::load_latest(store.as_ref())
            .await
            .is_ok()
        {
            return Ok(());
        }
        let schema =
            new_manifest.schemas.last().cloned().unwrap_or_else(|| {
                crate::core::manifest::Schema::from_arrow(&self.arrow_schema(), 0)
            });
        let mut metadata = crate::core::metadata::TableMetadata::new(
            self.get_format_version(),
            uuid::Uuid::new_v4().to_string(),
            self.uri.clone(),
            schema,
            new_manifest.partition_spec.clone(),
            new_manifest
                .sort_orders
                .first()
                .cloned()
                .unwrap_or_default(),
        );
        metadata
            .save_to_store(store.as_ref(), new_manifest.version as i32)
            .await?;
        Ok(())
    }
}
