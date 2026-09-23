// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Native ingest orchestrator (A4): plan → bounded parallel execute → OCC commit.
//!
//! Cluster-free bulk ingest that runs entirely inside the engine. A parquet
//! input set is planned into row-range work units; a bounded pool of workers
//! streams each unit into a *private* segment builder (segments are already
//! independent), builds its indexes, and the coordinator commits completed
//! segments through the existing OCC manifest CAS. Completed units are recorded
//! in a sidecar so an interrupted multi-TB load resumes at the unit boundary.

use anyhow::{Context, Result};
use arrow::record_batch::RecordBatch;
use futures::stream::{self, StreamExt};
use std::collections::{HashMap, HashSet};

use crate::core::manifest::{CommitMetadata, ManifestEntry, ManifestManager};
use crate::core::segment::HybridSegmentWriter;
use crate::core::table::state::ColumnIndexConfig;
use crate::core::table::Table;
use crate::SegmentConfig;

/// Sidecar (relative to the table root) recording completed work units.
const INGEST_STATE_PATH: &str = "_ingest_state.json";

/// Options for [`Table::ingest_async`].
#[derive(Debug, Clone)]
pub struct IngestOptions {
    /// Rows per work unit (one segment). Size from a memory budget: the demo
    /// measured ~4.5 GB per million 384-d vectors including allocator churn.
    pub chunk_rows: usize,
    /// Maximum work units in flight (bounded worker pool).
    pub parallelism: usize,
    /// Build indexes for every column (otherwise only configured columns).
    pub index_all: bool,
    /// Skip work units already recorded as complete in the resume sidecar.
    pub resume: bool,
    /// Run `rewrite_data_files` after the ingest so segment/manifest counts stay
    /// bounded at TB scale (chunked loads create many small segments by design).
    pub compact_after: bool,
}

impl Default for IngestOptions {
    fn default() -> Self {
        Self {
            chunk_rows: 1_000_000,
            parallelism: 4,
            index_all: false,
            resume: true,
            compact_after: false,
        }
    }
}

/// Result of an ingest run.
#[derive(Debug, Clone, Default)]
pub struct IngestReport {
    /// Total work units planned.
    pub units_total: usize,
    /// Units skipped because the resume sidecar already recorded them.
    pub units_skipped: usize,
    /// Units committed in this run.
    pub units_committed: usize,
    /// Rows committed in this run.
    pub rows_ingested: usize,
    /// Committed segment file paths.
    pub segments: Vec<String>,
}

/// A single unit of work: a row range within one parquet file.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct WorkUnit {
    path: String,
    row_start: usize,
    row_end: usize,
}

impl WorkUnit {
    fn key(&self) -> String {
        format!("{}:{}:{}", self.path, self.row_start, self.row_end)
    }
}

/// Row count of a parquet file (from its footer metadata).
fn parquet_row_count(path: &str) -> Result<usize> {
    let file = std::fs::File::open(path).with_context(|| format!("open parquet {}", path))?;
    let builder = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)?;
    Ok(builder.metadata().file_metadata().num_rows() as usize)
}

/// Read rows `[start, end)` from a parquet file into a single `RecordBatch`.
fn read_parquet_range(path: &str, start: usize, end: usize) -> Result<RecordBatch> {
    let file = std::fs::File::open(path).with_context(|| format!("open parquet {}", path))?;
    let builder = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)?
        .with_offset(start)
        .with_limit(end.saturating_sub(start));
    let schema = builder.schema().clone();
    let reader = builder.build()?;
    let mut batches = Vec::new();
    for b in reader {
        batches.push(b?);
    }
    if batches.is_empty() {
        return Ok(RecordBatch::new_empty(schema));
    }
    Ok(arrow::compute::concat_batches(&schema, &batches)?)
}

impl Table {
    /// Plan a bulk ingest of parquet `paths` into `(path, row_start, row_end)`
    /// work units of at most `chunk_rows` rows each.
    pub fn plan_ingest(
        &self,
        paths: &[String],
        chunk_rows: usize,
    ) -> Result<Vec<(String, usize, usize)>> {
        let chunk = chunk_rows.max(1);
        let mut units = Vec::new();
        for p in paths {
            let n = parquet_row_count(p)?;
            let mut s = 0usize;
            while s < n {
                let e = (s + chunk).min(n);
                units.push((p.clone(), s, e));
                s = e;
            }
        }
        Ok(units)
    }

    /// Native bulk ingest: plan → bounded parallel execute → OCC commit.
    ///
    /// Each work unit is read, written to a private segment (with its indexes
    /// built), and committed independently via the manifest CAS. Completed units
    /// are recorded in a sidecar so a re-run resumes at the unit boundary.
    pub async fn ingest_async(
        &self,
        paths: &[String],
        options: IngestOptions,
    ) -> Result<IngestReport> {
        let planned = self.plan_ingest(paths, options.chunk_rows)?;
        let report = self.ingest_units_async(planned, &options).await?;
        if options.compact_after {
            self.rewrite_data_files_async(None).await?;
        }
        Ok(report)
    }

    /// Ingest a single explicit row range — the serverless thin-runner entry
    /// point (`hdb ingest --range …`). Each runner commits independently via CAS.
    pub async fn ingest_range_async(
        &self,
        path: &str,
        row_start: usize,
        row_end: usize,
        options: IngestOptions,
    ) -> Result<IngestReport> {
        let report = self
            .ingest_units_async(vec![(path.to_string(), row_start, row_end)], &options)
            .await?;
        if options.compact_after {
            self.rewrite_data_files_async(None).await?;
        }
        Ok(report)
    }

    /// Core: execute a pre-planned list of work units.
    async fn ingest_units_async(
        &self,
        planned: Vec<(String, usize, usize)>,
        options: &IngestOptions,
    ) -> Result<IngestReport> {
        let mut report = IngestReport {
            units_total: planned.len(),
            ..Default::default()
        };

        // Resume: skip units already recorded as complete.
        let mut completed = if options.resume {
            self.load_ingest_state().await?
        } else {
            HashSet::new()
        };
        let pending: Vec<WorkUnit> = planned
            .into_iter()
            .map(|(path, row_start, row_end)| WorkUnit {
                path,
                row_start,
                row_end,
            })
            .filter(|u| !completed.contains(&u.key()))
            .collect();
        report.units_skipped = report.units_total - pending.len();

        if pending.is_empty() {
            return Ok(report);
        }

        let parallelism = options.parallelism.max(1);
        let index_all = options.index_all || self.indexing.index_all;
        let index_cols = self.indexing.index_columns.read().clone();
        let index_configs: HashMap<String, ColumnIndexConfig> =
            self.indexing.index_configs.read().clone();
        let default_device = self.indexing.default_device.read().clone();
        let pk = self.primary_key.read().clone();
        let base_path = self
            .uri
            .strip_prefix("file://")
            .unwrap_or(&self.uri)
            .to_string();
        std::fs::create_dir_all(&base_path)?;

        // Bounded worker pool: read + build a private segment per unit.
        let results: Vec<Result<(WorkUnit, ManifestEntry, usize)>> = stream::iter(pending)
            .map(|unit| {
                let table = self.clone();
                let base_path = base_path.clone();
                let index_cols = index_cols.clone();
                let index_configs = index_configs.clone();
                let default_device = default_device.clone();
                let pk = pk.clone();
                async move {
                    let batch = read_parquet_range(&unit.path, unit.row_start, unit.row_end)?;
                    let rows = batch.num_rows();
                    let entry = table
                        .build_ingest_segment(
                            &base_path,
                            &batch,
                            index_all,
                            &index_cols,
                            &index_configs,
                            default_device.as_deref(),
                            &pk,
                        )
                        .await?;
                    Ok((unit, entry, rows))
                }
            })
            .buffer_unordered(parallelism)
            .collect()
            .await;

        // Coordinator: commit each completed segment via the OCC CAS and record
        // resume state. Commits are serialized here so manifest versions stay
        // ordered and the sidecar is written consistently.
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        for res in results {
            let (unit, entry, rows) = res?;
            let path = entry.file_path.clone();
            manifest_manager
                .commit(
                    &[entry],
                    &[],
                    CommitMetadata {
                        skip_missing_remove_paths: true,
                        ..Default::default()
                    },
                )
                .await?;
            completed.insert(unit.key());
            self.save_ingest_state(&completed).await?;
            report.units_committed += 1;
            report.rows_ingested += rows;
            report.segments.push(path);
        }

        Ok(report)
    }

    /// Build one private segment (data + indexes) from `batch` and return its
    /// manifest entry. Does not commit.
    #[allow(clippy::too_many_arguments)]
    async fn build_ingest_segment(
        &self,
        base_path: &str,
        batch: &RecordBatch,
        index_all: bool,
        index_cols: &[String],
        index_configs: &HashMap<String, ColumnIndexConfig>,
        default_device: Option<&str>,
        pk: &[String],
    ) -> Result<ManifestEntry> {
        // Align to the table's evolved schema so every segment carries every
        // column (NULLs where absent), matching the flush path.
        let table_schema = self.arrow_schema();
        let batch = if table_schema.fields().is_empty() || batch.schema() == table_schema {
            batch.clone()
        } else {
            let mut cols = Vec::with_capacity(table_schema.fields().len());
            for field in table_schema.fields() {
                let col = if let Some(c) = batch.column_by_name(field.name()) {
                    c.clone()
                } else {
                    arrow::array::new_null_array(field.data_type(), batch.num_rows())
                };
                cols.push(col);
            }
            RecordBatch::try_new(table_schema.clone(), cols).unwrap_or_else(|_| batch.clone())
        };

        let segment_id = format!("seg_{}", uuid::Uuid::new_v4());
        let config = SegmentConfig::new(base_path, &segment_id)
            .with_index_all(index_all)
            .with_columns_to_index(index_cols.to_vec())
            .with_default_device(default_device.map(|s| s.to_string()));
        let mut writer = HybridSegmentWriter::new(config).with_index_configs(index_configs.clone());
        writer.primary_key = pk.to_vec();
        writer.set_store(self.store.clone());

        writer.write_batch(&batch)?;
        writer.build_indexes(&batch, 0)?;
        writer.finish_indexing().await?;
        writer.upload_to_store().await?;
        Ok(writer.to_manifest_entry())
    }

    /// Load the set of completed work-unit keys from the resume sidecar.
    async fn load_ingest_state(&self) -> Result<HashSet<String>> {
        let p = object_store::path::Path::from(INGEST_STATE_PATH);
        match self.store.get(&p).await {
            Ok(r) => {
                let bytes = r.bytes().await?;
                let keys: Vec<String> = serde_json::from_slice(&bytes).unwrap_or_default();
                Ok(keys.into_iter().collect())
            }
            Err(_) => Ok(HashSet::new()),
        }
    }

    /// Persist the completed work-unit keys to the resume sidecar.
    async fn save_ingest_state(&self, completed: &HashSet<String>) -> Result<()> {
        let mut keys: Vec<&String> = completed.iter().collect();
        keys.sort();
        let bytes = serde_json::to_vec(&keys)?;
        self.store
            .put(
                &object_store::path::Path::from(INGEST_STATE_PATH),
                bytes.into(),
            )
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int32Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn write_parquet(path: &std::path::Path, n: i32) {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from((0..n).collect::<Vec<_>>()))],
        )
        .unwrap();
        let file = std::fs::File::create(path).unwrap();
        let mut w = parquet::arrow::ArrowWriter::try_new(file, schema, None).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
    }

    #[tokio::test]
    async fn ingest_plans_and_commits_all_units() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let src = dir.path().join("src.parquet");
        write_parquet(&src, 2500);

        let table_uri = format!("file://{}", dir.path().join("tbl").to_string_lossy());
        let table = Table::new_async(table_uri).await?;

        let paths = vec![src.to_string_lossy().to_string()];
        let report = table
            .ingest_async(
                &paths,
                IngestOptions {
                    chunk_rows: 1000,
                    parallelism: 2,
                    ..Default::default()
                },
            )
            .await?;

        assert_eq!(report.units_total, 3, "2500 rows / 1000 = 3 units");
        assert_eq!(report.units_committed, 3);
        assert_eq!(report.rows_ingested, 2500);

        // Re-running with resume must skip everything.
        let report2 = table
            .ingest_async(
                &paths,
                IngestOptions {
                    chunk_rows: 1000,
                    parallelism: 2,
                    ..Default::default()
                },
            )
            .await?;
        assert_eq!(report2.units_committed, 0);
        assert_eq!(report2.units_skipped, 3);

        // And the rows are actually readable.
        let batches = table.read_async(None, None, None).await?;
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 2500);
        Ok(())
    }
}
