// Copyright (c) 2026 Richard Albright. All rights reserved.

use crate::core::reader::HybridReader;
use crate::core::segment::HybridSegmentWriter;
use crate::SegmentConfig;
use crate::core::storage::create_object_store;
use crate::core::manifest::ManifestManager;
use futures::StreamExt;
use object_store::ObjectStore;
use object_store::path::Path;
use std::sync::Arc;
use std::collections::HashMap;
use serde_json::Value;
use anyhow::{Result, Context};
use tokio::fs;
use tracing;
use crate::telemetry::metrics::COMPACTION_DURATION_SECONDS;


#[derive(Clone, Debug)]
pub struct ClusteringOptions {
    pub columns: Vec<String>,
    pub strategy: String, // "zorder" | "hilbert"
}

#[derive(Clone, Debug)]
pub struct CompactionOptions {
    pub target_file_size_bytes: i64,
    pub min_file_size_bytes: i64,
    pub strategy: String,
    pub max_concurrent_bins: usize,
    pub clustering: Option<ClusteringOptions>,
}

impl Default for CompactionOptions {
    fn default() -> Self {
        Self {
            target_file_size_bytes: 512 * 1024 * 1024, // 512MB
            min_file_size_bytes: 384 * 1024 * 1024,   // ~384MB
            strategy: "binpack".to_string(),
            max_concurrent_bins: 4,
            clustering: None,
        }
    }
}

#[derive(Clone)]
pub struct Compactor {
    store: Arc<dyn ObjectStore>,
    manifest: ManifestManager,
    options: CompactionOptions,
    base_path: String,
    root_uri: String,
}

impl Compactor {
    pub fn new(uri: &str, options: CompactionOptions) -> Result<Self> {
        let store = create_object_store(uri)?;
        
        // Handle base_path differentiation
        let url = url::Url::parse(uri).context("Invalid URI")?;
        let base_path = if uri.starts_with("file://") {
            "".to_string() // LocalStore is already rooted at the path
        } else {
             // For S3/Azure/GCS, the URI is s3://bucket/prefix
             // We want 'prefix' as the base_path
             url.path().trim_start_matches('/').to_string()
        };

        let manifest = ManifestManager::new(store.clone(), &base_path, uri);
        
        Ok(Self {
            store,
            manifest,
            root_uri: uri.to_string(),
            options,
            base_path,
        })
    }

    pub async fn rewrite_data_files(&self) -> Result<()> {
        let _timer = COMPACTION_DURATION_SECONDS.start_timer();
        tracing::info!("Starting compaction with strategy: {}", self.options.strategy);
        
        // 1. Discovery: ONLY consider segments active in the latest manifest
        let (_manifest, all_entries, _) = self.manifest.load_latest_full().await?;
        if all_entries.is_empty() {
            tracing::info!("Manifest is empty. Nothing to compact.");
            return Ok(());
        }

        let mut candidates = Vec::new();
        for entry in &all_entries {
            // Only consider reasonably small files for compaction
            if entry.file_size_bytes < self.options.min_file_size_bytes {
                 tracing::debug!("Found candidate: {} ({} bytes)", entry.file_path, entry.file_size_bytes);
                 candidates.push(entry.clone());
            }
        }

        if candidates.is_empty() {
            tracing::info!("No segments require compaction.");
            return Ok(());
        }

        // 2. Group by Partition & BinPack
        let mut partition_groups: HashMap<Vec<(String, Value)>, Vec<crate::core::manifest::ManifestEntry>> = HashMap::new();
        for candidate in candidates {
            let mut key: Vec<(String, Value)> = candidate.partition_values.clone().into_iter().collect();
            key.sort_by(|a, b| a.0.cmp(&b.0));
            partition_groups.entry(key).or_insert_with(Vec::new).push(candidate);
        }

        let mut bins: Vec<Vec<crate::core::manifest::ManifestEntry>> = Vec::new();
        for (_part_key, group) in partition_groups {
            let mut current_bin = Vec::new();
            let mut current_size = 0_i64;

            for candidate in group {
                if current_size + candidate.file_size_bytes > self.options.target_file_size_bytes
                    && !current_bin.is_empty() {
                        bins.push(current_bin);
                        current_bin = Vec::new();
                        current_size = 0;
                    }
                current_size += candidate.file_size_bytes;
                current_bin.push(candidate);
            }
            if !current_bin.is_empty() {
                bins.push(current_bin);
            }
        }

        tracing::info!("Plan: Identified {} bins across partitions to compact.", bins.len());

        // 3. Parallel Execution
        // We want to process bins in parallel, but commit atomically at the end.
        let max_concurrent = self.options.max_concurrent_bins;
        tracing::info!("Executing with parallelism: {}", max_concurrent);

        // This requires cloning 'self' for the async move block.
        // Since 'self' contains Arc<Store> and ManifestManager (which owns Arcs), it's cheap to clone if we derive Clone.
        // Compactor doesn't derive Clone yet. Let's wrap meaningful parts in Arc or clone manually if needed.
        // Ideally Compactor should be cheap to clone or we pass a reference.
        // Iterate over bins and map to futures.
        
        // We'll use a stream to limit concurrency.
        let results: Vec<Result<(crate::core::manifest::ManifestEntry, Vec<String>)>> = futures::stream::iter(bins)
            .map(|bin| {
                let compactor = self.clone(); // Needs Clone derive on Compactor
                async move {
                    compactor.compact_bin(bin).await
                }
            })
            .buffer_unordered(max_concurrent)
            .collect()
            .await;

        // 4. Collect Results & Batch Commit
        let mut all_new_entries = Vec::new();
        let mut all_old_paths = Vec::new();

        for res in results {
            match res {
                Ok((new_entry, old_paths)) => {
                    all_new_entries.push(new_entry);
                    all_old_paths.extend(old_paths);
                },
                Err(e) => {
                    tracing::error!("Error during parallel compaction: {}", e);
                    // Decide strategy: Abort all? Or partial commit?
                    // For now: Abort functionality to maintain consistency.
                    return Err(e);
                }
            }
        }

        if !all_new_entries.is_empty() {
            tracing::info!("Committing Batch: +{} entries, -{} paths", all_new_entries.len(), all_old_paths.len());
            self.manifest.commit(&all_new_entries, &all_old_paths, crate::core::manifest::CommitMetadata::default()).await?;
        }

        Ok(())
    }

    /// Compacts a single bin and returns the (NewEntry, OldPaths) without committing.
    async fn compact_bin(&self, bin: Vec<crate::core::manifest::ManifestEntry>) -> Result<(crate::core::manifest::ManifestEntry, Vec<String>)> {
        if bin.is_empty() {
             return Err(anyhow::anyhow!("Empty bin"));
        }

        // A. Setup Local Temp Environment
        let temp_id = uuid::Uuid::new_v4();
        let temp_dir_path = std::env::temp_dir().join(format!("hyperstream_compact_{}", temp_id));
        fs::create_dir_all(&temp_dir_path).await?;
        let temp_dir_str = temp_dir_path.to_str().unwrap().to_string();

        let new_segment_id = format!("compacted_{}_{}", chrono::Utc::now().format("%Y%m%d%H%M%S"), temp_id);
        
        let writer_config = SegmentConfig::new(&temp_dir_str, &new_segment_id);
        let writer = HybridSegmentWriter::new(writer_config);

        // B. Stream and Accumulate All Batches
        let mut all_batches: Vec<arrow::record_batch::RecordBatch> = Vec::new();
        
        for entry in &bin {
             let path = std::path::Path::new(&entry.file_path);
             let segment_id = path.file_stem().unwrap().to_str().unwrap();
             let rel_parent = path.parent().and_then(|p| p.to_str()).unwrap_or("");
             
             let config = SegmentConfig::new(rel_parent, segment_id); 
             let reader = HybridReader::new(config, self.store.clone(), &self.root_uri);
             
             // Compaction reads all columns to preserve full data
             let mut stream = reader.stream_all(None as Option<arrow::datatypes::SchemaRef>).await?;
             
             while let Some(batch_res) = stream.next().await {
                 let batch = batch_res?;
                 all_batches.push(batch);
             }
        }
        
        // Concatenate all batches into one and write
        if all_batches.is_empty() {
            return Err(anyhow::anyhow!("No data to compact"));
        }
        
        let schema = all_batches[0].schema();
        let mut merged_batch = arrow::compute::concat_batches(&schema, &all_batches)?;
        let total_rows = merged_batch.num_rows() as i64;
        
        // APPLY CLUSTERING
        if let Some(clustering) = &self.options.clustering {
            if clustering.strategy == "zorder" {
                tracing::info!("Applying Z-Order clustering on columns: {:?}", clustering.columns);
                merged_batch = crate::core::clustering::apply_zorder(&merged_batch, &clustering.columns)?;
            } else if clustering.strategy == "hilbert" {
                tracing::info!("Applying Hilbert clustering on columns: {:?}", clustering.columns);
                merged_batch = crate::core::clustering::apply_hilbert(&merged_batch, &clustering.columns)?;
            }
        }

        writer.write_batch(&merged_batch)?;
        
        // C. Upload Artifacts to Object Store
        let generated_files = writer.get_generated_files();
        
        let mut main_parquet_path = String::new();
        let mut main_parquet_size = 0;
        let mut index_files = Vec::new();
        
        for local_path in generated_files {
             let file_name = std::path::Path::new(&local_path).file_name().unwrap().to_str().unwrap();
             let file_size = fs::metadata(&local_path).await?.len();
             
             let remote_path = if self.base_path.is_empty() {
                 Path::from(file_name)
             } else {
                 Path::from(format!("{}/{}", self.base_path, file_name))
             };
             
             let content = fs::read(&local_path).await?;
             self.store.put(&remote_path, content.into()).await?;
             
             let remote_path_str = remote_path.to_string();
             
            if file_name.ends_with(".parquet") && !file_name.contains(".inv.parquet") {
                main_parquet_path = remote_path_str;
                main_parquet_size = file_size;
            } else if file_name.ends_with(".inv.parquet") {
                // Inverted index file
                let parts: Vec<&str> = file_name.split('.').collect();
                // Format: segment_id.column_name.inv.parquet -> column is parts[1]
                let column_name = if parts.len() >= 4 {
                    Some(parts[1].to_string())
                } else {
                    None
                };
                index_files.push(crate::core::manifest::IndexFile {
                    file_path: remote_path_str,
                    index_type: "inverted".to_string(),
                    column_name,
                    blob_type: None,
                    offset: None,
                    length: None,
                });
            } else {
                 let index_type = if file_name.contains(".hnsw") {
                     "vector"
                 } else if file_name.contains(".idx") {
                     "scalar"
                 } else {
                     "unknown"
                 }.to_string();
                 
                 let parts: Vec<&str> = file_name.split('.').collect();
                 let column_name = if parts.len() >= 3 {
                     Some(parts[parts.len()-2].to_string())
                 } else {
                     None
                 };

                index_files.push(crate::core::manifest::IndexFile {
                    file_path: remote_path_str,
                    index_type,
                    column_name,
                    blob_type: None,
                    offset: None,
                    length: None,
                });
             }
        }

        // e. Cleanup Local
        fs::remove_dir_all(&temp_dir_path).await?;

        // D. Prepare Result (No Commit)
        let column_stats = writer.get_stats(); // Use the merged stats
        
        let mut min_clustering_score = None;
        let mut max_clustering_score = None;
        let mut clustering_strategy = None;
        let mut clustering_columns = None;
        let mut normalization_mins = None;
        let mut normalization_maxs = None;

        if let Some(clustering) = &self.options.clustering {
             clustering_strategy = Some(clustering.strategy.clone());
             clustering_columns = Some(clustering.columns.clone());
             
             let (scores, mins, maxs) = if clustering.strategy == "zorder" {
                 crate::core::clustering::compute_zorder_scores(&merged_batch, &clustering.columns)?
             } else {
                 crate::core::clustering::compute_hilbert_scores(&merged_batch, &clustering.columns)?
             };
             
             if !scores.is_empty() {
                 min_clustering_score = Some(scores.value(0));
                 max_clustering_score = Some(scores.value(scores.len() - 1));
                 normalization_mins = Some(mins);
                 normalization_maxs = Some(maxs);
             }
        }

        let new_entry = crate::core::manifest::ManifestEntry {
            file_path: main_parquet_path,
            file_size_bytes: main_parquet_size as i64,
            record_count: total_rows,
            index_files,
            delete_files: vec![], // Compaction garbage collects deletes, so new segment has no deletes
            column_stats,
            partition_values: bin[0].partition_values.clone(),
            clustering_strategy,
            clustering_columns,
            min_clustering_score,
            max_clustering_score,
            normalization_mins,
            normalization_maxs,
        };
        
        let mut old_paths = Vec::new();
        for entry in &bin {
             old_paths.push(entry.file_path.clone());
        }

        tracing::info!("Compacted bin {} -> {}", old_paths.len(), new_entry.file_path);

        Ok((new_entry, old_paths))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::array::Int32Array;
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_compaction_local_fs() -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let uri = format!("file://{}", temp_dir.path().to_str().unwrap());
        
        let table = crate::Table::new_async(uri.clone()).await?;
        
        // 1. Write 3 small segments
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
        ]));
        
        for i in 0..3 {
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int32Array::from(vec![i]))]
            )?;
            table.write_async(vec![batch]).await?;
            table.commit_async().await?;
        }
        
        // 2. Setup Compactor options
        let options = CompactionOptions {
            min_file_size_bytes: 1024 * 1024, // catch all
            target_file_size_bytes: 1024 * 1024,
            ..Default::default()
        };
        
        // 3. Compact
        table.rewrite_data_files_async(Some(options)).await?;
        
        // 4. Verify Manifest
        let entries = table.get_snapshot_segments().await?;
        assert_eq!(entries.len(), 1, "Should have 1 compacted segment");
        assert_eq!(entries[0].record_count, 3);
        
        // Cleanup cache
        let cache_key = format!("{}/{}", uri, "");
        crate::core::cache::LATEST_VERSION_CACHE.invalidate(&cache_key).await;
        
        Ok(())
    }

    #[tokio::test]
    async fn test_concurrent_compaction() -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let uri = format!("file://{}", temp_dir.path().to_str().unwrap());
        
        let table = crate::Table::new_async(uri.clone()).await?;
        
        // 1. Write many segments
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
        ]));
        
        let num_segments = 10;
        for i in 0..num_segments {
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int32Array::from(vec![i; 10]))]
            )?;
            table.write_async(vec![batch]).await?;
            table.commit_async().await?;
        }
        
        // 2. Configure Compactor (small target to ensure multiple bins)
        let options = CompactionOptions {
            target_file_size_bytes: 100, // VERY small target
            min_file_size_bytes: 1024 * 1024,
            max_concurrent_bins: 4,
            ..Default::default()
        };
        
        // 3. Execution
        table.rewrite_data_files_async(Some(options)).await?;
        
        // 4. Verify
        let entries = table.get_snapshot_segments().await?;
        assert!(entries.len() > 1, "Should have created multiple segments");
        let total_count: i64 = entries.iter().map(|e| e.record_count).sum();
        assert_eq!(total_count, (num_segments * 10) as i64);
        
        // Cleanup cache
        let cache_key = format!("{}/{}", uri, "");
        crate::core::cache::LATEST_VERSION_CACHE.invalidate(&cache_key).await;
        
        Ok(())
    }

    #[tokio::test]
    async fn test_vacuum_old_versions() -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let uri = format!("file://{}", temp_dir.path().to_str().unwrap());
        let table = crate::Table::new_async(uri.clone()).await?;
        
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
        ]));
        
        // 1. Write and commit 3 times (Creating v1, v2, v3)
        for i in 0..3 {
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int32Array::from(vec![i]))]
            )?;
            table.write_async(vec![batch]).await?;
            table.commit_async().await?;
        }
        
        let (_, v3) = table.get_snapshot_segments_with_version().await?;
        assert_eq!(v3, 3);
        
        // 2. Compact (Creates v4)
        // Ensure compaction happens by setting small target
        let options = CompactionOptions {
            target_file_size_bytes: 1024 * 1024,
            min_file_size_bytes: 1024 * 1024,
            ..Default::default()
        };
        table.rewrite_data_files_async(Some(options)).await?;
        
        let (_, v4) = table.get_snapshot_segments_with_version().await?;
        assert_eq!(v4, 4);
        
        // Check entries using get_snapshot_segments which loads from manifest lists
        let entries_v4 = table.get_snapshot_segments().await?;
        assert_eq!(entries_v4.len(), 1, "Should have 1 compacted segment");
        
        // 3. Vacuum keeping only the last version (v4)
        // This should delete:
        // - Manifests v1, v2, v3
        // - The 3 original data files (since they are ONLY in v1-v3)
        let deleted = table.vacuum_async(1).await?;
        tracing::info!("Vacuum deleted {} files", deleted);
        
        // We expect at least 3 data files + 3 manifest files = 6 files deleted
        assert!(deleted >= 6);
        
        // 4. Verify we can still read the table (v4)
        let entries = table.get_snapshot_segments().await?;
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].record_count, 3);
        
        // Cleanup cache
        let cache_key = format!("{}/{}", uri, "");
        crate::core::cache::LATEST_VERSION_CACHE.invalidate(&cache_key).await;
        
        Ok(())
    }

    #[tokio::test]
    async fn test_zorder_clustering() -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let uri = format!("file://{}", temp_dir.path().to_str().unwrap());
        let table = crate::Table::new_async(uri.clone()).await?;
        
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int32, false),
            Field::new("y", DataType::Int32, false),
        ]));
        
        // Write data that is NOT in Z-order
        let x = Int32Array::from(vec![1, 0, 1, 0]);
        let y = Int32Array::from(vec![1, 1, 0, 0]);
        // Z-Order (x,y) coordinates:
        // (0,0): x=0, y=0. Normalized to bits. 
        // Interleave bit 0 of x, then bit 0 of y (assuming 1 bit for simplicity in mental model)
        // (0,0) -> 00 (0)
        // (0,1) -> 01 (1)
        // (1,0) -> 10 (2)
        // (1,1) -> 11 (3)
        // Sorted: (0,0), (0,1), (1,0), (1,1)
        // x: [0, 0, 1, 1], y: [0, 1, 0, 1]
        
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(x), Arc::new(y)])?;
        table.write_async(vec![batch]).await?;
        table.commit_async().await?;
        
        let options = CompactionOptions {
            target_file_size_bytes: 1024 * 1024,
            min_file_size_bytes: 1024 * 1024,
            clustering: Some(ClusteringOptions {
                columns: vec!["x".to_string(), "y".to_string()],
                strategy: "zorder".to_string(),
            }),
            ..Default::default()
        };
        
        table.rewrite_data_files_async(Some(options)).await?;
        
        // Read back and verify order
        let results = table.read_async(None, None, None).await?;
        for batch in results {
            let rx = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            let ry = batch.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
            
            assert_eq!(rx.value(0), 0); assert_eq!(ry.value(0), 0);
            assert_eq!(rx.value(1), 0); assert_eq!(ry.value(1), 1);
            assert_eq!(rx.value(2), 1); assert_eq!(ry.value(2), 0);
            assert_eq!(rx.value(3), 1); assert_eq!(ry.value(3), 1);
        }
        
        // Cleanup cache
        let cache_key = format!("{}/{}", uri, "");
        crate::core::cache::LATEST_VERSION_CACHE.invalidate(&cache_key).await;
        
        Ok(())
    }

    #[tokio::test]
    async fn test_hilbert_clustering() -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let uri = format!("file://{}", temp_dir.path().to_str().unwrap());
        let table = crate::Table::new_async(uri.clone()).await?;
        
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int32, false),
            Field::new("y", DataType::Int32, false),
        ]));
        
        // Write data in a way that Hilbert should reorganize
        // Let's use 4 points in a square
        let x = Int32Array::from(vec![1, 0, 1, 0]);
        let y = Int32Array::from(vec![1, 1, 0, 0]);
        
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(x), Arc::new(y)])?;
        table.write_async(vec![batch]).await?;
        table.commit_async().await?;
        
        let options = CompactionOptions {
            target_file_size_bytes: 1024 * 1024,
            min_file_size_bytes: 1024 * 1024,
            clustering: Some(ClusteringOptions {
                columns: vec!["x".to_string(), "y".to_string()],
                strategy: "hilbert".to_string(),
            }),
            ..Default::default()
        };
        
        table.rewrite_data_files_async(Some(options)).await?;
        
        // Read back and verify order
        let results = table.read_async(None, None, None).await?;
        for batch in results {
            let rx = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            let ry = batch.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
            
            // For our implementation of Hilbert (Gray-code interleaving), the order for 2x2 is:
            // (0,0) -> 0
            // (1,0) -> 1
            // (1,1) -> 2
            // (0,1) -> 3
            assert_eq!(rx.value(0), 0); assert_eq!(ry.value(0), 0);
            assert_eq!(rx.value(1), 1); assert_eq!(ry.value(1), 0);
            assert_eq!(rx.value(2), 1); assert_eq!(ry.value(2), 1);
            assert_eq!(rx.value(3), 0); assert_eq!(ry.value(3), 1);
        }
        
        // Cleanup cache
        let cache_key = format!("{}/{}", uri, "");
        crate::core::cache::LATEST_VERSION_CACHE.invalidate(&cache_key).await;
        
        Ok(())
    }
}
