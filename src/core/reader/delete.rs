// Copyright (c) 2026 Richard Albright. All rights reserved.

use anyhow::Result;
use object_store::{path::Path, ObjectStore};
use roaring::RoaringBitmap;

use super::*;

/// Result of fetching a single delete file, before it is merged into the
/// segment-wide bitmap. Kept as an enum so the (cheap) conversion from the
/// parsed representation to a `RoaringBitmap` can be timed separately from
/// the (async) fetch in `load_merged_deletes_inner`.
enum FileDeletes {
    /// Parsed position-delete content, keyed by data-file path. The CPU-bound
    /// filtering to the target data file is deferred to the parallel merge.
    Map(std::sync::Arc<std::collections::HashMap<String, std::collections::HashSet<i64>>>),
    /// A native roaring bitmap read directly from storage.
    Bitmap(RoaringBitmap),
}

impl HybridReader {
    pub async fn load_merged_deletes(&self) -> Result<RoaringBitmap> {
        crate::telemetry::metrics::MERGED_DELETES_CALLS_TOTAL.inc();
        let bitmap = self
            .cached_deletes
            .get_or_try_init(|| async { self.load_merged_deletes_inner().await })
            .await?;
        let t_clone = std::time::Instant::now();
        let cloned = bitmap.clone();
        crate::telemetry::metrics::MERGED_DELETES_PHASE_SECONDS
            .with_label_values(&["clone"])
            .observe(t_clone.elapsed().as_secs_f64());
        Ok(cloned)
    }

    /// Deduplicated view of the segment's delete files, keyed by path.
    ///
    /// The manifest can attach the same physical delete file to a data entry
    /// many times (once per manifest version / matching data entry), which
    /// makes `config.delete_files` contain thousands of duplicate paths.
    /// Merging a file more than once is pure waste (the OR is idempotent), so
    /// every consumer works from this deduplicated list.
    fn unique_delete_files(&self) -> Vec<&crate::core::manifest::DeleteFile> {
        let mut seen = std::collections::HashSet::with_capacity(self.config.delete_files.len());
        self.config
            .delete_files
            .iter()
            .filter(|d| seen.insert(d.file_path.as_str()))
            .collect()
    }

    /// Build a cache key from the delete file list. Delete files are immutable
    /// and identified by path, so the sorted list of paths + the target data
    /// file is a stable fingerprint for the merged bitmap.
    fn merged_deletes_cache_key(&self) -> Option<String> {
        if self.config.delete_files.is_empty() {
            return None;
        }
        let target_path = if let Some(p) = &self.config.parquet_path {
            p.clone()
        } else {
            format!(
                "{}/{}.parquet",
                self.config.base_path, self.config.segment_id
            )
        };
        let mut parts: Vec<&str> = self
            .unique_delete_files()
            .iter()
            .map(|d| d.file_path.as_str())
            .collect();
        parts.sort_unstable();
        Some(format!("{}::{}", target_path, parts.join(",")))
    }

    async fn load_merged_deletes_inner(&self) -> Result<RoaringBitmap> {
        let t_total = std::time::Instant::now();

        let t_key = std::time::Instant::now();
        let cache_key_opt = self.merged_deletes_cache_key();
        crate::telemetry::metrics::MERGED_DELETES_PHASE_SECONDS
            .with_label_values(&["cache_key"])
            .observe(t_key.elapsed().as_secs_f64());

        if let Some(cache_key) = &cache_key_opt {
            let t_lookup = std::time::Instant::now();
            let cached = crate::core::cache::POSITION_DELETE_CACHE
                .get(cache_key)
                .await;
            crate::telemetry::metrics::MERGED_DELETES_PHASE_SECONDS
                .with_label_values(&["merged_cache_lookup"])
                .observe(t_lookup.elapsed().as_secs_f64());
            if let Some(cached) = cached {
                crate::telemetry::metrics::MERGED_DELETES_CACHE_TOTAL
                    .with_label_values(&["hit"])
                    .inc();
                // Convert HashSet<i64> → RoaringBitmap
                let t_conv = std::time::Instant::now();
                let mut bm = RoaringBitmap::new();
                for &pos in cached.iter() {
                    if pos >= 0 && pos <= u32::MAX as i64 {
                        bm.insert(pos as u32);
                    }
                }
                crate::telemetry::metrics::MERGED_DELETES_PHASE_SECONDS
                    .with_label_values(&["merged_cache_convert"])
                    .observe(t_conv.elapsed().as_secs_f64());
                crate::telemetry::metrics::MERGED_DELETES_PHASE_SECONDS
                    .with_label_values(&["total"])
                    .observe(t_total.elapsed().as_secs_f64());
                return Ok(bm);
            }
            crate::telemetry::metrics::MERGED_DELETES_CACHE_TOTAL
                .with_label_values(&["miss"])
                .inc();
        }

        let mut deleted_bitmap = RoaringBitmap::new();

        // Determine target path for Iceberg delete matching
        let target_path = if let Some(p) = &self.config.parquet_path {
            p.clone()
        } else {
            format!(
                "{}/{}.parquet",
                self.config.base_path, self.config.segment_id
            )
        };

        // Deduplicate: the manifest can attach the same physical delete file to
        // a data entry thousands of times, and merging it more than once is
        // pure waste (the OR is idempotent).
        let unique = self.unique_delete_files();
        crate::telemetry::metrics::MERGED_DELETES_FILES_TOTAL
            .with_label_values(&["unique"])
            .inc_by(unique.len() as u64);

        // Separate tasks into native deletes and iceberg position deletes for concurrent fetching
        let mut futures = Vec::new();
        
        for delete_file in unique.iter().copied() {
            if let crate::core::manifest::DeleteContent::Position = &delete_file.content {
                crate::telemetry::metrics::MERGED_DELETES_FILES_TOTAL
                    .with_label_values(&["position"])
                    .inc();
                let path_str = delete_file.file_path.clone();
                let store = self.store.clone();
                let root_uri = self.root_uri.clone();
                
                futures.push(async move {
                    let resolved_path = if path_str.starts_with("file://") {
                        let root_local = root_uri
                            .strip_prefix("file://")
                            .unwrap_or(&root_uri)
                            .trim_end_matches('/');
                        let path_clean = path_str.strip_prefix("file://").unwrap_or(&path_str);

                        if !root_local.is_empty() && path_clean.starts_with(root_local) {
                            path_clean[root_local.len()..]
                                .trim_start_matches('/')
                                .to_string()
                        } else {
                            path_clean.trim_start_matches('/').to_string()
                        }
                    } else {
                        path_str.to_string()
                    };

                    if resolved_path.ends_with(".parquet") || resolved_path.ends_with(".avro") {
                        let reader = crate::core::iceberg::PositionDeleteReader::new(store);
                        match reader.fetch_deletes_map(&resolved_path).await {
                            Ok(map) => FileDeletes::Map(map),
                            Err(e) => {
                                tracing::warn!("Failed to read Iceberg delete file {}: {}", path_str, e);
                                FileDeletes::Map(std::sync::Arc::new(std::collections::HashMap::new()))
                            }
                        }
                    } else {
                        let mut local_bitmap = RoaringBitmap::new();
                        let path = Path::from(path_str.as_str());
                        if let Ok(ret) = store.get(&path).await {
                            if let Ok(bytes) = ret.bytes().await {
                                crate::telemetry::metrics::IO_BYTES_READ_TOTAL.inc_by(bytes.len() as u64);
                                if let Ok(bm) = RoaringBitmap::deserialize_from(&bytes[..]) {
                                    local_bitmap |= bm;
                                }
                            }
                        }
                        FileDeletes::Bitmap(local_bitmap)
                    }
                });
            }
        }
        
        // Wait for all Position deletes concurrently
        let t_join = std::time::Instant::now();
        let results = futures::future::join_all(futures).await;
        crate::telemetry::metrics::MERGED_DELETES_PHASE_SECONDS
            .with_label_values(&["position_join_all"])
            .observe(t_join.elapsed().as_secs_f64());

        // Convert the parsed representations to bitmaps. NOTE: this is kept
        // sequential on purpose — measurement showed the CPU-bound filter +
        // bitmap construction is a tiny fraction of the merge cost, and
        // offloading it to `spawn_blocking` + rayon cost more in scheduling
        // overhead than it saved (see plans/production_readiness_plan.md).
        let t_conv = std::time::Instant::now();
        let mut maps = Vec::new();
        let mut native_bitmaps = Vec::new();
        for r in results {
            match r {
                FileDeletes::Map(m) => maps.push(m),
                FileDeletes::Bitmap(bm) => native_bitmaps.push(bm),
            }
        }

        let target_clean = target_path
            .strip_prefix("file://")
            .unwrap_or(&target_path);
        let mut merged_from_maps = RoaringBitmap::new();
        for m in &maps {
            merged_from_maps |= crate::core::iceberg::PositionDeleteReader::filter_map_to_bitmap(
                m,
                target_clean,
            );
        }
        crate::telemetry::metrics::MERGED_DELETES_PHASE_SECONDS
            .with_label_values(&["position_convert"])
            .observe(t_conv.elapsed().as_secs_f64());

        let t_merge = std::time::Instant::now();
        deleted_bitmap |= merged_from_maps;
        for bm in native_bitmaps {
            deleted_bitmap |= bm;
        }
        crate::telemetry::metrics::MERGED_DELETES_PHASE_SECONDS
            .with_label_values(&["position_merge"])
            .observe(t_merge.elapsed().as_secs_f64());

        // Process deletion vectors (Puffin) sequentially as they are fewer
        let t_dv = std::time::Instant::now();
        for delete_file in unique.iter().copied() {
            if let crate::core::manifest::DeleteContent::DeletionVector {
                puffin_file_path,
                content_offset,
                content_size_in_bytes,
            } = &delete_file.content
            {
                crate::telemetry::metrics::MERGED_DELETES_FILES_TOTAL
                    .with_label_values(&["deletion_vector"])
                    .inc();
                let path = Path::from(puffin_file_path.as_str());
                match self
                    .store
                    .get_range(
                        &path,
                        (*content_offset as u64)
                            ..((*content_offset + *content_size_in_bytes) as u64),
                    )
                    .await
                {
                    Ok(bytes) => {
                        crate::telemetry::metrics::IO_BYTES_READ_TOTAL.inc_by(bytes.len() as u64);
                        match crate::core::puffin::read_deletion_vector_from_bytes(&bytes) {
                            Ok(dv_bitmap) => {
                                deleted_bitmap |= dv_bitmap;
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "Failed to deserialize deletion vector from {}: {}",
                                    puffin_file_path,
                                    e
                                );
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Failed to read deletion vector from Puffin file {}: {}",
                            puffin_file_path,
                            e
                        );
                    }
                }
            }
        }
        crate::telemetry::metrics::MERGED_DELETES_PHASE_SECONDS
            .with_label_values(&["deletion_vector"])
            .observe(t_dv.elapsed().as_secs_f64());

        // Store the merged result in the global cache so other HybridReader
        // instances for the same segment + delete-file set get an instant hit.
        if let Some(cache_key) = cache_key_opt {
            let t_insert = std::time::Instant::now();
            let as_set: std::collections::HashSet<i64> =
                deleted_bitmap.iter().map(|v| v as i64).collect();
            crate::core::cache::POSITION_DELETE_CACHE
                .insert(cache_key, std::sync::Arc::new(as_set))
                .await;
            crate::telemetry::metrics::MERGED_DELETES_PHASE_SECONDS
                .with_label_values(&["merged_cache_insert"])
                .observe(t_insert.elapsed().as_secs_f64());
        }

        crate::telemetry::metrics::MERGED_DELETES_PHASE_SECONDS
            .with_label_values(&["total"])
            .observe(t_total.elapsed().as_secs_f64());

        Ok(deleted_bitmap)
    }

    pub(crate) async fn load_equality_deletes(&self) -> Result<Vec<EqualityDelete>> {
        let t_eq = std::time::Instant::now();
        let mut results = Vec::new();

        for delete_file in self.unique_delete_files().iter().copied() {
            if let crate::core::manifest::DeleteContent::Equality { equality_ids } =
                &delete_file.content
            {
                let path_str = delete_file.file_path.as_str();

                // Relativize path matches load_merged_deletes logic
                let resolved_path = if path_str.starts_with("file://") {
                    let root_local = self
                        .root_uri
                        .strip_prefix("file://")
                        .unwrap_or(&self.root_uri)
                        .trim_end_matches('/');
                    let path_clean = path_str.strip_prefix("file://").unwrap_or(path_str);

                    if !root_local.is_empty() && path_clean.starts_with(root_local) {
                        path_clean[root_local.len()..]
                            .trim_start_matches('/')
                            .to_string()
                    } else {
                        path_clean.trim_start_matches('/').to_string()
                    }
                } else {
                    path_str.to_string()
                };

                // Use provided schema or return error
                let schema = if let Some(s) = &self.iceberg_schema {
                    s.clone()
                } else {
                    return Err(anyhow::anyhow!("Cannot apply equality deletes (ID based) without table schema in HybridReader"));
                };

                let iceberg_reader =
                    crate::core::iceberg::EqualityDeleteReader::new(self.store.clone());
                match iceberg_reader
                    .read_equality_deletes(&resolved_path, equality_ids, &schema)
                    .await
                {
                    Ok(batches) => {
                        if equality_ids.len() == 1 {
                            let field_id = equality_ids[0];
                            if let Some(field) = schema.fields.iter().find(|f| f.id == field_id) {
                                let col_name = field.name.clone();

                                // Collect all values for this column from all batches
                                let mut arrays = Vec::new();
                                for batch in batches {
                                    arrays.push(batch.column(0).clone());
                                }

                                if !arrays.is_empty() {
                                    let array_refs: Vec<&dyn arrow::array::Array> =
                                        arrays.iter().map(|a| a.as_ref()).collect();
                                    match arrow::compute::concat(&array_refs) {
                                        Ok(combined_values) => {
                                            results.push(EqualityDelete {
                                                column_name: col_name,
                                                values: combined_values,
                                            });
                                        }
                                        Err(e) => tracing::warn!(
                                            "Failed to concat equality delete values: {}",
                                            e
                                        ),
                                    }
                                }
                            }
                        } else {
                            tracing::warn!("Multi-column equality deletes not yet optimized");
                        }
                    }
                    Err(e) => tracing::warn!(
                        "Failed to read equality delete file {}: {}",
                        resolved_path,
                        e
                    ),
                }
            }
        }
        crate::telemetry::metrics::MERGED_DELETES_PHASE_SECONDS
            .with_label_values(&["equality"])
            .observe(t_eq.elapsed().as_secs_f64());
        Ok(results)
    }
}
