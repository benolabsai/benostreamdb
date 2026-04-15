// Copyright (c) 2026 Richard Albright. All rights reserved.

use anyhow::Result;
use arrow::record_batch::RecordBatch;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use futures::StreamExt;

use crate::core::planner::{VectorSearchParams, QueryPlanner, FilterExpr, QueryFilter};
use crate::core::reader::HybridReader;
use crate::core::manifest::{ManifestManager, ManifestEntry, IndexFile, IndexAlgorithm};
use crate::core::query::{QueryConfig, execute_vector_search_with_config, VectorSearchRequest};
use super::fluent::TableQuery;
use arrow::datatypes::Schema;
use roaring::RoaringBitmap;
use crate::core::index::gpu::get_global_gpu_context;
use crate::SegmentConfig;

use super::Table;
use crate::core::search::{HybridSearchCoordinator, KeywordSearchParams, ScoredResult};

impl Table {

    pub fn read(&self, filter: Option<&str>, vector_filter: Option<VectorSearchParams>) -> Result<Vec<RecordBatch>> {
        self.runtime().block_on(self.read_async(filter, vector_filter, None))
    }

    pub fn read_with_columns(&self, filter: Option<&str>, vector_filter: Option<VectorSearchParams>, columns: Vec<String>) -> Result<Vec<RecordBatch>> {
        let columns_refs: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();
        self.runtime().block_on(self.read_async(filter, vector_filter, Some(&columns_refs)))
    }

    pub async fn sql(&self, query: &str) -> Result<Vec<RecordBatch>> {
        use datafusion::prelude::SessionContext;
        use crate::core::sql::HyperStreamTableProvider;

        let mut ctx = SessionContext::new();        let _ = crate::core::sql::vector_operators::register_vector_operators(&mut ctx);
        let provider = Arc::new(HyperStreamTableProvider::new(Arc::new(self.clone())));
        ctx.register_table("t", provider)?;
        let df = ctx.sql(query).await?;
        Ok(df.collect().await?)
    }

    pub async fn read_async(&self, filter_str: Option<&str>, vector_filter: Option<VectorSearchParams>, columns: Option<&[&str]>) -> Result<Vec<RecordBatch>> {
        self.read_with_config_async(filter_str, vector_filter, columns, self.query_config.clone()).await
    }

    pub fn query(&self) -> TableQuery<'_> {
        TableQuery::new(self)
    }

    /// Generate a detailed execution plan with hit counts and pruning stats
    pub async fn explain(&self, filter_str: Option<&str>, vector_param: Option<VectorSearchParams>) -> String {
        let manifest_manager = crate::core::manifest::ManifestManager::new(self.store.clone(), "", &self.uri);
        let (_manifest, all_entries, version) = manifest_manager.load_latest_full().await.unwrap_or((crate::core::manifest::Manifest::default(), Vec::new(), 0));
        let total_rows_table: usize = all_entries.iter().map(|e| e.record_count as usize).sum();
        let total_segments_table = all_entries.len();
        
        let mut plan = Vec::new();
        let divider = "-".repeat(60);
        plan.push(divider.clone());
        plan.push(format!("HYPERSTREAM QUERY PLAN [Table: {}]", self.uri));
        
        // 1. Initial Pruning (Partition & Stats)
        let expr = if let Some(f) = filter_str {
            FilterExpr::parse_sql(f, self.arrow_schema()).await.ok()
        } else {
            None
        };

        let planner = QueryPlanner::new();
        let pruned_entries: Vec<(ManifestEntry, Option<IndexFile>)> = if version > 0 {
            planner.prune_entries(&all_entries, expr.as_ref(), vector_param.as_ref())
        } else {
            all_entries.iter().map(|e| (e.clone(), None)).collect()
        };

        let scanned_segments = pruned_entries.len();
        let scanned_rows: usize = pruned_entries.iter().map(|(e, _)| e.record_count as usize).sum();
        
        plan.push(format!("Context Selection:"));
        plan.push(format!("  -> Total Table Scope: {} rows in {} segments", total_rows_table, total_segments_table));
        if scanned_segments < total_segments_table {
            plan.push(format!("  -> Pruning Activity: {} segments pruned via Partition/Stats mapping", total_segments_table - scanned_segments));
        }
        plan.push(format!("  -> Execution Scope: {} rows in {} segments", scanned_rows, scanned_segments));
        plan.push("".to_string());

        // 2. Index Dry-run of Filters
        let mut scalar_hits = scanned_rows;
        let mut access_paths = Vec::new();

        if let Some(ref e) = expr {
            let sub_filters = e.extract_and_conditions();
            let mut total_hits = 0;
            let base_uri = self.uri.clone();
            
            // Convert file:// URI to filesystem path for directory listing
            let fs_path = if base_uri.starts_with("file://") {
                base_uri.strip_prefix("file://").unwrap_or(&base_uri).to_string()
            } else {
                base_uri.clone()
            };
            
            // Map to track which index type was used for each sub-filter
            let mut filter_index_types: HashMap<String, HashSet<&'static str>> = HashMap::new();

            for (entry, _) in &pruned_entries {
                let file_path_str = entry.file_path.clone();
                let segment_id = file_path_str
                   .split('/')
                   .next_back()
                   .unwrap_or(&file_path_str)
                   .strip_suffix(".parquet")
                   .unwrap_or(&file_path_str);

                let config = SegmentConfig::new(&base_uri, segment_id)
                   .with_parquet_path(entry.file_path.clone())
                   .with_index_files(entry.index_files.clone())
                   .with_delete_files(entry.delete_files.clone())
                   .with_record_count(entry.record_count as u64);

                let reader = HybridReader::new(config, self.store.clone(), &base_uri);
                
                let mut seg_bm: Option<roaring::RoaringBitmap> = None;
                for sub_f in &sub_filters {
                    // Detect access path by checking for actual index files on disk
                    let path = {
                        // Check for inverted index (.inv.parquet files)
                        let inv_pattern = format!("{}.{}.inv.parquet", segment_id, sub_f.column);
                        let has_inverted = std::fs::read_dir(&fs_path)
                            .ok()
                            .and_then(|dir| {
                                dir.flatten()
                                    .find(|e| e.file_name().to_string_lossy().contains(&inv_pattern))
                            })
                            .is_some();
                        
                        if has_inverted {
                            "Inverted Index (Parquet)"
                        } else {
                            // Check for bitmap index (.idx files)
                            let bitmap_pattern = format!("{}.{}.idx", segment_id, sub_f.column);
                            let has_bitmap = std::fs::read_dir(&fs_path)
                                .ok()
                                .and_then(|dir| {
                                    dir.flatten()
                                        .find(|e| e.file_name().to_string_lossy().contains(&bitmap_pattern))
                                })
                                .is_some();
                            
                            if has_bitmap {
                                "Bitmap Index (.idx)"
                            } else {
                                "Full Scan"
                            }
                        }
                    };
                    filter_index_types.entry(sub_f.column.clone()).or_default().insert(path);

                    if let Ok(Some(bm)) = reader.get_scalar_filter_bitmap(sub_f).await {
                        match seg_bm {
                            Some(ref mut existing) => *existing &= bm,
                            None => seg_bm = Some(bm),
                        }
                    }
                }

                if let Some(bm) = seg_bm {
                    total_hits += bm.len() as usize;
                } else {
                    total_hits += entry.record_count as usize;
                }
            }
            scalar_hits = total_hits;
            
            for sub_f in &sub_filters {
                let paths: Vec<&'static str> = filter_index_types.get(&sub_f.column)
                    .map(|s: &HashSet<&'static str>| s.iter().cloned().collect())
                    .unwrap_or_else(|| vec!["Scan"]);
                access_paths.push(format!("  -> Filter (col: {}, op: {}, access: {})", sub_f.column, sub_f.op_to_string(), paths.join(", ")));
            }
        }

        // 3. Scalar Plan (First: Pre-filter the dataset)
        if !access_paths.is_empty() {
            plan.push("Scalar Execution:".to_string());
            for path in access_paths {
                plan.push(path);
            }
            let pct = if scanned_rows > 0 { (scalar_hits as f32 / scanned_rows as f32) * 100.0 } else { 0.0 };
            plan.push(format!("     [Selectivity: {} / {} rows ({:.2}%)]", scalar_hits, scanned_rows, pct));
            plan.push("".to_string());
        }

        // 4. Vector Plan (Second: Vector search on pre-filtered rows)
        if let Some(ref vs) = vector_param {
            plan.push("Vector Execution:".to_string());
            plan.push(format!("  -> VectorSearch (col: {}, k: {}, metric: {:?})", vs.column, vs.k, vs.metric));
            
            // Check for vector index by detecting .hnsw.graph files on disk
            let mut has_vector_index = false;
            
            // Convert file:// URI to filesystem path
            let fs_path = if self.uri.starts_with("file://") {
                self.uri.strip_prefix("file://").unwrap_or(&self.uri)
            } else {
                &self.uri
            };
            
            for (entry, _) in &pruned_entries {
                let segment_id = entry.file_path
                    .split('/')
                    .next_back()
                    .unwrap_or(&entry.file_path)
                    .strip_suffix(".parquet")
                    .unwrap_or(&entry.file_path);
                
                // Look for HNSW graph files for this column: segment_id.{column_name}.cluster_*.hnsw.graph
                let hnsw_pattern_prefix = format!("{}.{}.cluster_", segment_id, vs.column);
                let hnsw_pattern_suffix = ".hnsw.graph";
                
                // Try to list files in the table directory to detect index files
                if let Ok(dirs) = std::fs::read_dir(fs_path) {
                    for entry in dirs.flatten() {
                        if let Some(filename) = entry.file_name().to_str() {
                            if filename.starts_with(&hnsw_pattern_prefix) && filename.ends_with(hnsw_pattern_suffix) {
                                has_vector_index = true;
                                break;
                            }
                        }
                    }
                }
                if has_vector_index {
                    break;
                }
            }
            
            let access_mode = if has_vector_index { "HNSW-IVF Cluster Index" } else { "Brute Force Scan (No Index)" };
            plan.push(format!("     [Access: {}] [Eligibility: {} rows]", access_mode, scalar_hits));
            plan.push("".to_string());
        }
        
        plan.push(format!("Final Retrieval:"));
        plan.push(format!("  -> ParallelRead (threads: {}, format: Parquet)", self.query_config.max_parallel_readers.unwrap_or(16)));
        plan.push(divider);
        
        plan.join("\n")
    }

    pub fn filter(&self, expr: &str) -> TableQuery<'_> {
        TableQuery::new(self).filter(expr)
    }

    pub async fn read_with_config_async(
        &self, 
        filter_str: Option<&str>, 
        vector_filter: Option<VectorSearchParams>, 
        columns: Option<&[&str]>,
        config: QueryConfig
    ) -> Result<Vec<RecordBatch>> {
        let expr = match filter_str {
            Some(f) => {
                let schema = self.arrow_schema();
                Some(FilterExpr::parse_sql(f, schema).await?)
            }
            _ => None,
        };
        self.read_expr_with_config_async(expr, vector_filter, columns, config, filter_str).await
    }

    pub async fn read_expr_with_config_async(
        &self,
        expr: Option<FilterExpr>,
        vector_filter: Option<VectorSearchParams>,
        columns: Option<&[&str]>,
        config: QueryConfig,
        filter_str: Option<&str>,
    ) -> Result<Vec<RecordBatch>> {
        use futures::StreamExt;

        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (_manifest, all_entries, version) = match manifest_manager.load_latest_full().await {
            Ok((m, e, v)) => (m, e, v),
            Err(_) => {
                if manifest_manager.exists().await.unwrap_or(false) {
                   (crate::core::manifest::Manifest::default(), Vec::new(), 0)
                } else {
                   let segments = self.list_segments_from_store().await.unwrap_or_default();
                   (crate::core::manifest::Manifest::default(), segments, 0)
                }
            }
        };        let entries_to_read = if version > 0 {
            if expr.is_some() || vector_filter.is_some() {
                let planner = QueryPlanner::new();
                planner.prune_entries(&all_entries, expr.as_ref(), vector_filter.as_ref()).into_iter().map(|(e, _)| e).collect()
            } else {
                all_entries.clone()
            }
        } else {
            // Version 0. Check if we want autodetection.
            if manifest_manager.exists().await.unwrap_or(false) {
                Vec::new() // Strictly follow the empty manifest
            } else {
                all_entries.clone() // Already contains discovered segments if in autodetection path
            }
        };
        
        // --- SMART HYBRID TRIGGER ---
        // If we have both a vector filter AND a text filter on a BM25/Inverted indexed column,
        // we switch to the Hybrid Coordinator path.
        if let (Some(ref vs_params), Some(ref e)) = (&vector_filter, &expr) {
             let manifest = self.manifest().await?;
             let filtered_cols = e.get_referenced_columns();
             
             let current_schema = manifest.schemas.iter()
                 .find(|s| s.schema_id == manifest.current_schema_id);
             
             let has_bm25_index = current_schema.map(|s| {
                 s.fields.iter().any(|f| {
                     let matches_col = filtered_cols.contains(&f.name);
                     let has_index = f.indexes.iter().any(|idx| {
                         matches!(idx, IndexAlgorithm::Bm25 { .. })
                     });
                     if matches_col {
                         println!("Smart Trigger Check: Column '{}' has BM25 index: {}", f.name, has_index);
                         if !has_index {
                             println!("  Found indexes: {:?}", f.indexes);
                         }
                     }
                     matches_col && has_index
                 })
             }).unwrap_or(false);

             if has_bm25_index {
                 tracing::info!("Smart Trigger: Executing Hybrid Search (RRF) for columns {:?}", filtered_cols);
                 let coordinator = HybridSearchCoordinator::new();
                 
                 // Extract search terms from FilterExpr for the BM25 engine
                 let extracted_query = {
                     let conditions = e.extract_and_conditions();
                     let mut terms = Vec::new();
                     for f in conditions {
                         if filtered_cols.contains(&f.column) {
                            if let Some(v) = &f.min {
                                if let Some(s) = v.as_str() { terms.push(s.to_string()); }
                            }
                            if let Some(vals) = &f.values {
                                for v in vals {
                                    if let Some(s) = v.as_str() { terms.push(s.to_string()); }
                                }
                            }
                         }
                     }
                     if terms.is_empty() {
                         filter_str.unwrap_or("").to_string()
                     } else {
                         terms.join(" ")
                     }
                 };

                 let keyword_params = KeywordSearchParams {
                     column: filtered_cols.iter().next().unwrap().clone(),
                     query: extracted_query,
                 };

                 let scored_results = coordinator.execute_hybrid(
                     self,
                     filter_str,
                     Some(vs_params.clone()),
                     Some(keyword_params),
                     vs_params.k,
                     config.rrf_k,
                 ).await?;

                 // Convert ScoredResults back to RecordBatches by fetching from Parquet
                 // This is a simplified version of the final row-fetcher
                 return self.fetch_results_by_id(scored_results, columns).await;
             }
        }

        // Handle standard vector search
        if let Some(ref vs_params) = vector_filter {
             // 1. Search Disk
             let request = VectorSearchRequest::new(
                 vs_params.column.clone(),
                 vs_params.query.clone(),
                 vs_params.k,
                 vs_params.metric,
             )
             .with_filter(expr.clone())
             .with_config(config.clone())
             .with_ef_search(vs_params.ef_search)
             .with_columns(columns.map(|c| c.iter().map(|s| s.to_string()).collect()));
             
             let mut results = execute_vector_search_with_config(
                entries_to_read.clone(),
                self.store.clone(),
                self.data_store.clone(),
                &self.uri,
                request,
            ).await?;

             // 2. Search Memory
             let memory_hits = {
                 let idx = self.indexing.memory_index.read().unwrap();
                 if let Some(mem_idx) = idx.as_ref() {
                     let filter_bitmap = if let Some(ref e) = expr {
                         let buffer = self.write_buffer.read().unwrap();
                         let mut bitmap = RoaringBitmap::new();
                          let mut offset = 0;
                          let planner = QueryPlanner::new();
                          for batch in buffer.iter() {
                               if let Ok(mask) = planner.evaluate_expr(batch, e) {
                                   for i in 0..batch.num_rows() {
                                      if mask.value(i) {
                                          bitmap.insert((offset + i) as u32);
                                      }
                                  }
                              }
                              offset += batch.num_rows();
                          }
                          Some(bitmap)
                      } else {
                          None
                      };
                      mem_idx.search(&vs_params.query, vs_params.k, filter_bitmap.as_ref())
                  } else {
                      vec![]
                  }
              };

              if !memory_hits.is_empty() {
                  let buffer = self.write_buffer.read().unwrap();
                  if let Some(first) = buffer.first() {
                      let schema = first.schema();
                      let batch_offsets: Vec<usize> = buffer.iter().scan(0, |state, b| {
                          let start = *state;
                          *state += b.num_rows();
                          Some(start)
                      }).collect();
                      
                      let mut result_rows = Vec::new();
                      for (id, _dist) in &memory_hits {
                          for (i, offset) in batch_offsets.iter().enumerate().rev() {
                              if *id >= *offset {
                                  let row_idx = *id - offset;
                                  if i < buffer.len() && row_idx < buffer[i].num_rows() {
                                      result_rows.push(buffer[i].slice(row_idx, 1));
                                  }
                                  break;
                              }
                          }
                      }
                      
                      if !result_rows.is_empty() {
                          let mem_batch = arrow::compute::concat_batches(&schema, result_rows.iter().collect::<Vec<&RecordBatch>>())?;
                          
                          // Append _distance column to match disk search schema
                          let mut new_fields = schema.fields().to_vec();
                          new_fields.push(std::sync::Arc::new(arrow::datatypes::Field::new("distance", arrow::datatypes::DataType::Float32, false)));
                          let new_schema = std::sync::Arc::new(arrow::datatypes::Schema::new(new_fields));
                          
                          let mut new_columns = mem_batch.columns().to_vec();
                          let distance_array = arrow::array::Float32Array::from(memory_hits.iter().map(|(_, dist)| *dist).collect::<Vec<f32>>());
                          new_columns.push(std::sync::Arc::new(distance_array));
                          
                          if let Ok(dist_batch) = RecordBatch::try_new(new_schema, new_columns) {
                              results.push(dist_batch);
                          }
                      }
                  }
              }

              return Ok(results);
        }

        // Extract Iceberg schema from the already-loaded manifest to avoid
        // redundant manifest loads inside each per-segment read.
        let iceberg_schema = _manifest.schemas.iter()
            .find(|s| s.schema_id == _manifest.current_schema_id)
            .cloned();
        let iceberg_schema_arc = iceberg_schema.map(Arc::new);

        // Capture current GPU context so it can be propagated into each async
        // worker closure (thread_local is per-thread, not per-future).
        let current_gpu_context = get_global_gpu_context();

        let expr_arc = expr.map(Arc::new);
        let concurrency = config.max_parallel_readers.unwrap_or_else(|| {
            std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
        });
        let stream = futures::stream::iter(entries_to_read)
            .map(|entry| {
                let expr_clone = expr_arc.clone();
                let schema_clone = iceberg_schema_arc.clone();
                let ctx = current_gpu_context.clone();
                async move {
                    if let Some(c) = ctx {
                        crate::core::index::gpu::set_global_gpu_context(Some(c));
                    }
                    self.read_segment_expr(
                        &entry, expr_clone.as_deref(), version, columns,
                        schema_clone.as_deref(),
                    ).await
                }
            })
            .buffer_unordered(concurrency);

        let results: Vec<Result<Vec<RecordBatch>>> = stream.collect().await;
        let mut all_batches = Vec::new();
        for (i, res) in results.into_iter().enumerate() {
            match res {
                Ok(b_vec) => {
                     all_batches.extend(b_vec);
                },
                Err(e) => tracing::error!("Error reading batch {}: {}", i, e),
            }
        }

        // --- Read from In-Memory Write Buffer ---
        {
            let buffer = self.write_buffer.read().unwrap();
            if !buffer.is_empty() {
                if let Some(ref e) = expr_arc {
                    let planner = QueryPlanner::new();
                    for batch in buffer.iter() {
                         let batch_to_scan = if let Some(cols) = columns {
                             let indices: Vec<usize> = cols.iter()
                                .filter_map(|name| batch.schema().index_of(name).ok())
                                .collect();
                             batch.project(&indices).unwrap_or(batch.clone())
                         } else {
                             batch.clone()
                         };

                         if let Ok(filtered) = planner.filter_expr(&batch_to_scan, e) {
                             if filtered.num_rows() > 0 {
                                 all_batches.push(filtered);
                             }
                         }
                    }
                } else {
                     for batch in buffer.iter() {
                         if let Some(cols) = columns {
                             let indices: Vec<usize> = cols.iter()
                                .filter_map(|name| batch.schema().index_of(name).ok())
                                .collect();
                             if let Ok(projected) = batch.project(&indices) {
                                 all_batches.push(projected);
                             }
                         } else {
                             all_batches.push(batch.clone());
                         }
                     }
                }
            }
        }
        

        Ok(all_batches)
    }

    pub async fn read_filter_async(
        &self,
        filters: Vec<QueryFilter>,
        vector_filter: Option<VectorSearchParams>,
        columns: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>> {
        self.read_filter_with_config_async(filters, vector_filter, columns, self.query_config.clone()).await
    }

    pub async fn read_filter_with_config_async(
        &self,
        filters: Vec<QueryFilter>,
        vector_filter: Option<VectorSearchParams>,
        columns: Option<&[&str]>,
        config: QueryConfig,
    ) -> Result<Vec<RecordBatch>> {
        let expr = FilterExpr::from_filters(filters);
        self.read_expr_with_config_async(expr, vector_filter, columns, config, None).await
    }

    pub async fn read_segment_expr(
        &self,
        entry: &ManifestEntry,
        expr: Option<&FilterExpr>,
        manifest_version: u64,
        columns: Option<&[&str]>,
        cached_iceberg_schema: Option<&crate::core::manifest::Schema>,
    ) -> Result<Vec<RecordBatch>> {
        let file_path_str = entry.file_path.clone();
        let segment_id = file_path_str.split('/').next_back().unwrap_or(&file_path_str)
            .strip_suffix(".parquet").unwrap_or(&file_path_str);

        // Use cached schema if provided by caller; otherwise fall back to manifest load.
        let iceberg_schema = if let Some(schema) = cached_iceberg_schema {
            Some(schema.clone())
        } else {
            let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
            let (manifest, _, _) = manifest_manager.load_latest_full().await.unwrap_or_default();
            manifest.schemas.iter().find(|s| s.schema_id == manifest.current_schema_id).cloned()
        };

        // Resolve partition-aware path
        let path = std::path::Path::new(&file_path_str);
        let rel_parent = path.parent().and_then(|p| p.to_str()).unwrap_or("");
        let full_base_path = if rel_parent.is_empty() {
             self.uri.clone()
        } else {
             format!("{}/{}", self.uri, rel_parent)
        };

        let config = SegmentConfig::new(&full_base_path, segment_id)
            .with_parquet_path(entry.file_path.clone())
            .with_data_store(self.data_store.clone().unwrap_or(self.store.clone()))
            .with_delete_files(entry.delete_files.clone())
            .with_index_files(entry.index_files.clone())
            .with_file_size(entry.file_size_bytes as u64)
            .with_index_all(self.indexing.index_all)
            .with_columns_to_index(self.indexing.index_columns.read().unwrap().clone());

        let mut reader = HybridReader::new(config, self.store.clone(), &self.uri);
        if let Some(s) = &iceberg_schema {
            reader = reader.with_iceberg_schema(s.clone());
        }

        let full_schema = if let Some(schema) = &iceberg_schema {
             Arc::new(schema.to_arrow())
        } else {
             self.arrow_schema()
        };

        let target_schema = if let Some(cols) = columns {
             let fields: Vec<arrow::datatypes::Field> = cols.iter()
                 .filter_map(|name| full_schema.field_with_name(name).ok().cloned())
                 .collect();
             Some(Arc::new(Schema::new(fields)))
        } else {
             Some(full_schema)
        };

        if manifest_version == 0 || expr.is_none() {
            let mut stream = reader.stream_all(target_schema).await?;
            let mut batches = Vec::new();
            while let Some(batch_result) = stream.next().await {
                batches.push(batch_result?);
            }
            return Ok(batches);
        }

        let expr = expr.unwrap();
        let and_filters = expr.extract_and_conditions();

        // Try to use index for the FIRST filter that has one
        let mut batches = Vec::new();
        let mut index_used = false;

        for filter in &and_filters {
             if let Ok(indexed_batches) = reader.query_index_first(filter, target_schema.clone()).await {
                 batches = indexed_batches;
                 index_used = true;
                 break;
             } 
        }

        if !index_used {
            // Point Selection Optimization: Check Bloom Filters before scanning
            for filter in &and_filters {
                if let Some(vals) = &filter.values {
                    if vals.len() == 1 {
                        if !reader.check_bloom_filter(&filter.column, &vals[0]).await.unwrap_or(true) {
                            tracing::debug!("Bloom Filter Pruned segment: {} for col: {}", segment_id, filter.column);
                            return Ok(vec![]);
                        }
                    }
                } else if let (Some(min), Some(max)) = (&filter.min, &filter.max) {
                    if min == max {
                        if !reader.check_bloom_filter(&filter.column, min).await.unwrap_or(true) {
                            tracing::debug!("Bloom Filter Pruned segment: {} for col: {}", segment_id, filter.column);
                            return Ok(vec![]);
                        }
                    }
                }
            }
            
            let mut stream = reader.stream_all(target_schema).await?;
            while let Some(batch_result) = stream.next().await {
                batches.push(batch_result?);
            }
        }

        let planner = QueryPlanner::new();
        let mut filtered_batches = Vec::new();
        for batch in batches {
            match planner.filter_expr(&batch, expr) {
                Ok(filtered) => {
                    if filtered.num_rows() > 0 {
                         filtered_batches.push(filtered);
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to evaluate filter expression on batch: {}", e);
                }
            }
        }
        Ok(filtered_batches)
    }

    pub async fn read_segment_multi(
        &self,
        entry: &ManifestEntry,
        filters: &[QueryFilter],
        manifest_version: u64,
        columns: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>> {
        let expr = FilterExpr::from_filters(filters.to_vec());
        self.read_segment_expr(entry, expr.as_ref(), manifest_version, columns, None).await
    }

    pub async fn read_segment(
        &self,
        entry: &ManifestEntry,
        query_filter_opt: Option<&QueryFilter>,
        manifest_version: u64,
        columns: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>> {
        let filters = match query_filter_opt {
            Some(f) => vec![f.clone()],
            None => vec![],
        };
        self.read_segment_multi(entry, &filters, manifest_version, columns).await
    }

    pub async fn stream_all(&self, columns: Option<&[&str]>) -> Result<futures::stream::BoxStream<'static, Result<RecordBatch>>> {
        use futures::StreamExt;
        let batches = self.read_async(None, None, columns).await?;
        Ok(futures::stream::iter(batches.into_iter().map(Ok)).boxed())
    }

    async fn list_segments_from_store(&self) -> Result<Vec<ManifestEntry>> {
        use futures::StreamExt;
        let mut entries = Vec::new();
        let mut stream = self.store.list(None);
        while let Some(res) = stream.next().await {
            if let Ok(meta) = res {
                let path = meta.location.to_string();
                if (!path.contains("/") || path.contains("data/")) && path.ends_with(".parquet") 
                   && !path.contains(".inv.parquet") && !path.contains(".hnsw.") {
                       entries.push(ManifestEntry {
                           file_path: path,
                           file_size_bytes: meta.size as i64,
                           ..Default::default()
                       });
                }
            }
        }
        Ok(entries)
    }

    pub async fn execute_vector_search_as_scored(
        &self,
        params: VectorSearchParams,
    ) -> Result<Vec<ScoredResult>> {
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (_, all_entries, _) = manifest_manager.load_latest_full().await?;
        
        let request = VectorSearchRequest::new(
            params.column.clone(),
            params.query.clone(),
            params.k,
            params.metric,
        ).with_ef_search(params.ef_search);

        // For now, we reuse the existing vector search and convert RecordBatches to ScoredResults
        // In a future optimization, we'll return ScoredResults directly from the reader to avoid Parquet I/O if possible
        let batches = execute_vector_search_with_config(
            all_entries,
            self.store.clone(),
            self.data_store.clone(),
            &self.uri,
            request,
        ).await?;

        let mut scored_results = Vec::new();
        for batch in batches {
            // RecordBatch results from vector search include a "distance" column
            let dist_col = batch.column(batch.num_columns() - 1)
                .as_any()
                .downcast_ref::<arrow::array::Float32Array>()
                .ok_or_else(|| anyhow::anyhow!("Missing distance column in vector search result"))?;
            
            // Note: Parallel vector search currently doesn't preserve segment/row IDs in the final RecordBatch
            // We'll need to add those to the schema in Phase 3. 
            // For Phase 2, we use a synthetic segment ID "vector_path"
            for i in 0..batch.num_rows() {
                scored_results.push(ScoredResult {
                    segment_id: "vector_path".to_string(),
                    row_id: i as u32,
                    score: dist_col.value(i),
                });
            }
        }
        Ok(scored_results)
    }

    pub async fn execute_keyword_search_as_scored(
        &self,
        params: KeywordSearchParams,
    ) -> Result<Vec<ScoredResult>> {
        let manifest = self.manifest().await?;
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let all_entries = manifest_manager.load_all_entries(&manifest).await?;
        
        let mut all_scored = Vec::new();
        for entry in all_entries {
            let file_path_str = entry.file_path.clone();
            let segment_id = file_path_str
                .split('/')
                .next_back()
                .unwrap_or(&file_path_str)
                .strip_suffix(".parquet")
                .unwrap_or(&file_path_str);

            let config = SegmentConfig::new(&self.uri, segment_id)
                .with_parquet_path(entry.file_path.clone())
                .with_index_files(entry.index_files.clone())
                .with_record_count(entry.record_count as u64);

            let reader = HybridReader::new(config, self.store.clone(), &self.uri);
            let matches = reader.keyword_search_index(&params.column, &params.query, 1000, None).await?;
            
            for (row_id, score) in matches {
                all_scored.push(ScoredResult {
                    segment_id: segment_id.to_string(),
                    row_id: row_id as u32,
                    score,
                });
            }
        }
        Ok(all_scored)
    }

    /// Helper to fetch full RecordBatches for a set of fused IDs
    pub async fn fetch_results_by_id(
        &self,
        results: Vec<ScoredResult>,
        columns: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>> {
        if results.is_empty() {
            return Ok(vec![]);
        }

        // Group by segment to minimize I/O
        let mut by_segment: HashMap<String, Vec<(u32, f32)>> = HashMap::new();
        for r in results {
            by_segment.entry(r.segment_id).or_default().push((r.row_id, r.score));
        }

        let mut final_batches = Vec::new();
        let manifest = self.manifest().await?;
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let all_entries = manifest_manager.load_all_entries(&manifest).await?;
        
        for (seg_id, rows) in by_segment {
            if seg_id == "vector_path" { continue; } 

            // Find segment in all entries
            let entry = all_entries.iter()
                .find(|e| e.file_path.contains(&seg_id))
                .ok_or_else(|| anyhow::anyhow!("Segment {} not found in manifest", seg_id))?;

            let config = SegmentConfig::new(&self.uri, &seg_id)
                .with_parquet_path(entry.file_path.clone())
                .with_index_files(entry.index_files.clone());

            let reader = HybridReader::new(config, self.store.clone(), &self.uri);
            
            // Convert fused row IDs and scores to RecordBatch
            let batch = reader.read_rows_by_id(rows, columns).await?;
            final_batches.push(batch);
        }

        Ok(final_batches)
    }
}
