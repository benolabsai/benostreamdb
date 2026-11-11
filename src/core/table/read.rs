// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use anyhow::Result;
use arrow::record_batch::RecordBatch;
use futures::StreamExt;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::fluent::TableQuery;
use crate::core::index::gpu::get_thread_gpu_context;
use crate::core::index::VectorValue;
use crate::core::manifest::{IndexAlgorithm, IndexFile, ManifestEntry, ManifestManager};
use crate::core::planner::{FilterExpr, QueryFilter, QueryPlanner, VectorSearchParams};
use crate::core::query::{QueryConfig, VectorSearchRequest};
use arrow::datatypes::Schema;
use roaring::RoaringBitmap;

use super::primary_key::PrimaryKeyFilter;
use super::Table;
use crate::core::search::{HybridSearchCoordinator, KeywordSearchParams, ScoredResult};
use futures::stream::BoxStream;

/// Default IVF probe count for a table-level vector index.
///
/// Scales with the cluster count so small indexes probe most clusters (recall
/// parity with a flat HNSW graph) while large indexes stay bounded:
/// `sqrt(n_lists) * 5`, clamped to `[16, n_lists]`. The old fixed default of
/// 16 probed only ~36% of a 44-cluster index and capped `lastfm-64-dot` recall
/// at 0.90; probing 33 recovers 0.987.
fn default_probes(n_lists: usize) -> usize {
    if n_lists == 0 {
        return 16;
    }
    (((n_lists as f64).sqrt() as usize) * 5)
        .max(16)
        .min(n_lists)
}

fn search_memory_buffer_batches(
    batches: &[RecordBatch],
    mem_idx: &crate::core::index::memory::InMemoryVectorIndex,
    vs_params_list: &[VectorSearchParams],
    expr: Option<&FilterExpr>,
    columns: Option<&[&str]>,
    table_schema: &arrow::datatypes::SchemaRef,
) -> Result<Option<RecordBatch>> {
    if batches.is_empty() {
        return Ok(None);
    }

    let filter_bitmap = if let Some(e) = expr {
        let mut bitmap = RoaringBitmap::new();
        let mut offset = 0usize;
        let planner = QueryPlanner::new();
        for batch in batches {
            let aligned = if batch.schema() == *table_schema {
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
            if let Ok(mask) = planner.evaluate_expr(&aligned, e) {
                for i in 0..aligned.num_rows() {
                    if mask.value(i) {
                        bitmap.insert((offset + i) as u32);
                    }
                }
            }
            offset += aligned.num_rows();
        }
        Some(bitmap)
    } else {
        None
    };

    let mut memory_hits = Vec::new();
    for vs_params in vs_params_list {
        let hits = mem_idx.search(&vs_params.query, vs_params.k, filter_bitmap.as_ref());
        memory_hits.extend(hits);
    }

    if memory_hits.is_empty() {
        return Ok(None);
    }

    let batch_offsets: Vec<usize> = batches
        .iter()
        .scan(0, |state, b| {
            let start = *state;
            *state += b.num_rows();
            Some(start)
        })
        .collect();

    let first_schema = batches[0].schema();
    let schemas_differ = batches.iter().any(|b| b.schema() != first_schema);
    let target_concat_schema = if schemas_differ {
        table_schema.clone()
    } else {
        first_schema
    };

    let mut result_rows = Vec::new();
    let mut kept_distances = Vec::with_capacity(memory_hits.len());
    for (id, dist) in &memory_hits {
        for (i, offset) in batch_offsets.iter().enumerate().rev() {
            if *id >= *offset {
                let row_idx = *id - offset;
                if i < batches.len() && row_idx < batches[i].num_rows() {
                    let slice = batches[i].slice(row_idx, 1);
                    let aligned_slice = if slice.schema() == target_concat_schema {
                        slice
                    } else {
                        let mut cols = Vec::with_capacity(target_concat_schema.fields().len());
                        for field in target_concat_schema.fields() {
                            let col = if let Some(c) = slice.column_by_name(field.name()) {
                                c.clone()
                            } else {
                                arrow::array::new_null_array(field.data_type(), slice.num_rows())
                            };
                            cols.push(col);
                        }
                        RecordBatch::try_new(target_concat_schema.clone(), cols)
                            .unwrap_or_else(|_| slice.clone())
                    };
                    result_rows.push(aligned_slice);
                    kept_distances.push(*dist);
                }
                break;
            }
        }
    }

    if result_rows.is_empty() {
        return Ok(None);
    }

    let mem_batch = arrow::compute::concat_batches(
        &target_concat_schema,
        result_rows.iter().collect::<Vec<&RecordBatch>>(),
    )?;

    let projected_batch = if let Some(cols) = columns {
        let indices: Vec<usize> = cols
            .iter()
            .filter_map(|name| mem_batch.schema().index_of(name).ok())
            .collect();
        mem_batch.project(&indices).unwrap_or(mem_batch.clone())
    } else {
        mem_batch
    };

    let projected_schema = projected_batch.schema();
    let mut new_fields = projected_schema.fields().to_vec();
    new_fields.push(std::sync::Arc::new(arrow::datatypes::Field::new(
        "distance",
        arrow::datatypes::DataType::Float32,
        false,
    )));
    let new_schema = std::sync::Arc::new(arrow::datatypes::Schema::new(new_fields));

    let mut new_columns = projected_batch.columns().to_vec();
    let distance_array = arrow::array::Float32Array::from(kept_distances);
    new_columns.push(std::sync::Arc::new(distance_array));

    Ok(RecordBatch::try_new(new_schema, new_columns).ok())
}

impl Table {
    pub fn read(
        &self,
        filter: Option<&str>,
        vector_filters: Option<Vec<VectorSearchParams>>,
    ) -> Result<Vec<RecordBatch>> {
        self.runtime()
            .block_on(self.read_async(filter, vector_filters, None))
    }

    pub fn read_with_columns(
        &self,
        filter: Option<&str>,
        vector_filters: Option<Vec<VectorSearchParams>>,
        columns: Vec<String>,
    ) -> Result<Vec<RecordBatch>> {
        let columns_refs: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();
        self.runtime()
            .block_on(self.read_async(filter, vector_filters, Some(&columns_refs)))
    }

    #[tracing::instrument(skip(self))]
    pub async fn sql(&self, query: &str) -> Result<Vec<RecordBatch>> {
        use crate::core::sql::optimizer::VectorSearchConfig;
        use crate::core::sql::BenoStreamTableProvider;
        use datafusion::prelude::{SessionConfig, SessionContext};

        // Apply any `/*+ ... */` optimizer hints to the session configuration
        // so the vector-search optimizer rule can pick them up.
        let mut session_config = SessionConfig::new();
        if let Some(hints) = VectorSearchConfig::extract_sql_hints(query) {
            let vs_config = VectorSearchConfig::from_sql_hints(&hints)?;
            session_config.options_mut().extensions.insert(vs_config);
        }

        let mut ctx = SessionContext::new_with_config(session_config);
        // Register the full custom function surface (vector scalar UDFs, vector
        // aggregates, JSON UDFs, and graph UDAFs). Without this, connector
        // pass-through queries (`option("query", "SELECT vector_sum(...) ...")`)
        // fail with "Invalid function" because this ad-hoc context is separate
        // from the engine's primary `BenoStreamSession`.
        crate::core::sql::udf::register_all_custom_udfs(&mut ctx);
        // Graph traversal table functions (`FROM graph_neighbors(...)`).
        crate::core::sql::graph_udf::register_graph_table_functions(&mut ctx);
        crate::core::sql::subscribe_table_function::register_subscribe_table_functions(&mut ctx);
        let _ = crate::core::sql::vector_operators::register_vector_operators(&mut ctx);
        let provider = Arc::new(BenoStreamTableProvider::new(Arc::new(self.clone())));
        ctx.register_table("t", provider)?;
        let df = ctx.sql(query).await?;
        Ok(df.collect().await?)
    }

    /// Runs `query` with several tables registered under explicit aliases.
    ///
    /// <p>This is the multi-table counterpart of [`Table::sql`], which registers
    /// exactly one table as `t`. Functions that need a document table and an
    /// edge table in the same statement (for example a graph-expanded RAG
    /// lookup) pass `[("t", docs_uri), ("e", edges_uri)]` and then reference
    /// `t` and `'e'` in the query.</p>
    #[tracing::instrument(skip(self, tables))]
    pub async fn sql_multi(
        &self,
        tables: Vec<(String, String)>,
        query: &str,
    ) -> Result<Vec<RecordBatch>> {
        use crate::core::sql::optimizer::VectorSearchConfig;
        use crate::core::sql::BenoStreamTableProvider;
        use datafusion::prelude::{SessionConfig, SessionContext};

        if tables.is_empty() {
            anyhow::bail!("sql_multi requires at least one table alias");
        }

        let mut session_config = SessionConfig::new();
        if let Some(hints) = VectorSearchConfig::extract_sql_hints(query) {
            let vs_config = VectorSearchConfig::from_sql_hints(&hints)?;
            session_config.options_mut().extensions.insert(vs_config);
        }

        let mut ctx = SessionContext::new_with_config(session_config);
        // Same function surface as `sql()`: without it, `graph_neighbors(...)`,
        // the distance UDFs, and the vector operators are all unresolved.
        crate::core::sql::udf::register_all_custom_udfs(&mut ctx);
        crate::core::sql::graph_udf::register_graph_table_functions(&mut ctx);
        crate::core::sql::subscribe_table_function::register_subscribe_table_functions(&mut ctx);
        let _ = crate::core::sql::vector_operators::register_vector_operators(&mut ctx);

        for (alias, uri) in tables {
            let referenced = Table::new_async(uri).await?;
            let provider = Arc::new(BenoStreamTableProvider::new(Arc::new(referenced)));
            ctx.register_table(alias.as_str(), provider)?;
        }

        let df = ctx.sql(query).await?;
        Ok(df.collect().await?)
    }

    #[tracing::instrument(skip(self, filter_str, vector_filters, columns))]
    pub async fn read_async(
        &self,
        filter_str: Option<&str>,
        vector_filters: Option<Vec<VectorSearchParams>>,
        columns: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>> {
        self.read_with_config_async(
            filter_str,
            vector_filters,
            columns,
            self.query_config.clone(),
        )
        .await
    }

    pub async fn read_stream_async(
        &self,
        filter_str: Option<&str>,
        vector_filters: Option<Vec<VectorSearchParams>>,
        columns: Option<&[&str]>,
    ) -> Result<BoxStream<'static, Result<RecordBatch>>> {
        self.read_with_config_stream_async(
            filter_str,
            vector_filters,
            columns,
            self.query_config.clone(),
        )
        .await
    }

    pub async fn read_with_config_stream_async(
        &self,
        filter_str: Option<&str>,
        vector_filters: Option<Vec<VectorSearchParams>>,
        columns: Option<&[&str]>,
        config: QueryConfig,
    ) -> Result<BoxStream<'static, Result<RecordBatch>>> {
        let expr = match filter_str {
            Some(f) => {
                let schema = self.arrow_schema();
                Some(FilterExpr::parse_sql(f, schema).await?)
            }
            _ => None,
        };
        self.read_expr_stream_async(expr, vector_filters, columns, config, filter_str)
            .await
    }

    pub fn query(&self) -> TableQuery<'_> {
        TableQuery::new(self)
    }

    /// Generate a detailed execution plan with hit counts and pruning stats
    pub async fn explain(
        &self,
        filter_str: Option<&str>,
        vector_params: Option<Vec<VectorSearchParams>>,
    ) -> String {
        let manifest_manager =
            crate::core::manifest::ManifestManager::new(self.store.clone(), "", &self.uri);
        let (manifest, all_entries, version) = manifest_manager
            .load_latest_full()
            .await
            .unwrap_or((crate::core::manifest::Manifest::default(), Vec::new(), 0));
        let total_rows_table: usize = all_entries.iter().map(|e| e.record_count as usize).sum();
        let total_segments_table = all_entries.len();

        let mut plan = Vec::new();
        let divider = "-".repeat(60);
        plan.push(divider.clone());
        plan.push(format!("BENOSTREAM QUERY PLAN [Table: {}]", self.uri));

        // 1. Initial Pruning (Partition & Stats)
        let expr = if let Some(f) = filter_str {
            FilterExpr::parse_sql(f, self.arrow_schema()).await.ok()
        } else {
            None
        };

        let planner = QueryPlanner::new();
        let pruned_entries: Vec<(ManifestEntry, Option<IndexFile>)> = if version > 0 {
            planner.prune_entries(&all_entries, expr.as_ref(), vector_params.as_ref())
        } else {
            all_entries.iter().map(|e| (e.clone(), None)).collect()
        };

        let scanned_segments = pruned_entries.len();
        let scanned_rows: usize = pruned_entries
            .iter()
            .map(|(e, _)| e.record_count as usize)
            .sum();

        plan.push("Context Selection:".to_string());
        plan.push(format!(
            "  -> Total Table Scope: {} rows in {} segments",
            total_rows_table, total_segments_table
        ));
        if scanned_segments < total_segments_table {
            plan.push(format!(
                "  -> Pruning Activity: {} segments pruned via Partition/Stats mapping",
                total_segments_table - scanned_segments
            ));

            // Reason breakdown: which rule ruled each pruned segment out. An
            // entry is pruned as soon as any sub-filter rules it out, so we
            // report the first matching reason — the same precedence the scan
            // itself uses. Metrics are suppressed here: EXPLAIN must not
            // inflate the operational pruning counters.
            if let Some(ref e) = expr {
                let sub_filters = e.extract_and_conditions();
                let mut reasons: std::collections::HashMap<&'static str, usize> =
                    std::collections::HashMap::new();
                for entry in &all_entries {
                    if let Some(reason) = sub_filters
                        .iter()
                        .find_map(|f| planner.classify_condition(entry, f, false))
                    {
                        *reasons.entry(reason.label()).or_insert(0) += 1;
                    }
                }
                let mut pairs: Vec<(&'static str, usize)> = reasons.into_iter().collect();
                pairs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
                for (label, count) in pairs {
                    plan.push(format!("       - {} segment(s): {}", count, label));
                }
            }
        }
        plan.push(format!(
            "  -> Execution Scope: {} rows in {} segments",
            scanned_rows, scanned_segments
        ));
        plan.push("".to_string());

        // 2. Index Dry-run of Filters
        let mut scalar_hits = scanned_rows;
        let mut access_paths = Vec::new();

        if let Some(ref e) = expr {
            let sub_filters = e.extract_and_conditions();
            let mut total_hits = 0;

            // Map to track which index type was used for each sub-filter
            let mut filter_index_types: HashMap<String, HashSet<&'static str>> = HashMap::new();

            for (entry, _) in &pruned_entries {
                let Ok(mut reader) = self.segment_reader(entry) else {
                    total_hits += entry.record_count as usize;
                    continue;
                };
                if let Some(schema) = manifest
                    .schemas
                    .iter()
                    .find(|s| s.schema_id == manifest.current_schema_id)
                    .or_else(|| manifest.schemas.last())
                {
                    reader = reader.with_iceberg_schema(schema.clone());
                }

                let mut seg_bm: Option<roaring::RoaringBitmap> = None;
                for sub_f in &sub_filters {
                    // Report the access path from the manifest's `index_files` —
                    // the same source of truth the reader uses
                    // (`HybridReader::get_scalar_filter_bitmap`). Do NOT glob the
                    // filesystem: the writer's naming convention is the manifest's
                    // `file_path`, and a glob can silently miss it.
                    let path = entry
                        .index_files
                        .iter()
                        .find(|f| {
                            (f.is_lexical()
                                || f.index_category == "bitmap"
                                || f.index_category == "scalar")
                                && f.column_name == sub_f.column.as_str()
                        })
                        .map(|f| match f.index_category.as_str() {
                            "lexical" | "inverted" => "Inverted Index (Parquet)",
                            "bm25" => "BM25 Inverted Index",
                            "bitmap" | "scalar" => "Bitmap Index (.idx)",
                            _ => "Full Scan",
                        })
                        .unwrap_or("Full Scan");
                    filter_index_types
                        .entry(sub_f.column.clone())
                        .or_default()
                        .insert(path);
                }

                let rewritten_filters = reader.rewrite_composite_filters(sub_filters.clone());
                for sub_f in &rewritten_filters {
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
                let paths: Vec<&'static str> = filter_index_types
                    .get(&sub_f.column)
                    .map(|s: &HashSet<&'static str>| s.iter().cloned().collect())
                    .unwrap_or_else(|| vec!["Scan"]);
                access_paths.push(format!(
                    "  -> Filter (col: {}, op: {}, access: {})",
                    sub_f.column,
                    sub_f.op_to_string(),
                    paths.join(", ")
                ));
            }
        }

        // 3. Scalar Plan (First: Pre-filter the dataset)
        if !access_paths.is_empty() {
            plan.push("Scalar Execution:".to_string());
            for path in access_paths {
                plan.push(path);
            }
            let pct = if scanned_rows > 0 {
                (scalar_hits as f32 / scanned_rows as f32) * 100.0
            } else {
                0.0
            };
            plan.push(format!(
                "     [Selectivity: {} / {} rows ({:.2}%)]",
                scalar_hits, scanned_rows, pct
            ));
            plan.push("".to_string());
        }

        // 4. Vector Plan (Second: Vector search on pre-filtered rows)
        if let Some(ref vss) = vector_params {
            plan.push("Vector Execution:".to_string());
            for vs in vss {
                plan.push(format!(
                    "  -> VectorSearch (col: {}, k: {}, metric: {:?})",
                    vs.column, vs.k, vs.metric
                ));
            }

            // Report the access path from the manifest's `index_files` — the
            // same source of truth the search path uses (see
            // `reader/scan.rs::vector_search`). Do NOT glob the filesystem:
            // quantized indexes are written as
            // `{seg}.{col}.tq8.cluster_*.hnsw.graph`, so a `{seg}.{col}.cluster_*`
            // glob silently misses them and misreports a full scan.
            let mut access_mode = "Brute Force Scan (No Index)";
            for vs in vss {
                if let Some(f) = pruned_entries
                    .iter()
                    .flat_map(|(e, _)| e.index_files.iter())
                    .find(|f| f.index_category == "vector" && f.column_name == vs.column.as_str())
                {
                    // `algorithm` is the concrete quantization id ("tq8",
                    // "tq4", "pq", "hnsw"); `blob_type` is the Puffin blob type
                    // (e.g. "org.apache.iceberg.vector.hnsw-tq8"). Match on the
                    // algorithm so the report is independent of the storage
                    // container.
                    access_mode = match f.algorithm.as_str() {
                        "tq8" | "hnsw_tq8" => "HNSW-TQ8 Cluster Index",
                        "tq4" | "hnsw_tq4" => "HNSW-TQ4 Cluster Index",
                        "sq8" | "hnsw_sq8" => "HNSW-SQ8 Cluster Index",
                        "pq" | "hnsw_pq" => "HNSW-PQ Cluster Index",
                        _ => "HNSW-IVF Cluster Index",
                    };
                    break;
                }
            }
            plan.push(format!(
                "     [Access: {}] [Eligibility: {} rows]",
                access_mode, scalar_hits
            ));
            plan.push("".to_string());
        }

        plan.push("Final Retrieval:".to_string());
        plan.push(format!(
            "  -> ParallelRead (threads: {}, format: Parquet)",
            self.query_config.max_parallel_readers.unwrap_or(16)
        ));
        plan.push(divider);

        plan.join("\n")
    }

    pub fn filter(&self, expr: &str) -> TableQuery<'_> {
        TableQuery::new(self).filter(expr)
    }

    #[tracing::instrument(skip(self))]
    pub async fn read_with_config_async(
        &self,
        filter_str: Option<&str>,
        vector_filters: Option<Vec<VectorSearchParams>>,
        columns: Option<&[&str]>,
        config: QueryConfig,
    ) -> Result<Vec<RecordBatch>> {
        let expr = match filter_str {
            Some(f) => {
                let schema = self.arrow_schema();
                Some(FilterExpr::parse_sql(f, schema).await?)
            }
            _ => None,
        };
        self.read_expr_with_config_async(expr, vector_filters, columns, config, filter_str)
            .await
    }

    pub async fn read_expr_with_config_async(
        &self,
        expr: Option<FilterExpr>,
        vector_filters: Option<Vec<VectorSearchParams>>,
        columns: Option<&[&str]>,
        config: QueryConfig,
        filter_str: Option<&str>,
    ) -> Result<Vec<RecordBatch>> {
        let stream = self
            .read_expr_stream_async(expr, vector_filters, columns, config, filter_str)
            .await?;
        let results: Vec<Result<RecordBatch>> = stream.collect().await;
        results.into_iter().collect()
    }

    pub async fn read_expr_stream_async(
        &self,
        expr: Option<FilterExpr>,
        mut vector_filters: Option<Vec<VectorSearchParams>>,
        columns: Option<&[&str]>,
        config: QueryConfig,
        filter_str: Option<&str>,
    ) -> Result<BoxStream<'static, Result<RecordBatch>>> {
        use futures::StreamExt;

        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let need_mem_index = vector_filters.is_some();
        let mut force_direct = false;
        let (manifest, all_entries, version, staging_snap, active_snap) = {
            let mut resolved = None;
            for attempt in 0..8usize {
                let epoch1 = self.buffer_epoch.load(std::sync::atomic::Ordering::SeqCst);
                let (m, e, v) = if force_direct {
                    match manifest_manager.load_latest_direct().await {
                        Ok((m, v)) => {
                            let entries = manifest_manager.load_all_entries(&m).await?;
                            (m, entries, v)
                        }
                        Err(_) => {
                            if manifest_manager.exists().await.unwrap_or(false) {
                                (crate::core::manifest::Manifest::default(), Vec::new(), 0)
                            } else {
                                let segments =
                                    self.list_segments_from_store().await.unwrap_or_default();
                                (crate::core::manifest::Manifest::default(), segments, 0)
                            }
                        }
                    }
                } else {
                    match manifest_manager.load_latest_full().await {
                        Ok((m, e, v)) => (m, e, v),
                        Err(_) => {
                            if manifest_manager.exists().await.unwrap_or(false) {
                                (crate::core::manifest::Manifest::default(), Vec::new(), 0)
                            } else {
                                let segments =
                                    self.list_segments_from_store().await.unwrap_or_default();
                                (crate::core::manifest::Manifest::default(), segments, 0)
                            }
                        }
                    }
                };

                // Canonical lock order: pending_commit -> pending_writes -> indexing.memory_index.
                let (staging, active, committed_ver) = {
                    let pc_guard = self.pending_commit.lock();
                    let pw_guard = self.pending_writes.read();
                    let idx_guard = self.indexing.memory_index.read();

                    let committed_ver = pc_guard
                        .as_ref()
                        .filter(|pc| pc.phase != super::CommitPhase::Staging)
                        .and_then(|pc| pc.committed_manifest.as_ref().map(|cm| cm.version));

                    let staging = pc_guard.as_ref().and_then(|pc| {
                        if pc.phase == super::CommitPhase::Staging && !pc.batches.is_empty() {
                            Some((
                                pc.batches.clone(),
                                if need_mem_index {
                                    pc.frozen_memory_index.clone()
                                } else {
                                    None
                                },
                            ))
                        } else {
                            None
                        }
                    });

                    let active = if !pw_guard.is_empty() {
                        let batches: Vec<RecordBatch> =
                            pw_guard.iter().map(|p| p.batch.clone()).collect();
                        let idx = if need_mem_index {
                            idx_guard.clone()
                        } else {
                            None
                        };
                        (batches, idx)
                    } else {
                        (Vec::new(), None)
                    };

                    (staging, active, committed_ver)
                };

                let epoch2 = self.buffer_epoch.load(std::sync::atomic::Ordering::SeqCst);
                let stale_manifest = committed_ver.is_some_and(|cv| v < cv);
                if (epoch2 != epoch1 || stale_manifest) && attempt < 7 {
                    force_direct = true;
                    continue;
                }
                resolved = Some((m, e, v, staging, active));
                break;
            }
            resolved.ok_or_else(|| {
                anyhow::anyhow!("Failed to obtain linearizable snapshot during concurrent commit")
            })?
        };

        // Extract Iceberg schema from the already-loaded manifest for both
        // vector search requests and per-segment readers.
        let iceberg_schema = manifest
            .schemas
            .iter()
            .find(|s| s.schema_id == manifest.current_schema_id)
            .or_else(|| manifest.schemas.last())
            .cloned();

        // Table-level cluster index fast path.
        //
        // When the manifest carries a cluster-keyed artifact for the queried
        // column, one index call replaces the whole per-segment fan-out: the
        // artifact already holds every cluster whole, and its nodes carry
        // `(segment_id, row_id)`, so `fetch_results_by_id` hydrates the winners
        // unchanged. Restricted to a single vector filter with no scalar
        // predicate, no delete files on any entry, and no uncommitted/staging
        // buffer rows, because the artifact has no scalar-filter or delete
        // awareness and covers only persisted segments.
        let has_deletes = all_entries.iter().any(|e| !e.delete_files.is_empty());
        if !has_deletes && expr.is_none() && staging_snap.is_none() && active_snap.0.is_empty() {
            if let Some(ref vps) = vector_filters {
                if vps.len() == 1 {
                    let vp = &vps[0];
                    if let Ok(Some(index)) = self.ensure_cluster_index(&vp.column).await {
                        if let crate::core::index::VectorValue::Float32(q) = &vp.query {
                            let probes = vp
                                .probes
                                .unwrap_or_else(|| default_probes(index.centroids.len()));
                            let scored = index.search(q, vp.k, probes, vp.ef_search)?;
                            tracing::info!(
                                column = %vp.column,
                                probes = probes,
                                hits = scored.len(),
                                "cluster_index: served query from the table-level artifact"
                            );
                            let results = self.fetch_results_by_id(scored, columns).await?;
                            return Ok(futures::stream::iter(results.into_iter().map(Ok)).boxed());
                        }
                    }
                }
            }
        }

        // First prune by partition metadata & column statistics (`expr`) BEFORE
        // consulting `TableVectorRouter` so centroids in pruned partitions
        // never consume global probe budget or skew adaptive thresholds.
        let planner = QueryPlanner::new();
        let entries_after_filter =
            if version == 0 && manifest_manager.exists().await.unwrap_or(false) {
                Vec::new() // Strictly follow the empty manifest
            } else if expr.is_some() {
                planner
                    .prune_entries(&all_entries, expr.as_ref(), None)
                    .into_iter()
                    .map(|(e, _)| e)
                    .collect()
            } else {
                all_entries.clone()
            };

        // If vector filters are present, evaluate table-level vector routing
        // constrained to the surviving post-filter segments.
        if !entries_after_filter.is_empty() {
            if let Some(ref mut vps) = vector_filters {
                let allowed_seg_ids: HashSet<String> = entries_after_filter
                    .iter()
                    .map(|e| crate::core::segment::segment_id_from_path(&e.file_path).to_string())
                    .collect();

                for vp in vps.iter_mut() {
                    if vp.candidate_segments.is_none() {
                        if let Ok(Some(router)) = self.ensure_vector_router(&vp.column).await {
                            let probes = vp.probes.unwrap_or_else(|| default_probes(router.len()));
                            let plan = match &vp.query {
                                crate::core::index::VectorValue::Float32(q_f32) => Some(
                                    router.route_allowed(q_f32, probes, Some(&allowed_seg_ids)),
                                ),
                                _ => None,
                            };
                            if let Some(plan) = plan {
                                if !plan.candidate_segments.is_empty() {
                                    tracing::info!(
                                        column = %vp.column,
                                        probes = probes,
                                        candidates = plan.candidate_segments.len(),
                                        "vector_router: routed query across allowed partition centroids"
                                    );
                                    let routed: std::collections::HashSet<String> = router
                                        .segment_indices
                                        .iter()
                                        .map(|(seg, _)| seg.clone())
                                        .filter(|seg| allowed_seg_ids.contains(seg))
                                        .collect();
                                    vp.candidate_segments =
                                        Some(std::sync::Arc::new(plan.candidate_segments));
                                    vp.routed_segments = Some(std::sync::Arc::new(routed));
                                    vp.targeted_clusters =
                                        Some(std::sync::Arc::new(plan.segment_clusters));
                                }
                            }
                        }
                    }
                }
            }
        }

        let entries_to_read = if entries_after_filter.is_empty() {
            Vec::new()
        } else if vector_filters.is_some() {
            planner
                .prune_entries(&entries_after_filter, None, vector_filters.as_ref())
                .into_iter()
                .map(|(e, _)| e)
                .collect()
        } else {
            entries_after_filter
        };

        // --- SMART HYBRID TRIGGER ---
        // If we have both a vector filter AND a single text filter targeting a BM25/Inverted indexed column,
        // we switch to the Hybrid Coordinator path. If there are multiple filter columns or non-BM25 conditions,
        // we must stay on the standard SQL pushdown + vector search path.
        if let (Some(ref vs_params_list), Some(ref e)) = (&vector_filters, &expr) {
            let manifest = self.manifest().await?;
            let filtered_cols = e.get_referenced_columns();

            let current_schema = manifest
                .schemas
                .iter()
                .find(|s| s.schema_id == manifest.current_schema_id);

            let single_bm25_col = if filtered_cols.len() == 1 {
                filtered_cols.iter().next().cloned().and_then(|col| {
                    current_schema
                        .and_then(|s| {
                            s.fields.iter().find(|f| {
                                f.name == col
                                    && f.indexes
                                        .iter()
                                        .any(|idx| matches!(idx, IndexAlgorithm::Bm25 { .. }))
                            })
                        })
                        .map(|f| f.name.clone())
                })
            } else {
                None
            };

            if let Some(target_col) = single_bm25_col {
                tracing::info!(
                    "Smart Trigger: Executing Hybrid Search (RRF) for column '{}'",
                    target_col
                );
                let coordinator = HybridSearchCoordinator::new();

                // Extract search terms from FilterExpr for the BM25 engine
                let extracted_query = {
                    let conditions = e.extract_and_conditions();
                    let mut terms = Vec::new();
                    for f in conditions {
                        if f.column == target_col {
                            if let Some(v) = &f.min {
                                if let Some(s) = v.as_str() {
                                    terms.push(s.to_string());
                                }
                            }
                            if let Some(vals) = &f.values {
                                for v in vals {
                                    if let Some(s) = v.as_str() {
                                        terms.push(s.to_string());
                                    }
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

                let keyword_params = KeywordSearchParams::new(target_col, extracted_query);

                let first_vs_param = vs_params_list.first().cloned();
                let k = first_vs_param.as_ref().map(|p| p.k).unwrap_or(10);

                let scored_results = coordinator
                    .execute_hybrid(
                        self,
                        Some(e),
                        first_vs_param,
                        Some(keyword_params),
                        k,
                        config.rrf_k,
                    )
                    .await?;

                // Convert ScoredResults back to RecordBatches by fetching from Parquet
                // This is a simplified version of the final row-fetcher
                let results = self.fetch_results_by_id(scored_results, columns).await?;
                return Ok(futures::stream::iter(results.into_iter().map(Ok)).boxed());
            }
        }

        // Handle standard vector search
        if let Some(ref vs_params_list) = vector_filters {
            // 1. Search Disk
            let mut requests = Vec::new();
            for vs_params in vs_params_list {
                requests.push(
                    VectorSearchRequest::new(
                        vs_params.column.clone(),
                        vs_params.query.clone(),
                        vs_params.k,
                        vs_params.metric,
                        vs_params.use_mmap,
                    )
                    .with_filter(expr.clone())
                    .with_config(config.clone())
                    .with_ef_search(vs_params.ef_search)
                    .with_targeted_clusters(vs_params.targeted_clusters.clone())
                    .with_columns(columns.map(|c| c.iter().map(|s| s.to_string()).collect()))
                    .with_refine_factor(vs_params.refine_factor)
                    .with_probes(vs_params.probes)
                    .with_iceberg_schema(iceberg_schema.clone()),
                );
            }

            let mut results = crate::core::query::execute_multi_vector_search_with_config(
                entries_to_read.clone(),
                self.store.clone(),
                self.data_store.clone(),
                &self.uri,
                requests,
            )
            .await?;

            // 2. Search Memory (linearizable snapshot of Staging PendingCommit + Active pending_writes)
            let table_schema = self.arrow_schema();
            if let Some((ref staging_batches, Some(ref staging_idx))) = staging_snap {
                if let Some(dist_batch) = search_memory_buffer_batches(
                    staging_batches,
                    staging_idx,
                    vs_params_list,
                    expr.as_ref(),
                    columns,
                    &table_schema,
                )? {
                    results.push(("write_buffer_staging".to_string(), dist_batch));
                }
            }
            if let (ref active_batches, Some(ref active_idx)) = active_snap {
                if let Some(dist_batch) = search_memory_buffer_batches(
                    active_batches,
                    active_idx,
                    vs_params_list,
                    expr.as_ref(),
                    columns,
                    &table_schema,
                )? {
                    results.push(("write_buffer".to_string(), dist_batch));
                }
            }

            let mut expected_fields: Vec<std::sync::Arc<arrow::datatypes::Field>> =
                if let Some(cols) = columns {
                    cols.iter()
                        .filter_map(|name| {
                            table_schema
                                .column_with_name(name)
                                .map(|(_, f)| std::sync::Arc::new(f.clone()))
                        })
                        .collect()
                } else {
                    table_schema
                        .fields()
                        .iter()
                        .map(|f| std::sync::Arc::new(f.as_ref().clone()))
                        .collect()
                };
            if !expected_fields.iter().any(|f| f.name() == "distance") {
                expected_fields.push(std::sync::Arc::new(arrow::datatypes::Field::new(
                    "distance",
                    arrow::datatypes::DataType::Float32,
                    false,
                )));
            }
            let expected_schema =
                std::sync::Arc::new(arrow::datatypes::Schema::new(expected_fields));

            let conformed_results = results.into_iter().map(move |(_, batch)| {
                if batch.schema() == expected_schema {
                    Ok(batch)
                } else {
                    let mut new_columns = Vec::with_capacity(expected_schema.fields().len());
                    for field in expected_schema.fields() {
                        if let Ok(col) = batch.column_by_name(field.name()).ok_or(()) {
                            if col.data_type() != field.data_type() {
                                let casted = arrow::compute::cast(col, field.data_type())?;
                                new_columns.push(casted);
                            } else {
                                new_columns.push(col.clone());
                            }
                        } else {
                            let null_arr =
                                arrow::array::new_null_array(field.data_type(), batch.num_rows());
                            new_columns.push(null_arr);
                        }
                    }
                    arrow::record_batch::RecordBatch::try_new(expected_schema.clone(), new_columns)
                        .map_err(anyhow::Error::from)
                }
            });

            return Ok(futures::stream::iter(conformed_results).boxed());
        }

        let iceberg_schema_arc = iceberg_schema.map(Arc::new);

        // Capture current GPU context so it can be propagated into each async
        // worker closure (thread_local is per-thread, not per-future).
        let current_gpu_context = get_thread_gpu_context();

        let expr_arc = expr.map(Arc::new);
        let concurrency = config
            .max_parallel_readers
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4)
            })
            .min(64); // Cap to prevent resource exhaustion

        struct ReadCtx {
            table: Table,
            expr: Option<Arc<FilterExpr>>,
            schema: Option<Arc<crate::core::manifest::Schema>>,
            gpu: Option<crate::core::index::gpu::ComputeContext>,
            columns: Option<Vec<String>>,
            version: u64,
        }
        let read_ctx = Arc::new(ReadCtx {
            table: self.clone(),
            expr: expr_arc.clone(),
            schema: iceberg_schema_arc,
            gpu: current_gpu_context,
            columns: columns.map(|c| c.iter().map(|s| s.to_string()).collect()),
            version,
        });

        let stream = futures::stream::iter(entries_to_read)
            .map({
                let read_ctx = read_ctx.clone();
                move |entry| {
                    let ctx = read_ctx.clone();
                    async move {
                        if let Some(c) = ctx.gpu.clone() {
                            crate::core::index::gpu::set_thread_gpu_context(Some(c));
                        }
                        let cols_refs: Option<Vec<&str>> = ctx
                            .columns
                            .as_ref()
                            .map(|v| v.iter().map(|s| s.as_str()).collect());
                        ctx.table
                            .read_segment_expr(
                                &entry,
                                ctx.expr.as_deref(),
                                ctx.version,
                                cols_refs.as_deref(),
                                ctx.schema.as_deref(),
                            )
                            .await
                    }
                }
            })
            .buffer_unordered(concurrency);

        let results_stream = stream.flat_map(|res| match res {
            Ok(b_vec) => futures::stream::iter(b_vec.into_iter().map(Ok)).boxed(),
            Err(e) => futures::stream::once(async move { Err(e) }).boxed(),
        });

        // --- Read from In-Memory Write Buffer (Staging PendingCommit + Active pending_writes snapshot) ---
        let mut mem_batches = Vec::new();
        {
            let mut combined = Vec::new();
            if let Some((staging_batches, _)) = staging_snap {
                combined.extend(staging_batches);
            }
            combined.extend(active_snap.0);

            if !combined.is_empty() {
                let table_schema = self.arrow_schema();
                // Align batches to the full evolved schema first
                let mut aligned_buffer = Vec::with_capacity(combined.len());
                for b in &combined {
                    let aligned = if b.schema() != table_schema {
                        let mut cols = Vec::with_capacity(table_schema.fields().len());
                        for field in table_schema.fields() {
                            let col = if let Some(c) = b.column_by_name(field.name()) {
                                c.clone()
                            } else {
                                arrow::array::new_null_array(field.data_type(), b.num_rows())
                            };
                            cols.push(col);
                        }
                        RecordBatch::try_new(table_schema.clone(), cols)
                            .unwrap_or_else(|_| b.clone())
                    } else {
                        b.clone()
                    };
                    aligned_buffer.push(aligned);
                }

                if let Some(ref e) = expr_arc {
                    let planner = QueryPlanner::new();
                    for batch in aligned_buffer.iter() {
                        if let Ok(filtered) = planner.filter_expr(batch, e) {
                            if filtered.num_rows() > 0 {
                                let out_batch = if let Some(cols) = columns {
                                    let indices: Vec<usize> = cols
                                        .iter()
                                        .filter_map(|name| filtered.schema().index_of(name).ok())
                                        .collect();
                                    filtered.project(&indices).unwrap_or(filtered)
                                } else {
                                    filtered
                                };
                                mem_batches.push(Ok(out_batch));
                            }
                        }
                    }
                } else {
                    for batch in aligned_buffer.iter() {
                        if let Some(cols) = columns {
                            let indices: Vec<usize> = cols
                                .iter()
                                .filter_map(|name| batch.schema().index_of(name).ok())
                                .collect();
                            if let Ok(projected) = batch.project(&indices) {
                                mem_batches.push(Ok(projected));
                            }
                        } else {
                            mem_batches.push(Ok(batch.clone()));
                        }
                    }
                }
            }
        }

        let mem_stream = futures::stream::iter(mem_batches);

        Ok(results_stream.chain(mem_stream).boxed())
    }

    pub async fn read_filter_async(
        &self,
        filters: Vec<QueryFilter>,
        vector_filters: Option<Vec<VectorSearchParams>>,
        columns: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>> {
        self.read_filter_with_config_async(
            filters,
            vector_filters,
            columns,
            self.query_config.clone(),
        )
        .await
    }

    /// Read the rows matching a row-value `IN` list over the primary key
    /// columns, e.g. `(id, region) IN ((1, 'US'), (2, 'CA'))`.
    ///
    /// This is the read-side counterpart of the write-path PK enforcement. It
    /// pushes the whole tuple set down as a single DataFusion expression (so
    /// the predicate is evaluated inside the parquet scan instead of after it),
    /// and additionally consults the per-column inverted indexes to skip whole
    /// segments that contain no matching row — which is where the win comes
    /// from on a wide table.
    pub async fn read_pk_filter_async(
        &self,
        pk_filter: &PrimaryKeyFilter,
        columns: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>> {
        if pk_filter.is_empty() {
            return Ok(Vec::new());
        }

        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (manifest, all_entries, version) = manifest_manager.load_latest_full().await?;

        // Single-column PK: a plain IN list is already index-accelerated.
        if pk_filter.columns.len() == 1 {
            if let Some(qf) = pk_filter.to_query_filter() {
                return self
                    .read_filter_with_config_async(
                        vec![qf],
                        None,
                        columns,
                        self.query_config.clone(),
                    )
                    .await;
            }
        }

        let expr = FilterExpr::DataFusion(pk_filter.to_expr());
        let planner = QueryPlanner::new();
        let candidates = planner.prune_entries(&all_entries, Some(&expr), None);
        let cached_schema = manifest.schemas.last().cloned();

        let mut out = Vec::new();
        for (entry, _) in candidates {
            // Index-assisted segment skip: when the PK columns carry inverted
            // indexes and no row matches, don't read the segment at all.
            if let Ok(reader) = self.segment_reader(&entry) {
                if let Ok(Some(bm)) = Self::pk_match_bitmap(&reader, pk_filter).await {
                    let deleted = reader.load_merged_deletes().await.unwrap_or_default();
                    if (bm - deleted).is_empty() {
                        continue;
                    }
                }
            }

            let batches = self
                .read_segment_expr(
                    &entry,
                    Some(&expr),
                    version,
                    columns,
                    cached_schema.as_ref(),
                )
                .await?;
            out.extend(batches);
        }

        Ok(out)
    }

    pub async fn read_filter_with_config_async(
        &self,
        filters: Vec<QueryFilter>,
        vector_filters: Option<Vec<VectorSearchParams>>,
        columns: Option<&[&str]>,
        config: QueryConfig,
    ) -> Result<Vec<RecordBatch>> {
        let expr = FilterExpr::from_filters(filters);
        self.read_expr_with_config_async(expr, vector_filters, columns, config, None)
            .await
    }

    pub async fn read_segment_expr(
        &self,
        entry: &ManifestEntry,
        expr: Option<&FilterExpr>,
        manifest_version: u64,
        columns: Option<&[&str]>,
        cached_iceberg_schema: Option<&crate::core::manifest::Schema>,
    ) -> Result<Vec<RecordBatch>> {
        tracing::debug!(
            "read_segment_expr entry={} manifest_version={} columns={:?} expr={:?}",
            entry.file_path,
            manifest_version,
            columns,
            expr
        );
        let segment_id = crate::core::segment::segment_id_from_path(&entry.file_path);

        // Use cached schema if provided by caller; otherwise fall back to manifest load.
        let iceberg_schema = if let Some(schema) = cached_iceberg_schema {
            Some(schema.clone())
        } else {
            let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
            let (manifest, _, _) = manifest_manager
                .load_latest_full()
                .await
                .unwrap_or_default();
            manifest
                .schemas
                .iter()
                .find(|s| s.schema_id == manifest.current_schema_id)
                .or_else(|| manifest.schemas.last())
                .cloned()
        };

        let mut reader = self.segment_reader(entry)?;
        if let Some(s) = &iceberg_schema {
            reader = reader.with_iceberg_schema(s.clone());
        }

        let full_schema = if let Some(schema) = &iceberg_schema {
            Arc::new(schema.to_arrow())
        } else {
            self.arrow_schema()
        };

        let target_schema = if let Some(cols) = columns {
            let fields: Vec<arrow::datatypes::Field> = cols
                .iter()
                .filter_map(|name| full_schema.field_with_name(name).ok().cloned())
                .collect();
            Some(Arc::new(Schema::new(fields)))
        } else {
            Some(full_schema.clone())
        };

        let read_schema = if let (Some(cols), Some(e)) = (columns, expr) {
            let mut read_cols: std::collections::HashSet<String> =
                cols.iter().map(|s| s.to_string()).collect();
            for ref_col in e.get_referenced_columns() {
                read_cols.insert(ref_col.clone());
            }
            let fields: Vec<arrow::datatypes::Field> = read_cols
                .iter()
                .filter_map(|name| full_schema.field_with_name(name).ok().cloned())
                .collect();
            Some(Arc::new(Schema::new(fields)))
        } else {
            target_schema.clone()
        };

        if manifest_version == 0 || expr.is_none() {
            let t_setup = std::time::Instant::now();
            let mut stream = reader.stream_all(target_schema).await?;
            crate::telemetry::metrics::READ_PHASE_SECONDS
                .with_label_values(&["setup"])
                .observe(t_setup.elapsed().as_secs_f64());
            let t_decode = std::time::Instant::now();
            let mut batches = Vec::new();
            while let Some(batch_result) = stream.next().await {
                batches.push(batch_result?);
            }
            crate::telemetry::metrics::READ_PHASE_SECONDS
                .with_label_values(&["decode"])
                .observe(t_decode.elapsed().as_secs_f64());
            return Ok(batches);
        }

        // `expr.is_none()` returned above, so this is `Some`; the `None` arm is
        // defensive only.
        let expr = match expr {
            Some(e) => e,
            None => return Ok(Vec::new()),
        };
        // Coerce literals to the column types (e.g. an Int64 literal compared
        // against an Int32 column) so the comparison is valid in Arrow.
        let coerced_expr = expr.coerce_literals(&full_schema);
        let expr = &coerced_expr;
        let and_filters = reader.rewrite_composite_filters(expr.extract_and_conditions());

        // Try to use index for the FIRST filter that has one
        let mut batches = Vec::new();
        let mut index_used = false;

        for filter in &and_filters {
            if let Ok(indexed_batches) = reader.query_index_first(filter, read_schema.clone()).await
            {
                batches = indexed_batches;
                index_used = true;
                break;
            }
        }

        // JSON-path overlay: answer `json_*` predicates from the index when the
        // segment carries one. The result is a superset of the true matches;
        // the filter is re-applied below.
        if !index_used {
            let json_preds = expr.json_path_predicates();
            if !json_preds.is_empty() {
                if let Ok(Some(indexed_batches)) = reader
                    .query_json_path_first(&json_preds, read_schema.clone())
                    .await
                {
                    batches = indexed_batches;
                    index_used = true;
                }
            }
        }

        if !index_used {
            // Point Selection Optimization: Check Bloom Filters before scanning
            for filter in &and_filters {
                if let Some(vals) = &filter.values {
                    if vals.len() == 1
                        && !reader
                            .check_bloom_filter(&filter.column, &vals[0])
                            .await
                            .unwrap_or(true)
                    {
                        tracing::debug!(
                            "Bloom Filter Pruned segment: {} for col: {}",
                            segment_id,
                            filter.column
                        );
                        return Ok(vec![]);
                    }
                } else if let (Some(min), Some(max)) = (&filter.min, &filter.max) {
                    if min == max
                        && !reader
                            .check_bloom_filter(&filter.column, min)
                            .await
                            .unwrap_or(true)
                    {
                        tracing::debug!(
                            "Bloom Filter Pruned segment: {} for col: {}",
                            segment_id,
                            filter.column
                        );
                        return Ok(vec![]);
                    }
                }
            }

            let t_setup = std::time::Instant::now();
            let mut stream = reader.stream_all(read_schema).await?;
            crate::telemetry::metrics::READ_PHASE_SECONDS
                .with_label_values(&["setup"])
                .observe(t_setup.elapsed().as_secs_f64());
            let t_decode = std::time::Instant::now();
            while let Some(batch_result) = stream.next().await {
                batches.push(batch_result?);
            }
            crate::telemetry::metrics::READ_PHASE_SECONDS
                .with_label_values(&["decode"])
                .observe(t_decode.elapsed().as_secs_f64());
        }

        let planner = QueryPlanner::new();
        let t_filter = std::time::Instant::now();
        let mut filtered_batches = Vec::new();
        for batch in batches {
            match planner.filter_expr(&batch, expr) {
                Ok(filtered) => {
                    if filtered.num_rows() > 0 {
                        let projected = if let Some(ts) = &target_schema {
                            let indices: Vec<usize> = ts
                                .fields()
                                .iter()
                                .filter_map(|f| filtered.schema().index_of(f.name()).ok())
                                .collect();
                            filtered.project(&indices).unwrap_or(filtered.clone())
                        } else {
                            filtered
                        };
                        filtered_batches.push(projected);
                    }
                }
                Err(e) => {
                    // The pushed-down filter could not be evaluated — e.g. an
                    // Int32 column compared against an Int64 literal
                    // (`Invalid comparison operation: Int32 >= Int64`). The
                    // filter is `Inexact`, so DataFusion re-applies it above the
                    // scan; keep the batch unfiltered rather than dropping rows.
                    tracing::warn!(
                        "Failed to evaluate pushed-down filter on batch: {e}; keeping batch unfiltered"
                    );
                    let projected = if let Some(ts) = &target_schema {
                        let indices: Vec<usize> = ts
                            .fields()
                            .iter()
                            .filter_map(|f| batch.schema().index_of(f.name()).ok())
                            .collect();
                        batch.project(&indices).unwrap_or_else(|_| batch.clone())
                    } else {
                        batch.clone()
                    };
                    filtered_batches.push(projected);
                }
            }
        }
        crate::telemetry::metrics::READ_PHASE_SECONDS
            .with_label_values(&["filter"])
            .observe(t_filter.elapsed().as_secs_f64());
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
        self.read_segment_expr(entry, expr.as_ref(), manifest_version, columns, None)
            .await
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
        self.read_segment_multi(entry, &filters, manifest_version, columns)
            .await
    }

    pub async fn stream_segment_multi(
        &self,
        entry: &ManifestEntry,
        filters: &[QueryFilter],
        manifest_version: u64,
        columns: Option<&[&str]>,
    ) -> Result<futures::stream::BoxStream<'static, Result<RecordBatch>>> {
        let batches = self
            .read_segment_multi(entry, filters, manifest_version, columns)
            .await?;
        Ok(futures::stream::iter(batches.into_iter().map(Ok)).boxed())
    }

    pub async fn stream_segment(
        &self,
        entry: &ManifestEntry,
        query_filter_opt: Option<&QueryFilter>,
        manifest_version: u64,
        columns: Option<&[&str]>,
    ) -> Result<futures::stream::BoxStream<'static, Result<RecordBatch>>> {
        let filters = match query_filter_opt {
            Some(f) => vec![f.clone()],
            None => vec![],
        };
        self.stream_segment_multi(entry, &filters, manifest_version, columns)
            .await
    }

    pub async fn stream_all(
        &self,
        columns: Option<&[&str]>,
    ) -> Result<futures::stream::BoxStream<'static, Result<RecordBatch>>> {
        use futures::StreamExt;
        let manifest = self.manifest().await.unwrap_or_default();
        let all_entries = self.get_snapshot_segments().await?;

        let mut streams = Vec::new();
        for entry in all_entries {
            let stream = self
                .stream_segment(&entry, None, manifest.version, columns)
                .await?;
            streams.push(stream);
        }

        // Chain all streams
        let combined_stream = futures::stream::iter(streams).flatten();
        Ok(combined_stream.boxed())
    }

    async fn list_segments_from_store(&self) -> Result<Vec<ManifestEntry>> {
        use futures::StreamExt;
        let mut entries = Vec::new();
        let mut stream = self.store.list(None);
        while let Some(res) = stream.next().await {
            if let Ok(meta) = res {
                let path = meta.location.to_string();
                if crate::core::segment::is_data_parquet_path(&path) {
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

    #[tracing::instrument(skip(self))]
    pub async fn execute_vector_search_as_scored(
        &self,
        params: VectorSearchParams,
    ) -> Result<Vec<ScoredResult>> {
        self.execute_vector_search_as_scored_filtered(params, None)
            .await
    }

    /// Vector retrieval with an optional scalar pre-filter.
    ///
    /// The predicate is pushed into the reader, which resolves it to a row-id
    /// bitmap and intersects it with the KNN candidates.
    pub async fn execute_vector_search_as_scored_filtered(
        &self,
        mut params: VectorSearchParams,
        filter: Option<FilterExpr>,
    ) -> Result<Vec<ScoredResult>> {
        let start_time = std::time::Instant::now();

        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (manifest, all_entries, _) = manifest_manager.load_latest_full().await?;
        let iceberg_schema = manifest
            .schemas
            .iter()
            .find(|s| s.schema_id == manifest.current_schema_id)
            .or_else(|| manifest.schemas.last())
            .cloned();

        // Table-level cluster index fast path (see `read_expr_stream_async`).
        // Restricted to unfiltered queries with no delete files.
        let has_deletes = all_entries.iter().any(|e| !e.delete_files.is_empty());
        if filter.is_none() && !has_deletes {
            if let Ok(Some(index)) = self.ensure_cluster_index(&params.column).await {
                if let crate::core::index::VectorValue::Float32(q) = &params.query {
                    let probes = params
                        .probes
                        .unwrap_or_else(|| default_probes(index.centroids.len()));
                    let scored = index.search(q, params.k, probes, params.ef_search)?;
                    crate::telemetry::metrics::SEARCH_LATENCY_SECONDS
                        .observe(start_time.elapsed().as_secs_f64());
                    return Ok(scored);
                }
            }
        }

        let planner = QueryPlanner::new();
        let entries_after_filter: Vec<ManifestEntry> = if filter.is_some() {
            planner
                .prune_entries(&all_entries, filter.as_ref(), None)
                .into_iter()
                .map(|(e, _)| e)
                .collect()
        } else {
            all_entries
        };

        if entries_after_filter.is_empty() {
            return Ok(Vec::new());
        }

        let allowed_seg_ids: HashSet<String> = entries_after_filter
            .iter()
            .map(|e| crate::core::segment::segment_id_from_path(&e.file_path).to_string())
            .collect();

        if params.candidate_segments.is_none() {
            if let Ok(Some(router)) = self.ensure_vector_router(&params.column).await {
                let probes = params
                    .probes
                    .unwrap_or_else(|| default_probes(router.len()));
                if let crate::core::index::VectorValue::Float32(q_f32) = &params.query {
                    let plan = router.route_allowed(q_f32, probes, Some(&allowed_seg_ids));
                    if !plan.candidate_segments.is_empty() {
                        let routed: HashSet<String> = router
                            .segment_indices
                            .iter()
                            .map(|(seg, _)| seg.clone())
                            .filter(|seg| allowed_seg_ids.contains(seg))
                            .collect();
                        params.candidate_segments = Some(Arc::new(plan.candidate_segments));
                        params.routed_segments = Some(Arc::new(routed));
                        params.targeted_clusters = Some(Arc::new(plan.segment_clusters));
                    }
                }
            }
        }

        let vector_list = vec![params];
        let candidates: Vec<ManifestEntry> = planner
            .prune_entries(&entries_after_filter, None, Some(&vector_list))
            .into_iter()
            .map(|(e, _)| e)
            .collect();

        let p = &vector_list[0];
        let request =
            VectorSearchRequest::new(p.column.clone(), p.query.clone(), p.k, p.metric, p.use_mmap)
                .with_filter(filter)
                .with_ef_search(p.ef_search)
                .with_probes(p.probes)
                .with_refine_factor(p.refine_factor)
                .with_targeted_clusters(p.targeted_clusters.clone())
                .with_iceberg_schema(iceberg_schema)
                .with_config(self.query_config.clone());

        let scored_results = crate::core::query::execute_vector_search_raw_with_config(
            candidates,
            self.store.clone(),
            self.data_store.clone(),
            &self.uri,
            request,
        )
        .await?;

        crate::telemetry::metrics::SEARCH_LATENCY_SECONDS
            .observe(start_time.elapsed().as_secs_f64());
        Ok(scored_results)
    }

    #[tracing::instrument(skip(self))]
    pub async fn execute_keyword_search_as_scored(
        &self,
        params: KeywordSearchParams,
    ) -> Result<Vec<ScoredResult>> {
        self.execute_keyword_search_as_scored_filtered(params, None)
            .await
    }

    /// BM25 retrieval with an optional scalar pre-filter.
    ///
    /// The reader turns the predicate into a row-id bitmap and drops excluded
    /// documents before the per-segment top-k is truncated.
    pub async fn execute_keyword_search_as_scored_filtered(
        &self,
        params: KeywordSearchParams,
        filter: Option<FilterExpr>,
    ) -> Result<Vec<ScoredResult>> {
        let start_time = std::time::Instant::now();
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (manifest, all_entries, _) = manifest_manager.load_latest_full().await?;
        let iceberg_schema = manifest
            .schemas
            .iter()
            .find(|s| s.schema_id == manifest.current_schema_id)
            .or_else(|| manifest.schemas.last())
            .cloned();

        let candidates: Vec<ManifestEntry> = if filter.is_some() {
            QueryPlanner::new()
                .prune_entries(&all_entries, filter.as_ref(), None)
                .into_iter()
                .map(|(e, _)| e)
                .collect()
        } else {
            all_entries
        };

        let mut all_scored = Vec::new();
        for entry in candidates {
            let segment_id =
                crate::core::segment::segment_id_from_path(&entry.file_path).to_string();
            let mut reader = self.segment_reader(&entry)?;
            if let Some(ref s) = iceberg_schema {
                reader = reader.with_iceberg_schema(s.clone());
            }
            let matches = reader
                .keyword_search_index(
                    &params.column,
                    &params.query,
                    1000,
                    &params.params(),
                    params.analyzer.as_deref(),
                    filter.as_ref(),
                )
                .await?;

            for (row_id, score) in matches {
                all_scored.push(ScoredResult {
                    segment_id: segment_id.clone(),
                    row_id: row_id as u32,
                    score,
                });
            }
        }
        // Global best-first order: per-segment results were concatenated above,
        // and RRF rank fusion requires each input list sorted by score desc.
        all_scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        crate::telemetry::metrics::SEARCH_LATENCY_SECONDS
            .observe(start_time.elapsed().as_secs_f64());
        Ok(all_scored)
    }

    #[tracing::instrument(skip(self))]
    pub async fn execute_phrase_search_as_scored(
        &self,
        column: &str,
        phrase: &str,
        slop: usize,
        analyzer: Option<&str>,
    ) -> Result<Vec<ScoredResult>> {
        self.execute_phrase_search_as_scored_filtered(column, phrase, slop, analyzer, None)
            .await
    }

    /// Position-aware phrase search with an optional scalar pre-filter and
    /// partition pruning.
    #[tracing::instrument(skip(self))]
    pub async fn execute_phrase_search_as_scored_filtered(
        &self,
        column: &str,
        phrase: &str,
        slop: usize,
        analyzer: Option<&str>,
        filter: Option<FilterExpr>,
    ) -> Result<Vec<ScoredResult>> {
        let start_time = std::time::Instant::now();
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (manifest, all_entries, _) = manifest_manager.load_latest_full().await?;
        let iceberg_schema = manifest
            .schemas
            .iter()
            .find(|s| s.schema_id == manifest.current_schema_id)
            .or_else(|| manifest.schemas.last())
            .cloned();

        let candidates: Vec<ManifestEntry> = if filter.is_some() {
            QueryPlanner::new()
                .prune_entries(&all_entries, filter.as_ref(), None)
                .into_iter()
                .map(|(e, _)| e)
                .collect()
        } else {
            all_entries
        };

        let mut all_scored = Vec::new();
        for entry in candidates {
            let segment_id =
                crate::core::segment::segment_id_from_path(&entry.file_path).to_string();
            let mut reader = self.segment_reader(&entry)?;
            if let Some(ref s) = iceberg_schema {
                reader = reader.with_iceberg_schema(s.clone());
            }
            let matches = reader
                .phrase_search_index_filtered(column, phrase, 1000, slop, analyzer, filter.as_ref())
                .await?;

            for (row_id, score) in matches {
                all_scored.push(ScoredResult {
                    segment_id: segment_id.clone(),
                    row_id: row_id as u32,
                    score,
                });
            }
        }
        all_scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        crate::telemetry::metrics::SEARCH_LATENCY_SECONDS
            .observe(start_time.elapsed().as_secs_f64());
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

        // Score-only short-circuit (vector-search I/O optimisation).
        //
        // The score is already in hand from the HNSW/BM25 index search, and the
        // engine synthesises the distance column *after* reading Parquet. So a
        // query that projects nothing but the score was reading every column of
        // every matched row for a value it already had. Emit it directly.
        if let Some(cols) = columns.filter(|c| !c.is_empty()) {
            let score_like = |c: &str| c == "distance" || c == "_distance" || c == "score";
            if cols.iter().all(|c| score_like(c)) {
                let scores: Vec<f32> = results.iter().map(|r| r.score).collect();
                let fields: Vec<arrow::datatypes::Field> = cols
                    .iter()
                    .map(|c| {
                        arrow::datatypes::Field::new(*c, arrow::datatypes::DataType::Float32, false)
                    })
                    .collect();
                let schema = Arc::new(arrow::datatypes::Schema::new(fields));
                let arrays: Vec<Arc<dyn arrow::array::Array>> = cols
                    .iter()
                    .map(|_| {
                        Arc::new(arrow::array::Float32Array::from(scores.clone()))
                            as Arc<dyn arrow::array::Array>
                    })
                    .collect();
                return Ok(vec![RecordBatch::try_new(schema, arrays)?]);
            }
        }

        // Group by segment to minimize I/O
        let mut by_segment: HashMap<String, Vec<(u32, f32)>> = HashMap::new();
        for r in results {
            by_segment
                .entry(r.segment_id)
                .or_default()
                .push((r.row_id, r.score));
        }

        let mut final_batches = Vec::new();
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (manifest, all_entries, _) = manifest_manager.load_latest_full().await?;
        let iceberg_schema = manifest
            .schemas
            .iter()
            .find(|s| s.schema_id == manifest.current_schema_id)
            .or_else(|| manifest.schemas.last())
            .cloned();

        for (seg_id, rows) in by_segment {
            if seg_id == "vector_path" {
                continue;
            }

            // Find segment in all entries by exact segment id or path equality
            let entry = all_entries
                .iter()
                .find(|e| {
                    crate::core::segment::segment_id_from_path(&e.file_path) == seg_id
                        || e.file_path == seg_id
                })
                .ok_or_else(|| anyhow::anyhow!("Segment {} not found in manifest", seg_id))?;

            let mut reader = self.segment_reader(entry)?;
            if let Some(ref s) = iceberg_schema {
                reader = reader.with_iceberg_schema(s.clone());
            }

            // Convert fused row IDs and scores to RecordBatch
            let batch = reader.read_rows_by_id(rows, columns).await?;
            final_batches.push(batch);
        }

        Ok(final_batches)
    }

    /// Run dense-vector and BM25 retrieval, fuse them with RRF, and return the
    /// hydrated rows with the fused score in an `rrf_score` column.
    ///
    /// This is the shared core path behind the Python `hybrid_search` helper and
    /// the SQL/connector surfaces. Ranking is delegated to
    /// [`HybridSearchCoordinator`] so every surface uses the same RRF formula.
    pub async fn hybrid_search_batches(
        &self,
        text_column: &str,
        query_text: &str,
        vector_column: &str,
        query_vector: Vec<f32>,
        k: usize,
        rrf_k: f32,
    ) -> Result<Vec<RecordBatch>> {
        self.hybrid_search_batches_filtered(
            text_column,
            query_text,
            vector_column,
            query_vector,
            k,
            rrf_k,
            None,
        )
        .await
    }

    /// Hybrid retrieval with an optional scalar predicate.
    ///
    /// The predicate is parsed once and handed to both retrieval paths, which
    /// push it into candidate generation (row-id bitmap intersect for the vector
    /// path, bitmap retain before the BM25 heap truncation). This is what makes
    /// `filter` work on the Trino/Spark `hybrid_search` surfaces instead of
    /// being rejected or silently ignored.
    #[allow(clippy::too_many_arguments)]
    pub async fn hybrid_search_batches_filtered(
        &self,
        text_column: &str,
        query_text: &str,
        vector_column: &str,
        query_vector: Vec<f32>,
        k: usize,
        rrf_k: f32,
        filter: Option<&str>,
    ) -> Result<Vec<RecordBatch>> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let filter_expr = match filter.map(str::trim).filter(|sql| !sql.is_empty()) {
            Some(sql) => Some(FilterExpr::parse_sql(sql, self.arrow_schema()).await?),
            None => None,
        };
        let per_path = k.saturating_mul(2);
        let vector_params =
            VectorSearchParams::new(vector_column, VectorValue::Float32(query_vector), per_path);
        let keyword_params = KeywordSearchParams::new(text_column, query_text);
        let coordinator = HybridSearchCoordinator::new();
        let fused = coordinator
            .execute_hybrid(
                self,
                filter_expr.as_ref(),
                Some(vector_params),
                Some(keyword_params),
                k,
                Some(rrf_k),
            )
            .await?;
        if fused.is_empty() {
            return Ok(Vec::new());
        }

        let mut batches = self.fetch_results_by_id(fused, None).await?;
        // `fetch_results_by_id` appends the fused score as `distance`; rename it
        // so SQL callers see the same name as the Python API.
        for batch in &mut batches {
            if let Ok(index) = batch.schema().index_of("distance") {
                let mut fields = batch.schema().fields().to_vec();
                fields[index] = Arc::new(arrow::datatypes::Field::new(
                    "rrf_score",
                    arrow::datatypes::DataType::Float32,
                    false,
                ));
                let schema = Arc::new(Schema::new(fields));
                *batch = RecordBatch::try_new(schema, batch.columns().to_vec())?;
            }
        }
        // Residual predicate.
        //
        // Candidate generation already intersects an indexed column's row-id
        // bitmap, but a column without a scalar/inverted index yields no bitmap
        // at all — the predicate would then be a silent no-op. Evaluate it
        // against the hydrated rows so `filter` is always honoured.
        if let Some(expr) = filter_expr.as_ref() {
            let planner = QueryPlanner::new();
            let mut filtered = Vec::with_capacity(batches.len());
            for batch in batches {
                let kept = planner.filter_expr(&batch, expr)?;
                if kept.num_rows() > 0 {
                    filtered.push(kept);
                }
            }
            batches = filtered;
        }
        Ok(batches)
    }

    /// Verifies the data integrity of all segments in the table by comparing
    /// their actual file checksums against the ones stored in the manifest.
    pub async fn verify_integrity_async(&self) -> Result<()> {
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (_manifest, all_entries, _version) = manifest_manager.load_latest_full().await?;

        for entry in all_entries {
            if let Some(expected_checksum) = &entry.file_checksum {
                let file_path = &entry.file_path;
                let is_remote = file_path.contains("://") && !file_path.starts_with("file://");

                let mut hasher = sha2::Sha256::new();
                use futures::StreamExt;
                use sha2::Digest;

                if is_remote {
                    let path_str = if file_path.starts_with("s3://")
                        || file_path.starts_with("gcs://")
                        || file_path.starts_with("azure://")
                    {
                        let parts: Vec<&str> = file_path.split("://").collect();
                        if parts.len() > 1 {
                            let without_scheme = parts[1];
                            let path_parts: Vec<&str> = without_scheme.splitn(2, '/').collect();
                            if path_parts.len() > 1 {
                                path_parts[1].to_string()
                            } else {
                                without_scheme.to_string()
                            }
                        } else {
                            file_path.clone()
                        }
                    } else {
                        // Relative remote path
                        let uri_parts: Vec<&str> = self.uri.split("://").collect();
                        if uri_parts.len() > 1 {
                            let without_scheme = uri_parts[1];
                            let path_parts: Vec<&str> = without_scheme.splitn(2, '/').collect();
                            if path_parts.len() > 1 {
                                format!("{}/{}", path_parts[1], file_path)
                            } else {
                                file_path.to_string()
                            }
                        } else {
                            file_path.clone()
                        }
                    };

                    let path = object_store::path::Path::parse(&path_str)?;
                    let mut stream = self.store.get(&path).await?.into_stream();
                    while let Some(chunk_result) = stream.next().await {
                        let chunk = chunk_result?;
                        hasher.update(&chunk);
                    }
                } else {
                    let mut local_path = std::path::PathBuf::from(
                        self.uri.strip_prefix("file://").unwrap_or(&self.uri),
                    );
                    if !file_path.starts_with('/') && !file_path.starts_with("file://") {
                        local_path.push(file_path);
                    } else {
                        local_path = std::path::PathBuf::from(
                            file_path.strip_prefix("file://").unwrap_or(file_path),
                        );
                    }

                    tracing::debug!(
                        "verify_integrity_async: self.uri={}, file_path={}, local_path={:?}",
                        self.uri,
                        file_path,
                        local_path
                    );

                    let mut file = std::fs::File::open(&local_path).map_err(|e| {
                        anyhow::anyhow!(
                            "Failed to open local data file {:?} for integrity check: {}",
                            local_path,
                            e
                        )
                    })?;

                    let mut buffer = [0; 65536];
                    use std::io::Read;
                    while let Ok(n) = file.read(&mut buffer) {
                        if n == 0 {
                            break;
                        }
                        hasher.update(&buffer[..n]);
                    }
                }

                let actual_checksum = format!("{:x}", hasher.finalize());
                if &actual_checksum != expected_checksum {
                    anyhow::bail!("Data integrity validation failed for segment {}: expected checksum {}, but got {}", file_path, expected_checksum, actual_checksum);
                }
            }
        }
        Ok(())
    }
}
