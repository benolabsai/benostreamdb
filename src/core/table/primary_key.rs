// Copyright (c) 2026 Richard Albright. All rights reserved.

use crate::core::manifest::{ManifestEntry, ManifestManager, Schema};
use crate::core::planner::{FilterExpr, QueryFilter, QueryPlanner};
use crate::core::reader::HybridReader;
use crate::SegmentConfig;
/// Primary key management: setting, syncing, adding, dropping, and validation.
///
/// Contains methods on `Table` for:
/// - `set_primary_key`, `set_primary_key_async`
/// - `sync_primary_key_from_schema`, `sync_primary_key_from_schema_async`
/// - `get_primary_key`
/// - `add_primary_key`, `drop_primary_key`
/// - `_validate_pk_uniqueness`, `_check_pk_in_storage_async`
/// - `check_primary_key_uniqueness_async`
use anyhow::Result;
use arrow::array::Array;
use arrow::record_batch::RecordBatch;
use datafusion::logical_expr::{Expr, Operator};
use roaring::RoaringBitmap;
use serde_json::Value;

use super::Table;

impl Table {
    // -----------------------------------------------------------------------
    // Primary key setters / getters
    // -----------------------------------------------------------------------

    pub fn set_primary_key(&self, columns: Vec<String>) {
        if let Some(ref rt) = self.rt {
            let _ = rt.block_on(self.set_primary_key_async(columns));
        } else {
            let mut pk = self.primary_key.write();
            *pk = columns;
        }
    }

    /// Asynchronously set the primary key. This commits a new manifest version
    /// and ensures the columns are marked as NOT NULL (required: true).
    pub async fn set_primary_key_async(&self, columns: Vec<String>) -> Result<()> {
        let manifest = self.manifest().await?;
        let latest_schema = manifest
            .schemas
            .last()
            .ok_or_else(|| anyhow::anyhow!("No schema found"))?;

        let mut field_ids = Vec::new();
        for col in &columns {
            let id = latest_schema
                .fields
                .iter()
                .find(|f| f.name == *col)
                .map(|f| f.id)
                .ok_or_else(|| anyhow::anyhow!("Column '{}' not found in schema", col))?;
            field_ids.push(id);
        }

        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        manifest_manager.update_identifier_fields(field_ids).await?;

        // Update in-memory state
        {
            let mut pk = self.primary_key.write();
            *pk = columns;
        }

        // Update in-memory schema ref (it's slightly stale now, but will refresh on next use)
        // or we could force a reload.
        let (new_manifest, _) = manifest_manager.load_latest().await?;
        if let Some(s) = new_manifest.schemas.last() {
            let mut schema_lock = self.schema.write();
            *schema_lock = std::sync::Arc::new(s.to_arrow());
        }

        Ok(())
    }

    pub async fn sync_primary_key_from_schema_async(&self) -> Result<()> {
        let manifest = self.manifest().await?;
        if let Some(latest_schema) = manifest.schemas.last() {
            let mut pk_names = Vec::new();
            for id in &latest_schema.identifier_field_ids {
                if let Some(field) = latest_schema.fields.iter().find(|f| f.id == *id) {
                    pk_names.push(field.name.clone());
                }
            }
            let mut pk = self.primary_key.write();
            *pk = pk_names;
        }
        Ok(())
    }

    pub fn get_primary_key(&self) -> Vec<String> {
        self.primary_key.read().clone()
    }

    /// Synchronize PK columns (Public Sync)
    pub fn sync_primary_key_from_schema(&self) -> Result<()> {
        self.runtime()
            .block_on(self.sync_primary_key_from_schema_async())
    }

    // -----------------------------------------------------------------------
    // Add / drop primary key columns
    // -----------------------------------------------------------------------

    /// Add a column to the primary key.
    /// This is an atomic operation that commits a new manifest version.
    /// Validation: Ensures no duplicate keys exist for the new definition.
    pub async fn add_primary_key(&self, column: String) -> Result<()> {
        let manifest = self.manifest().await?;
        let latest_schema = manifest
            .schemas
            .last()
            .ok_or_else(|| anyhow::anyhow!("No schema found"))?;

        // Find field ID for column
        let field_id = latest_schema
            .fields
            .iter()
            .find(|f| f.name == column)
            .map(|f| f.id)
            .ok_or_else(|| anyhow::anyhow!("Column '{}' not found in schema", column))?;

        let mut next_ids = latest_schema.identifier_field_ids.clone();
        if next_ids.contains(&field_id) {
            return Ok(()); // Already in PK
        }
        next_ids.push(field_id);

        // Validate uniqueness before committing
        self._validate_pk_uniqueness(&next_ids, latest_schema)
            .await?;

        // Atomic commit to manifest
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        manifest_manager.update_identifier_fields(next_ids).await?;

        // Update in-memory state
        let mut pk = self.primary_key.write();
        if !pk.contains(&column) {
            pk.push(column);
        }
        Ok(())
    }

    /// Remove a column from the primary key.
    /// This is an atomic operation that commits a new manifest version.
    pub async fn drop_primary_key(&self, column: String) -> Result<()> {
        let manifest = self.manifest().await?;
        let latest_schema = manifest
            .schemas
            .last()
            .ok_or_else(|| anyhow::anyhow!("No schema found"))?;

        let field_id = latest_schema
            .fields
            .iter()
            .find(|f| f.name == column)
            .map(|f| f.id)
            .ok_or_else(|| anyhow::anyhow!("Column '{}' not found in schema", column))?;

        let mut next_ids = latest_schema.identifier_field_ids.clone();
        next_ids.retain(|id| id != &field_id);

        // Atomic commit to manifest
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        manifest_manager.update_identifier_fields(next_ids).await?;

        // Update in-memory state
        let mut pk = self.primary_key.write();
        pk.retain(|c| c != &column);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Internal validation helpers
    // -----------------------------------------------------------------------

    /// Internal helper to validate that a set of field IDs form a unique key across existing data.
    async fn _validate_pk_uniqueness(&self, field_ids: &[i32], schema: &Schema) -> Result<()> {
        let col_names: Vec<String> = field_ids
            .iter()
            .map(|id| {
                schema
                    .fields
                    .iter()
                    .find(|f| f.id == *id)
                    .map(|f| f.name.clone())
                    .ok_or_else(|| {
                        anyhow::anyhow!("Field id {} in primary key not found in schema", id)
                    })
            })
            .collect::<Result<Vec<String>>>()?;

        // NOTE: use the async read path. `read_with_columns` is a sync wrapper
        // that calls `runtime().block_on(...)`; invoking it from this async fn
        // (itself driven by `TOKIO_RUNTIME.block_on`) re-enters the runtime and
        // panics with "Cannot start a runtime from within a runtime".
        let col_refs: Vec<&str> = col_names.iter().map(|s| s.as_str()).collect();

        // 1. Acceleration: Single-column PK check using indexes
        if col_names.len() == 1 {
            let col_name = &col_names[0];

            // For now, we perform an optimized read of just the PK column.
            let batches = self
                .read_async(None, None, Some(&col_refs))
                .await
                .map_err(|e| anyhow::anyhow!("Validation read failed: {}", e))?;

            let mut seen = std::collections::HashSet::new();
            for batch in batches {
                let col = batch.column_by_name(col_name).ok_or_else(|| {
                    anyhow::anyhow!("Column '{}' not found in validation batch", col_name)
                })?;
                for i in 0..batch.num_rows() {
                    let val = crate::core::manifest::ManifestValue::from_array(col, i).to_string();
                    if !seen.insert(val) {
                        return Err(anyhow::anyhow!(
                            "Primary key violation detected for column {}: Duplicate value found.",
                            col_name
                        ));
                    }
                }
            }
            return Ok(());
        }

        // Fallback: Multi-column PK scan
        let batches = self
            .read_async(None, None, Some(&col_refs))
            .await
            .map_err(|e| anyhow::anyhow!("Validation read failed: {}", e))?;

        let mut seen = std::collections::HashSet::new();
        for batch in batches {
            let sort_fields = batch
                .schema()
                .fields()
                .iter()
                .map(|f| arrow::row::SortField::new(f.data_type().clone()))
                .collect::<Vec<_>>();
            let converter = arrow::row::RowConverter::new(sort_fields)
                .map_err(|e| anyhow::anyhow!("RowConverter error: {}", e))?;

            let rows = converter
                .convert_columns(batch.columns())
                .map_err(|e| anyhow::anyhow!("Row conversion error: {}", e))?;

            for row in rows.iter() {
                if !seen.insert(row.as_ref().to_vec()) {
                    return Err(anyhow::anyhow!("Primary key violation detected for columns {:?}: Duplicate row values found.", col_names));
                }
            }
        }

        Ok(())
    }

    /// Check if a single primary key value exists in the committed storage.
    /// Uses index-first searching (Bloom Filter -> Inverted Index -> Data Scan).
    pub(crate) async fn _check_pk_in_storage_async(
        &self,
        column: &str,
        value: &serde_json::Value,
    ) -> Result<bool> {
        let manifest = self.manifest().await?;
        let manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let all_entries = manager.load_all_entries(&manifest).await?;

        use futures::stream::{self, StreamExt};

        // Parallelize segment checks with a concurrency limit
        let concurrency = 8; // Adjust based on hardware
        let result_stream = stream::iter(all_entries.into_iter().rev())
            .map(|entry| {
                let store = self.store.clone();
                let uri = self.uri.clone();
                let config = self.get_config();
                let schema = manifest.schemas.last().cloned().unwrap_or_default();
                let column = column.to_string();
                let value = value.clone();
                let entry_path = entry.file_path.clone();
                let entry_size = entry.file_size_bytes as u64;

                async move {
                    let mut reader =
                        HybridReader::new(config, store, &uri).with_iceberg_schema(schema);

                    reader.config.parquet_path = Some(entry_path);
                    reader.config.file_size = Some(entry_size);

                    reader.check_value_exists(&column, &value).await
                }
            })
            .buffer_unordered(concurrency)
            .filter_map(|res| async {
                match res {
                    Ok(true) => Some(Ok(true)),
                    Ok(false) => None,
                    Err(e) => Some(Err(e)),
                }
            });

        let mut pinned_stream = Box::pin(result_stream);
        let result = pinned_stream.next().await;

        match result {
            Some(res) => res,  // Found a match or error
            None => Ok(false), // No matches found in any segment
        }
    }

    /// Check if any keys in the batch already exist in the table (Primary Key Enforcement)
    pub async fn check_primary_key_uniqueness_async(
        &self,
        batch: &RecordBatch,
        columns: &[String],
    ) -> Result<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }

        // Validate that all PK columns are present in the batch.
        for c in columns {
            batch.schema().index_of(c)?;
        }

        // Build a row-value IN list filter over the batch's primary key values.
        let mut pk_filter = match PrimaryKeyFilter::from_batch(batch, columns) {
            Some(f) if !f.is_empty() => f,
            _ => return Ok(()),
        };

        // Check manifests
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (_, all_entries, _) = manifest_manager.load_latest_full().await?;

        let planner = QueryPlanner::new();

        if columns.len() == 1 {
            // Single-column PK: push down as an IN list (Expr::InList) against the index.
            if let Some(query_filter) = pk_filter.to_query_filter() {
                let expr = FilterExpr::DataFusion(query_filter.to_expr());
                let candidates = planner.prune_entries(&all_entries, Some(&expr), None);

                for (entry, _) in candidates {
                    let reader = self.segment_reader(&entry)?;
                    if let Ok(Some(bm)) = reader.get_scalar_filter_bitmap(&query_filter).await {
                        let deleted = reader.load_merged_deletes().await?;
                        let alive_bm = bm - deleted;

                        if !alive_bm.is_empty() {
                            return Err(anyhow::anyhow!(
                                "Duplicate primary key error: {} IN batch already exists",
                                pk_filter.describe()
                            ));
                        }
                    }
                }
            }
            return Ok(());
        }

        // Multi-column PK: push down the whole row-value IN list as a single
        // expression: (c1 = v1 AND c2 = v2) OR (c1 = v3 AND c2 = v4) ...
        // Limit samples for performance in MVP.
        pk_filter.rows.truncate(100);

        let expr = FilterExpr::DataFusion(pk_filter.to_expr());
        let candidates = planner.prune_entries(&all_entries, Some(&expr), None);

        for (entry, _) in candidates {
            let reader = self.segment_reader(&entry)?;

            let bitmap_opt = Self::pk_match_bitmap(&reader, &pk_filter).await?;

            if let Some(bm) = bitmap_opt {
                // Subtract logically deleted rows!
                let deleted = reader.load_merged_deletes().await?;
                let alive_bm = bm - deleted;

                tracing::debug!(
                    "PK Check for {}: Alive bits: {}",
                    pk_filter.describe(),
                    alive_bm.len()
                );

                if !alive_bm.is_empty() {
                    return Err(anyhow::anyhow!(
                        "Duplicate primary key error: {} already exists",
                        pk_filter.describe()
                    ));
                }
            }
        }

        Ok(())
    }

    /// Build a [`HybridReader`] for a manifest entry, resolving the
    /// partition-aware base path and segment id.
    pub(crate) fn segment_reader(&self, entry: &ManifestEntry) -> Result<HybridReader> {
        let path = std::path::Path::new(&entry.file_path);
        let rel_parent = path.parent().and_then(|p| p.to_str()).unwrap_or("");
        let full_base_path = if rel_parent.is_empty() {
            self.uri.clone()
        } else {
            format!("{}/{}", self.uri, rel_parent)
        };

        let seg_id = entry
            .file_path
            .split('/')
            .next_back()
            .unwrap_or(&entry.file_path)
            .replace(".parquet", "");

        let config = SegmentConfig::new(&full_base_path, &seg_id)
            .with_index_files(entry.index_files.clone())
            .with_delete_files(entry.delete_files.clone());
        Ok(HybridReader::new(config, self.store.clone(), &self.uri))
    }

    /// Compute the bitmap of rows in `reader` matching any row of `pk_filter`.
    ///
    /// For each candidate PK row, intersects the per-column equality bitmaps,
    /// then unions the result across rows (row-value IN list semantics).
    /// Returns `None` if no inverted index is available for the PK columns.
    pub(crate) async fn pk_match_bitmap(
        reader: &HybridReader,
        pk_filter: &PrimaryKeyFilter,
    ) -> Result<Option<RoaringBitmap>> {
        let mut union_bm = RoaringBitmap::new();
        let mut any_index = false;

        for row in &pk_filter.rows {
            let mut row_bm: Option<RoaringBitmap> = None;
            for (col, val) in pk_filter.columns.iter().zip(row.iter()) {
                let eq_filter = QueryFilter {
                    column: col.clone(),
                    min: Some(val.clone()),
                    min_inclusive: true,
                    max: Some(val.clone()),
                    max_inclusive: true,
                    values: None,
                    negated: false,
                };
                match reader.get_scalar_filter_bitmap(&eq_filter).await {
                    Ok(Some(bm)) => {
                        row_bm = Some(match row_bm {
                            Some(prev) => prev & bm,
                            None => bm,
                        });
                    }
                    Ok(None) => {
                        row_bm = None;
                        break;
                    }
                    Err(e) => return Err(e),
                }
            }
            if let Some(rbm) = row_bm {
                any_index = true;
                union_bm |= rbm;
            }
        }

        if any_index {
            Ok(Some(union_bm))
        } else {
            Ok(None)
        }
    }
}

// -----------------------------------------------------------------------
// PrimaryKeyFilter — Row-Value In-List Pushdown
// -----------------------------------------------------------------------

/// A primary key filter representing a set of candidate primary key rows.
///
/// This is the internal representation of a "row-value IN list" pushdown,
/// e.g. `(id, region) IN ((1, 'US'), (2, 'CA'))`, which is logically
/// equivalent to `(id = 1 AND region = 'US') OR (id = 2 AND region = 'CA')`.
///
/// It can be built from a DataFusion expression (including `Expr::InList`)
/// via [`PrimaryKeyFilter::from_expr`], or directly from a [`RecordBatch`]
/// via [`PrimaryKeyFilter::from_batch`]. It can be converted back to a
/// DataFusion expression ([`PrimaryKeyFilter::to_expr`]) for planner pruning,
/// or to a [`QueryFilter`] ([`PrimaryKeyFilter::to_query_filter`]) for
/// single-column inverted-index pushdown.
#[derive(Debug, Clone)]
pub struct PrimaryKeyFilter {
    /// The primary key columns, in order.
    pub columns: Vec<String>,
    /// Candidate rows; each row holds one value per column in `columns`.
    pub rows: Vec<Vec<Value>>,
}

impl PrimaryKeyFilter {
    /// Create a new filter from explicit columns and rows.
    pub fn new(columns: Vec<String>, rows: Vec<Vec<Value>>) -> Self {
        Self { columns, rows }
    }

    /// Whether the filter has any candidate rows.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Build a filter from a DataFusion expression.
    ///
    /// Supported forms (all restricted to the given `pk_columns`):
    /// - `col = literal`
    /// - `col IN (literal, ...)` — `Expr::InList`
    /// - `AND` of the above on disjoint columns — a single multi-column row
    /// - `OR` of the above — multiple rows (row-value IN list)
    ///
    /// Returns `None` if the expression cannot be represented as a primary
    /// key row-value IN list (e.g. negation, ranges, unknown columns).
    pub fn from_expr(expr: &Expr, pk_columns: &[String]) -> Option<Self> {
        fn get_col(e: &Expr) -> Option<String> {
            match e {
                Expr::Column(c) => Some(c.name.clone()),
                Expr::Cast(c) => get_col(&c.expr),
                Expr::TryCast(c) => get_col(&c.expr),
                _ => None,
            }
        }

        fn get_lit(e: &Expr) -> Option<Value> {
            match e {
                Expr::Literal(scalar, _) => crate::core::planner::scalar_to_json_value(scalar),
                Expr::Cast(c) => get_lit(&c.expr),
                Expr::TryCast(c) => get_lit(&c.expr),
                _ => None,
            }
        }

        /// Extract (columns, rows) from a sub-expression.
        fn extract(e: &Expr) -> Option<(Vec<String>, Vec<Vec<Value>>)> {
            match e {
                // col IN (v1, v2, ...)
                Expr::InList(in_list) if !in_list.negated => {
                    let col = get_col(&in_list.expr)?;
                    let mut rows = Vec::new();
                    for v in &in_list.list {
                        if let Some(val) = get_lit(v) {
                            rows.push(vec![val]);
                        }
                    }
                    if rows.is_empty() {
                        return None;
                    }
                    Some((vec![col], rows))
                }
                // col = literal
                Expr::BinaryExpr(b) if b.op == Operator::Eq => {
                    let (col, val) =
                        if let (Some(c), Some(v)) = (get_col(&b.left), get_lit(&b.right)) {
                            (c, v)
                        } else if let (Some(c), Some(v)) = (get_col(&b.right), get_lit(&b.left)) {
                            (c, v)
                        } else {
                            return None;
                        };
                    Some((vec![col], vec![vec![val]]))
                }
                // AND: combine disjoint column sets (cartesian product of rows)
                Expr::BinaryExpr(b) if b.op == Operator::And => {
                    let (lcols, lrows) = extract(&b.left)?;
                    let (rcols, rrows) = extract(&b.right)?;
                    if lcols.iter().any(|c| rcols.contains(c)) {
                        return None; // overlapping columns are ambiguous
                    }
                    let mut cols = lcols;
                    cols.extend(rcols);
                    let mut rows = Vec::new();
                    for lr in &lrows {
                        for rr in &rrows {
                            let mut row = lr.clone();
                            row.extend(rr.iter().cloned());
                            rows.push(row);
                        }
                    }
                    Some((cols, rows))
                }
                // OR: union of rows (same column order required)
                Expr::BinaryExpr(b) if b.op == Operator::Or => {
                    let (lcols, lrows) = extract(&b.left)?;
                    let (rcols, rrows) = extract(&b.right)?;
                    if lcols != rcols {
                        return None;
                    }
                    let mut rows = lrows;
                    rows.extend(rrows);
                    Some((lcols, rows))
                }
                _ => None,
            }
        }

        let (cols, rows) = extract(expr)?;
        // Reject expressions referencing non-PK columns (would change semantics).
        if cols.iter().any(|c| !pk_columns.contains(c)) {
            return None;
        }
        // Reorder to PK column order.
        let col_idx: Vec<usize> = pk_columns
            .iter()
            .filter_map(|c| cols.iter().position(|x| x == c))
            .collect();
        if col_idx.is_empty() {
            return None;
        }
        let ordered_cols: Vec<String> = col_idx.iter().map(|&i| cols[i].clone()).collect();
        let rows = rows
            .into_iter()
            .map(|r| col_idx.iter().map(|&i| r[i].clone()).collect())
            .collect();
        Some(Self {
            columns: ordered_cols,
            rows,
        })
    }

    /// Build a filter from the primary key columns of a [`RecordBatch`].
    ///
    /// Rows with null or unsupported cell types are skipped. Returns `None`
    /// if any of `columns` is not present in the batch.
    pub fn from_batch(batch: &RecordBatch, columns: &[String]) -> Option<Self> {
        let col_arrays: Vec<&arrow::array::ArrayRef> = columns
            .iter()
            .map(|c| batch.column_by_name(c))
            .collect::<Option<Vec<_>>>()?;

        let mut rows = Vec::new();
        for i in 0..batch.num_rows() {
            let mut row = Vec::new();
            let mut ok = true;
            for arr in &col_arrays {
                match Self::cell_value(arr, i) {
                    Some(v) => row.push(v),
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                rows.push(row);
            }
        }

        Some(Self {
            columns: columns.to_vec(),
            rows,
        })
    }

    /// Extract a single cell value from an array, if the type is supported.
    fn cell_value(arr: &arrow::array::ArrayRef, i: usize) -> Option<Value> {
        if arr.is_null(i) {
            return None;
        }
        if let Some(a) = arr.as_any().downcast_ref::<arrow::array::Int32Array>() {
            Some(Value::Number(a.value(i).into()))
        } else if let Some(a) = arr.as_any().downcast_ref::<arrow::array::Int64Array>() {
            Some(Value::Number(a.value(i).into()))
        } else if let Some(a) = arr.as_any().downcast_ref::<arrow::array::Float64Array>() {
            serde_json::Number::from_f64(a.value(i)).map(serde_json::Value::Number)
        } else if let Some(a) = arr.as_any().downcast_ref::<arrow::array::StringArray>() {
            Some(Value::String(a.value(i).to_string()))
        } else {
            arr.as_any()
                .downcast_ref::<arrow::array::BooleanArray>()
                .map(|a| Value::Bool(a.value(i)))
        }
    }

    /// Convert back to a DataFusion expression.
    ///
    /// - Single column: `col = v` (one row) or `col IN (v1, v2, ...)` (many rows).
    /// - Multiple columns: OR of ANDs of equalities (row-value IN list).
    pub fn to_expr(&self) -> Expr {
        use datafusion::prelude::{col, lit};

        if self.columns.len() == 1 {
            let col_expr = col(&self.columns[0]);
            let values: Vec<Value> = self.rows.iter().map(|r| r[0].clone()).collect();
            if values.len() == 1 {
                col_expr.eq(crate::core::planner::json_to_scalar(&values[0]))
            } else {
                let list = values
                    .iter()
                    .map(crate::core::planner::json_to_scalar)
                    .collect();
                col_expr.in_list(list, false)
            }
        } else {
            let mut row_exprs: Vec<Expr> = Vec::new();
            for row in &self.rows {
                let mut row_expr: Option<Expr> = None;
                for (c, v) in self.columns.iter().zip(row.iter()) {
                    let e = col(c).eq(crate::core::planner::json_to_scalar(v));
                    row_expr = Some(match row_expr {
                        Some(prev) => prev.and(e),
                        None => e,
                    });
                }
                if let Some(re) = row_expr {
                    row_exprs.push(re);
                }
            }
            row_exprs
                .into_iter()
                .reduce(|a: Expr, b: Expr| a.or(b))
                .unwrap_or_else(|| lit(true))
        }
    }

    /// Convert to a single-column [`QueryFilter`] with an IN list, for
    /// inverted-index pushdown. Returns `None` for multi-column filters.
    pub fn to_query_filter(&self) -> Option<QueryFilter> {
        if self.columns.len() != 1 || self.rows.is_empty() {
            return None;
        }
        let values: Vec<Value> = self.rows.iter().map(|r| r[0].clone()).collect();
        Some(QueryFilter {
            column: self.columns[0].clone(),
            min: None,
            min_inclusive: false,
            max: None,
            max_inclusive: false,
            values: Some(values),
            negated: false,
        })
    }

    /// Human-readable description of the filter, for error messages.
    pub fn describe(&self) -> String {
        fn format_value(v: &Value) -> String {
            match v {
                Value::String(s) => format!("'{}'", s),
                other => other.to_string(),
            }
        }

        if self.columns.len() == 1 {
            let vals: Vec<String> = self.rows.iter().map(|r| format_value(&r[0])).collect();
            format!("{} IN ({})", self.columns[0], vals.join(", "))
        } else {
            let rows: Vec<String> = self
                .rows
                .iter()
                .map(|r| r.iter().map(format_value).collect::<Vec<_>>().join(", "))
                .collect();
            format!("({}) IN ({})", self.columns.join(", "), rows.join("), ("))
        }
    }
}

#[cfg(test)]
mod primary_key_filter_tests {
    use super::*;
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use datafusion::prelude::{col, lit};
    use std::sync::Arc;

    #[test]
    fn test_from_expr_in_list() {
        // id IN (1, 2, 3)
        let expr = col("id").in_list(vec![lit(1i64), lit(2i64), lit(3i64)], false);
        let pk = PrimaryKeyFilter::from_expr(&expr, &["id".to_string()]).unwrap();
        assert_eq!(pk.columns, vec!["id".to_string()]);
        assert_eq!(pk.rows.len(), 3);
        assert_eq!(pk.rows[0], vec![Value::from(1i64)]);
        assert_eq!(pk.rows[2], vec![Value::from(3i64)]);
    }

    #[test]
    fn test_from_expr_negated_in_list_rejected() {
        let expr = col("id").in_list(vec![lit(1i64), lit(2i64)], true);
        assert!(PrimaryKeyFilter::from_expr(&expr, &["id".to_string()]).is_none());
    }

    #[test]
    fn test_from_expr_equality() {
        let expr = col("id").eq(lit(42i64));
        let pk = PrimaryKeyFilter::from_expr(&expr, &["id".to_string()]).unwrap();
        assert_eq!(pk.columns, vec!["id".to_string()]);
        assert_eq!(pk.rows, vec![vec![Value::from(42i64)]]);
    }

    #[test]
    fn test_from_expr_multi_column_and() {
        // (a = 1 AND b = 'x')
        let expr = col("a").eq(lit(1i64)).and(col("b").eq(lit("x")));
        let pk = PrimaryKeyFilter::from_expr(&expr, &["a".to_string(), "b".to_string()]).unwrap();
        assert_eq!(pk.columns, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(pk.rows, vec![vec![Value::from(1i64), Value::from("x")]]);
    }

    #[test]
    fn test_from_expr_row_value_in_list() {
        // (a = 1 AND b = 'x') OR (a = 2 AND b = 'y')
        let row1 = col("a").eq(lit(1i64)).and(col("b").eq(lit("x")));
        let row2 = col("a").eq(lit(2i64)).and(col("b").eq(lit("y")));
        let expr = row1.or(row2);
        let pk = PrimaryKeyFilter::from_expr(&expr, &["a".to_string(), "b".to_string()]).unwrap();
        assert_eq!(pk.columns, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(pk.rows.len(), 2);
        assert_eq!(pk.rows[0], vec![Value::from(1i64), Value::from("x")]);
        assert_eq!(pk.rows[1], vec![Value::from(2i64), Value::from("y")]);
    }

    #[test]
    fn test_from_expr_non_pk_column_rejected() {
        let expr = col("id").eq(lit(1i64)).and(col("other").eq(lit(2i64)));
        assert!(PrimaryKeyFilter::from_expr(&expr, &["id".to_string()]).is_none());
    }

    #[test]
    fn test_from_batch_and_to_query_filter() {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
        )
        .unwrap();

        let pk = PrimaryKeyFilter::from_batch(&batch, &["id".to_string()]).unwrap();
        assert_eq!(pk.columns, vec!["id".to_string()]);
        assert_eq!(pk.rows.len(), 3);

        let qf = pk.to_query_filter().unwrap();
        assert_eq!(qf.column, "id");
        assert_eq!(qf.values.as_ref().unwrap().len(), 3);
        assert!(!qf.negated);
    }

    #[test]
    fn test_to_expr_multi_column_roundtrip() {
        let pk = PrimaryKeyFilter::new(
            vec!["a".to_string(), "b".to_string()],
            vec![
                vec![Value::from(1i64), Value::from("x")],
                vec![Value::from(2i64), Value::from("y")],
            ],
        );
        let expr = pk.to_expr();
        let parsed =
            PrimaryKeyFilter::from_expr(&expr, &["a".to_string(), "b".to_string()]).unwrap();
        assert_eq!(parsed.columns, pk.columns);
        assert_eq!(parsed.rows, pk.rows);
    }
}
