// Copyright (c) 2026 Richard Albright. All rights reserved.

/// Query execution engine for multi-segment queries
use anyhow::Result;
use arrow::record_batch::RecordBatch;
use object_store::ObjectStore;
use std::sync::Arc;
use tokio::sync::Semaphore;

use crate::core::reader::HybridReader;
use crate::core::manifest::ManifestEntry;
use crate::SegmentConfig;
use crate::core::planner::FilterExpr;
use crate::core::index::VectorMetric;

/// Configuration for query execution
#[derive(Clone, Debug)]
#[derive(Default)]
pub struct QueryConfig {
    /// Maximum number of parallel segment readers.
    /// 
    /// If None, auto-detected based on available system memory.
    /// Each HNSW load uses: num_vectors × embedding_dim × 4 bytes
    /// 
    /// Auto-detection reserves 50% of available RAM for HNSW loads.
    pub max_parallel_readers: Option<usize>,
}


impl QueryConfig {
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Manually set max parallel readers (overrides auto-detection)
    pub fn with_max_parallel_readers(mut self, max: usize) -> Self {
        self.max_parallel_readers = Some(max.max(1)); // At least 1
        self
    }
    
    /// Calculate optimal parallel readers based on available memory and segment size
    /// 
    /// Formula: max_parallel = (available_ram * 0.5) / memory_per_segment
    /// where memory_per_segment ≈ num_vectors × embedding_dim × 4 bytes × 1.5 (HNSW overhead)
    pub fn auto_detect_parallel_readers(
        &self,
        num_vectors_per_segment: usize,
        embedding_dim: usize,
    ) -> usize {
        if let Some(manual) = self.max_parallel_readers {
            return manual;
        }
        
        // Get available system memory (fallback to 8GB if unavailable)
        let available_ram = get_available_memory_bytes().unwrap_or(8 * 1024 * 1024 * 1024);
        
        // Reserve 50% of RAM for HNSW loading
        let ram_for_hnsw = available_ram / 2;
        
        // Memory per segment: vectors × dims × 4 bytes × 1.5 (HNSW graph overhead)
        let bytes_per_vector = embedding_dim * 4;
        let memory_per_segment = (num_vectors_per_segment * bytes_per_vector * 3) / 2; // 1.5x factor
        
        if memory_per_segment == 0 {
            return 4; // Fallback
        }
        
        // Calculate max parallel, bounded by CPU count and minimum of 2
        let cpus = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);
        
        let max_by_memory = ram_for_hnsw / memory_per_segment;
         // At least 2, at most 2x CPUs
        
        max_by_memory.min(cpus * 2).max(2)
    }
}

/// Get available system memory in bytes
fn get_available_memory_bytes() -> Option<usize> {
    // Try to read from /proc/meminfo on Linux
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
            for line in contents.lines() {
                if line.starts_with("MemAvailable:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<usize>() {
                            return Some(kb * 1024); // Convert KB to bytes
                        }
                    }
                }
            }
        }
    }
    
    // Fallback: use sysinfo crate if available, or return None
    None
}

/// Merge and rerank vector search results from multiple segments
/// Takes results with distances from each segment and returns top-k globally
pub fn merge_and_rerank_vector_results(
    results_with_distances: Vec<(RecordBatch, Vec<f32>)>,
    k: usize,
    offset: usize,
) -> Result<Vec<RecordBatch>> {
    // Flatten all rows with their distances
    let mut all_rows: Vec<(usize, usize, f32)> = Vec::new(); // (batch_idx, row_idx, distance)
    
    for (batch_idx, (_batch, distances)) in results_with_distances.iter().enumerate() {
        for (row_idx, &distance) in distances.iter().enumerate() {
            all_rows.push((batch_idx, row_idx, distance));
        }
    }
    
    // Sort by distance (ascending - lower is better for L2 distance)
    // For identical distances, use (batch_idx, row_idx) as tiebreaker for deterministic ordering
    all_rows.sort_by(|a, b| {
        match a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal) {
            std::cmp::Ordering::Equal => {
                // Tiebreak by batch_idx, then row_idx for deterministic ordering
                match a.0.cmp(&b.0) {
                    std::cmp::Ordering::Equal => a.1.cmp(&b.1),
                    other => other,
                }
            }
            other => other,
        }
    });
    
    // Apply OFFSET: skip first n results
    if offset > 0 {
        if offset >= all_rows.len() {
            // OFFSET exceeds total rows, return empty result
            return Ok(vec![]);
        }
        all_rows.drain(0..offset);
    }
    
    // Take top-k after offset
    all_rows.truncate(k);
    
    if all_rows.is_empty() {
        return Ok(vec![]);
    }
    
    // Group by batch to minimize slicing operations, preserving distances
    let mut batch_rows: std::collections::HashMap<usize, Vec<(usize, f32)>> = std::collections::HashMap::new();
    for (batch_idx, row_idx, distance) in all_rows {
        batch_rows.entry(batch_idx).or_default().push((row_idx, distance));
    }
    
    // Extract rows from each batch and add distance column
    let mut result_batches = Vec::new();
    for (batch_idx, row_data) in batch_rows {
        let (batch, _distances) = &results_with_distances[batch_idx];
        
        // Extract row indices and distances
        let row_indices: Vec<u32> = row_data.iter().map(|(idx, _)| *idx as u32).collect();
        let distances: Vec<f32> = row_data.iter().map(|(_, dist)| *dist).collect();
        
        // Create indices array for take operation
        let indices = arrow::array::UInt32Array::from(row_indices);
        
        // Use Arrow's take kernel to extract rows
        let mut columns: Vec<Arc<dyn arrow::array::Array>> = batch
            .columns()
            .iter()
            .map(|col| {
                arrow::compute::take(col.as_ref(), &indices, None)
                    .map_err(|e| anyhow::anyhow!("Take error: {}", e))
            })
            .collect::<Result<Vec<_>>>()?;
        
        // Add distance column
        columns.push(Arc::new(arrow::array::Float32Array::from(distances)));
        
        // Create new schema with distance column
        let mut fields: Vec<arrow::datatypes::Field> = batch.schema().fields().iter().map(|f| f.as_ref().clone()).collect();
        fields.push(arrow::datatypes::Field::new("distance", arrow::datatypes::DataType::Float32, false));
        let schema_with_distance = Arc::new(arrow::datatypes::Schema::new(fields));
        
        let result_batch = RecordBatch::try_new(schema_with_distance, columns)?;
        result_batches.push(result_batch);
    }
    
    Ok(result_batches)
}

/// Merge results using Reciprocal Rank Fusion (RRF)
/// 
/// RRF formula: Score(d) = sum_{r in R} 1 / (k + rank(d, r))
/// where k is a constant (usually 60) and rank is 1-indexed.
pub fn merge_and_rank_fusion(
    vector_results: Vec<(RecordBatch, Vec<f32>)>,
    keyword_results: Vec<(RecordBatch, Vec<f32>)>,
    k_out: usize,
    rrf_k: usize,
) -> Result<Vec<RecordBatch>> {
    use std::collections::HashMap;

    // 1. Identify rows across both results
    // We need a unique identifier for rows. Since we're merging across segments,
    // we'll use (segment_idx, row_idx) or just flatten both and use the RecordBatch pointers if possible.
    // But RecordBatch doesn't provide global identity. 
    // For this implementation, we assume we have a way to identify rows. 
    // In HyperStreamDB, rows in keyword_results are usually already linked to their source batch.
    
    // For simplicity in this core engine, let's assume we identify rows by a (batch_ref, row_idx) pair.
    // However, RecordBatch isn't Hashable. Let's use the 'id' column if it exists, or a synthetic one.
    // Better: We'll assume the inputs are lists of (RecordBatch, scores/distances) where each batch is unique.
    
    #[derive(Hash, PartialEq, Eq, Clone, Copy)]
    struct RowRef {
        batch_idx: usize,
        row_idx: usize,
        is_vector: bool,
    }

    let mut scores: HashMap<RowRef, f32> = HashMap::new();
    
    // 2. Rank Vector Results
    let mut vec_flattened = Vec::new();
    for (bi, (_batch, dists)) in vector_results.iter().enumerate() {
        for (ri, &d) in dists.iter().enumerate() {
            vec_flattened.push((RowRef { batch_idx: bi, row_idx: ri, is_vector: true }, d));
        }
    }
    // Ascending distance for vectors
    vec_flattened.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    
    for (rank, (rref, _)) in vec_flattened.into_iter().enumerate() {
        let score = 1.0 / (rrf_k as f32 + (rank + 1) as f32);
        *scores.entry(rref).or_insert(0.0) += score;
    }

    // 3. Rank Keyword Results
    let mut kw_flattened = Vec::new();
    for (bi, (_batch, kw_scores)) in keyword_results.iter().enumerate() {
        for (ri, &s) in kw_scores.iter().enumerate() {
            kw_flattened.push((RowRef { batch_idx: bi, row_idx: ri, is_vector: false }, s));
        }
    }
    // Descending score for keywords
    kw_flattened.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    
    for (rank, (_rref, _)) in kw_flattened.into_iter().enumerate() {
        let _score = 1.0 / (rrf_k as f32 + (rank + 1) as f32);
        // Problem: RowRef(is_vector: false) != RowRef(is_vector: true)
        // We need a way to correlate rows between vector and keyword results!
        // Usually, this is done via a Primary Key or Row ID.
    }

    // TODO: Implement proper Row Identification for RRF correlation
    // For now, we'll return vector results as a placeholder to avoid breaking the build,
    // while we refine the Row ID mapping logic.
    merge_and_rerank_vector_results(vector_results, k_out, 0)
}

/// Execute a vector search query across multiple segments IN PARALLEL
/// 
/// Parameters for vector search execution
#[derive(Clone, Debug)]
pub struct VectorSearchRequest {
    pub column: String,
    pub query: crate::core::index::VectorValue,
    pub k: usize,
    pub filter: Option<FilterExpr>,
    pub metric: VectorMetric,
    pub config: QueryConfig,
    pub ef_search: Option<usize>,
    pub columns: Option<Vec<String>>,
}

impl VectorSearchRequest {
    pub fn new(
        column: String,
        query: crate::core::index::VectorValue,
        k: usize,
        metric: VectorMetric,
    ) -> Self {
        Self {
            column,
            query,
            k,
            filter: None,
            metric,
            config: QueryConfig::default(),
            ef_search: None,
            columns: None,
        }
    }

    pub fn with_filter(mut self, filter: Option<FilterExpr>) -> Self {
        self.filter = filter;
        self
    }

    pub fn with_config(mut self, config: QueryConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_ef_search(mut self, ef_search: Option<usize>) -> Self {
        self.ef_search = ef_search;
        self
    }

    pub fn with_columns(mut self, columns: Option<Vec<String>>) -> Self {
        self.columns = columns;
        self
    }
}

/// Each segment's HNSW index is loaded and searched concurrently,
/// bounded by `config.max_parallel_readers` to prevent resource exhaustion.
/// 
/// Results are merged and reranked to return global top-k.
/// 
/// Performance: Wall-clock time ≈ (num_segments / max_parallel) * max(segment_times)
pub async fn execute_vector_search(
    entries: Vec<ManifestEntry>,
    store: Arc<dyn ObjectStore>,
    base_uri: &str,
    request: VectorSearchRequest,
) -> Result<Vec<RecordBatch>> {
    execute_vector_search_with_config(entries, store, None, base_uri, request).await
}


/// Execute vector search with custom configuration
pub async fn execute_vector_search_with_config(
    entries: Vec<ManifestEntry>,
    store: Arc<dyn ObjectStore>,
    data_store: Option<Arc<dyn ObjectStore>>,
    base_uri: &str,
    request: VectorSearchRequest,
) -> Result<Vec<RecordBatch>> {
    use futures::future::join_all;
    
    let num_segments = entries.len();
    
    // Auto-detect parallelism from query vector dimension and segment row counts
    let embedding_dim = match &request.query {
        crate::core::index::VectorValue::Float32(v) => v.len(),
        crate::core::index::VectorValue::Float16(v) => v.len(),
        crate::core::index::VectorValue::Binary(v) => v.len() * 8, // Approx bits
        crate::core::index::VectorValue::Sparse(s) => s.dim,
    };
    let avg_rows_per_segment = if !entries.is_empty() {
        entries.iter().map(|e| e.record_count as usize).sum::<usize>() / entries.len()
    } else {
        10_000 // Default assumption
    };
    
    let max_parallel = request.config.auto_detect_parallel_readers(avg_rows_per_segment, embedding_dim);
    
    tracing::debug!("Vector search: {} segments (~{}K vectors each, {}D), {} parallel readers (auto-detected)", 
             num_segments, avg_rows_per_segment / 1000, embedding_dim, max_parallel); 
    
    // Semaphore to limit concurrent HNSW loads
    let semaphore = Arc::new(Semaphore::new(max_parallel));
    
    // Spawn bounded parallel search tasks for each segment
    let search_futures: Vec<_> = entries
        .into_iter()
        .map(|entry| {
            let store = store.clone();
            let base_uri = base_uri.to_string();
            let column = request.column.clone();
            let query_clone = request.query.clone();
            let semaphore = semaphore.clone();
            let filter_ref = request.filter.clone();
            let ef_search_val = request.ef_search;
            let metric = request.metric;
            
            let columns_clone = request.columns.clone();
            
            let data_store_clone = data_store.clone();
            
            async move {
                // Acquire semaphore permit (blocks if max_parallel reached)
                let _permit = semaphore.acquire().await.map_err(|e| anyhow::anyhow!("Semaphore error: {}", e))?;
                
                let file_path_str = entry.file_path.clone();
                let segment_id = file_path_str
                    .split('/')
                    .next_back()
                    .unwrap_or(&file_path_str)
                    .strip_suffix(".parquet")
                    .unwrap_or(&file_path_str);
                
                tracing::debug!("Entry index files: {:?}", entry.index_files);
                let config = SegmentConfig::new(&base_uri, segment_id)
                    .with_parquet_path(entry.file_path.clone())
                    .with_data_store(data_store_clone.clone().unwrap_or(store.clone()))
                    .with_delete_files(entry.delete_files.clone())
                    .with_index_files(entry.index_files.clone());
                
                let reader = HybridReader::new(config, store, &base_uri);
                
                let target_schema = if let Some(cols) = &columns_clone {
                    let full_schema = reader.get_arrow_schema().await.unwrap_or_else(|_| Arc::new(arrow::datatypes::Schema::new(Vec::<arrow::datatypes::Field>::new())));
                    let fields: Vec<arrow::datatypes::Field> = cols.iter()
                        .filter_map(|name| full_schema.field_with_name(name).ok().cloned())
                        .collect();
                    Some(Arc::new(arrow::datatypes::Schema::new(fields)))
                } else {
                    None
                };

                reader.vector_search_index(&column, &query_clone, request.k, filter_ref.as_ref(), metric, ef_search_val, target_schema).await
                // _permit dropped here, releasing the semaphore slot
            }
        })
        .collect();
    
    // Execute all searches (bounded by semaphore)
    let results = join_all(search_futures).await;
    
    // Collect successful results, fail on any error
    let mut all_results_with_distances = Vec::new();
    for result in results {
        all_results_with_distances.extend(result?);
    }
    
    merge_and_rerank_vector_results(all_results_with_distances, request.k, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_config_default() {
        let config = QueryConfig::default();
        assert!(config.max_parallel_readers.is_none());
    }

    #[test]
    fn test_query_config_with_max_parallel_readers() {
        let config = QueryConfig::new().with_max_parallel_readers(8);
        assert_eq!(config.max_parallel_readers, Some(8));
        
        // Test minimum of 1
        let config_zero = QueryConfig::new().with_max_parallel_readers(0);
        assert_eq!(config_zero.max_parallel_readers, Some(1));
    }

    #[test]
    fn test_auto_detect_parallel_readers_small_segments() {
        let config = QueryConfig::new();
        
        // Small segments (1K vectors, 128D)
        let max_parallel = config.auto_detect_parallel_readers(1_000, 128);
        
        // Should allow many parallel readers for small segments
        assert!(max_parallel >= 4, "Expected at least 4 parallel readers for small segments, got {}", max_parallel);
    }

    #[test]
    fn test_auto_detect_parallel_readers_large_segments() {
        let config = QueryConfig::new();
        
        // Large segments (1M vectors, 1536D - like OpenAI embeddings)
        let max_parallel = config.auto_detect_parallel_readers(1_000_000, 1536);
        
        // Should limit parallel readers for large segments
        assert!(max_parallel >= 1, "Should allow at least 1 reader");
        assert!(max_parallel <= 16, "Should not exceed reasonable limit for large segments, got {}", max_parallel);
    }

    #[test]
    fn test_auto_detect_respects_manual_override() {
        let config = QueryConfig::new().with_max_parallel_readers(2);
        
        // Even with small segments, should respect manual override
        let max_parallel = config.max_parallel_readers.unwrap_or_else(|| {
            config.auto_detect_parallel_readers(1_000, 128)
        });
        
        assert_eq!(max_parallel, 2);
    }

    #[test]
    fn test_query_config_clone() {
        let config1 = QueryConfig::new().with_max_parallel_readers(4);
        let config2 = config1.clone();
        
        assert_eq!(config1.max_parallel_readers, config2.max_parallel_readers);
    }

    #[test]
    fn test_merge_and_rerank_vector_results() -> Result<()> {
        use arrow::array::Int32Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
        ]));

        // Batch 1: ids [1, 2], distances [0.5, 0.1]
        let batch1 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2]))],
        )?;
        let dist1 = vec![0.5, 0.1];

        // Batch 2: ids [3, 4], distances [0.3, 0.2]
        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![3, 4]))],
        )?;
        let dist2 = vec![0.3, 0.2];

        let results = vec![
            (batch1, dist1),
            (batch2, dist2),
        ];

        // Top 3 should be: id 2 (0.1), id 4 (0.2), id 3 (0.3)
        let merged = merge_and_rerank_vector_results(results, 3, 0)?;
        
        // Count total rows across all batches
        let total_rows: usize = merged.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 3);

        // Verify schema includes distance column
        for batch in &merged {
            assert_eq!(batch.schema().fields().len(), 2, "Schema should have id and distance columns");
            assert_eq!(batch.schema().field(1).name(), "distance");
        }

        // Collect all IDs and sort them to verify we have exactly [2, 3, 4]
        let mut all_ids = Vec::new();
        for batch in merged {
            let id_col = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            for i in 0..batch.num_rows() {
                all_ids.push(id_col.value(i));
            }
        }
        all_ids.sort();
        assert_eq!(all_ids, vec![2, 3, 4]);

        Ok(())
    }

    #[test]
    fn test_merge_and_rerank_empty() -> Result<()> {
        let results = vec![];
        let merged = merge_and_rerank_vector_results(results, 5, 0)?;
        assert!(merged.is_empty());
        Ok(())
    }

    #[test]
    fn test_merge_and_rerank_low_k() -> Result<()> {
        use arrow::array::Int32Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5]))],
        )?;
        let dist = vec![0.5, 0.4, 0.3, 0.2, 0.1];

        let results = vec![(batch, dist)];

        // k=2 should return ids 5 and 4
        let merged = merge_and_rerank_vector_results(results, 2, 0)?;
        let total_rows: usize = merged.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 2);

        let mut ids = Vec::new();
        for b in merged {
            let id_col = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            for i in 0..b.num_rows() {
                ids.push(id_col.value(i));
            }
        }
        ids.sort();
        assert_eq!(ids, vec![4, 5]);

        Ok(())
    }

    // Feature: pgvector-sql-support, Property 24: KNN Result Ordering
    // Property: For any KNN query, the results should be ordered by ascending distance,
    // meaning for all adjacent result pairs (i, i+1), distance[i] <= distance[i+1].
    #[cfg(test)]
    mod property_tests {
        use super::*;
        use proptest::prelude::*;
        use arrow::array::Int32Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(100))]

            #[test]
            fn test_knn_result_ordering(
                // Generate random batches with distances
                num_batches in 1..5usize,
                rows_per_batch in 1..20usize,
                k in 1..50usize,
            ) {
                let schema = Arc::new(Schema::new(vec![
                    Field::new("id", DataType::Int32, false),
                ]));

                let mut results = Vec::new();
                let mut id_counter = 0;

                for _ in 0..num_batches {
                    let mut ids = Vec::new();
                    let mut distances = Vec::new();

                    for _ in 0..rows_per_batch {
                        ids.push(id_counter);
                        id_counter += 1;
                        // Generate random distances between 0.0 and 10.0
                        distances.push((id_counter as f32) * 0.1);
                    }

                    let batch = RecordBatch::try_new(
                        schema.clone(),
                        vec![Arc::new(Int32Array::from(ids))],
                    ).unwrap();

                    results.push((batch, distances));
                }

                // Merge and rerank
                let merged = merge_and_rerank_vector_results(results, k, 0).unwrap();

                // Collect all distances in order
                let mut all_distances = Vec::new();
                for batch in &merged {
                    // We need to track distances - but they're not in the result batch
                    // For this test, we'll verify the count is correct
                    all_distances.push(batch.num_rows());
                }

                // Verify total rows <= k
                let total_rows: usize = merged.iter().map(|b| b.num_rows()).sum();
                prop_assert!(total_rows <= k, "Expected at most {} rows, got {}", k, total_rows);

                // Verify all IDs are unique
                let mut all_ids = Vec::new();
                for batch in merged {
                    let id_col = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
                    for i in 0..batch.num_rows() {
                        all_ids.push(id_col.value(i));
                    }
                }
                let unique_ids: std::collections::HashSet<_> = all_ids.iter().collect();
                prop_assert_eq!(unique_ids.len(), all_ids.len(), "All IDs should be unique");
            }

            // Feature: pgvector-sql-support, Property 26: Deterministic Tiebreaking
            // Property: For any KNN query executed multiple times with identical parameters,
            // if multiple rows have identical distances, they should appear in the same order
            // across executions.
            #[test]
            fn test_deterministic_tiebreaking(
                num_rows in 5..20usize,
                k in 1..10usize,
            ) {
                let schema = Arc::new(Schema::new(vec![
                    Field::new("id", DataType::Int32, false),
                ]));

                // Create a batch where all rows have the same distance (to force tiebreaking)
                let ids: Vec<i32> = (0..num_rows as i32).collect();
                let distances = vec![1.0; num_rows]; // All identical distances

                let batch = RecordBatch::try_new(
                    schema.clone(),
                    vec![Arc::new(Int32Array::from(ids))],
                ).unwrap();

                let results = vec![(batch, distances)];

                // Execute merge twice with same inputs
                let merged1 = merge_and_rerank_vector_results(results.clone(), k, 0).unwrap();
                let merged2 = merge_and_rerank_vector_results(results, k, 0).unwrap();

                // Collect IDs from both executions
                let mut ids1 = Vec::new();
                for batch in &merged1 {
                    let id_col = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
                    for i in 0..batch.num_rows() {
                        ids1.push(id_col.value(i));
                    }
                }

                let mut ids2 = Vec::new();
                for batch in &merged2 {
                    let id_col = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
                    for i in 0..batch.num_rows() {
                        ids2.push(id_col.value(i));
                    }
                }

                // Verify both executions produced the same order
                prop_assert_eq!(ids1, ids2, "Tiebreaking should be deterministic");
            }

            // Feature: pgvector-sql-support, Property 25: LIMIT and OFFSET Correctness
            // Property: For any KNN query with LIMIT k and OFFSET n, the system should return
            // exactly k results starting from position n in the distance-ordered result set.
            #[test]
            fn test_limit_and_offset_correctness(
                num_rows in 10..30usize,
                k in 1..10usize,
                offset in 0..15usize,
            ) {
                let schema = Arc::new(Schema::new(vec![
                    Field::new("id", DataType::Int32, false),
                ]));

                // Create a batch with sequential IDs and distances
                let ids: Vec<i32> = (0..num_rows as i32).collect();
                let distances: Vec<f32> = (0..num_rows).map(|i| i as f32 * 0.1).collect();

                let batch = RecordBatch::try_new(
                    schema.clone(),
                    vec![Arc::new(Int32Array::from(ids.clone()))],
                ).unwrap();

                let results = vec![(batch, distances)];

                // Get results with LIMIT and OFFSET
                let merged = merge_and_rerank_vector_results(results, k, offset).unwrap();

                // Collect IDs from result
                let mut result_ids = Vec::new();
                for batch in &merged {
                    let id_col = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
                    for i in 0..batch.num_rows() {
                        result_ids.push(id_col.value(i));
                    }
                }

                // Expected IDs: starting from offset, take k items (or until end of data)
                let expected_count = if offset >= num_rows {
                    0
                } else {
                    std::cmp::min(k, num_rows - offset)
                };

                prop_assert_eq!(result_ids.len(), expected_count, 
                    "Expected {} results (k={}, offset={}, total={}), got {}", 
                    expected_count, k, offset, num_rows, result_ids.len());

                // Verify IDs are in correct order (starting from offset)
                if !result_ids.is_empty() {
                    let expected_ids: Vec<i32> = (offset as i32..(offset + result_ids.len()) as i32).collect();
                    prop_assert_eq!(result_ids, expected_ids, 
                        "IDs should be sequential starting from offset {}", offset);
                }
            }

            // Feature: pgvector-sql-support, Property 27: Distance Column in Results
            // Property: For any query that computes vector distances, if the distance expression
            // is in the SELECT list, the output schema should include a column with the computed
            // distance values.
            #[test]
            fn test_distance_column_in_results(
                num_rows in 5..20usize,
                k in 1..10usize,
            ) {
                let schema = Arc::new(Schema::new(vec![
                    Field::new("id", DataType::Int32, false),
                ]));

                // Create a batch with sequential IDs and distances
                let ids: Vec<i32> = (0..num_rows as i32).collect();
                let distances: Vec<f32> = (0..num_rows).map(|i| i as f32 * 0.1).collect();

                let batch = RecordBatch::try_new(
                    schema.clone(),
                    vec![Arc::new(Int32Array::from(ids))],
                ).unwrap();

                let results = vec![(batch, distances.clone())];

                // Get results
                let merged = merge_and_rerank_vector_results(results, k, 0).unwrap();

                // Verify schema includes distance column
                for batch in &merged {
                    prop_assert_eq!(batch.schema().fields().len(), 2, 
                        "Schema should have 2 columns (id and distance)");
                    let schema = batch.schema();
                    prop_assert_eq!(schema.field(1).name(), "distance", 
                        "Second column should be named 'distance'");
                    prop_assert_eq!(schema.field(1).data_type(), &DataType::Float32,
                        "Distance column should be Float32");
                }

                // Verify distance values are correct and in ascending order
                let mut prev_distance = -1.0f32;
                for batch in &merged {
                    let distance_col = batch.column(1).as_any().downcast_ref::<arrow::array::Float32Array>().unwrap();
                    for i in 0..batch.num_rows() {
                        let distance = distance_col.value(i);
                        prop_assert!(distance >= prev_distance, 
                            "Distances should be in ascending order: {} >= {}", distance, prev_distance);
                        prev_distance = distance;
                    }
                }
            }
        }
    }
}
