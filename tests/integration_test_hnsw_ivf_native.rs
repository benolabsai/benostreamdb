// Copyright (c) 2026 Richard Albright. All rights reserved.

use anyhow::Result;
use arrow::array::Int32Array;
use arrow::record_batch::RecordBatch;
use hyperstreamdb::SegmentConfig;
use hyperstreamdb::core::reader::HybridReader;

use object_store::memory::InMemory;
use std::sync::Arc;
// use object_store::ObjectStore;

#[tokio::test]
async fn test_hnsw_ivf_native_integration() -> Result<()> {
    // 1. Setup In-Memory Store
    let store = Arc::new(InMemory::new());
    let _segment_id = "seg_test_native";
    
    // 2. Create Validation Data (Vectors)
    let dim = 128;
    let n_vectors = 1000;
    
    // Create random vectors
    // For reproducibility, we can use a simple generator or zero/one patterns
    // Let's make vector i have i at index 0..
    let mut vectors: Vec<Vec<f32>> = Vec::new();
    for i in 0..n_vectors {
        let mut vec = vec![0.0; dim];
        vec[0] = i as f32 / n_vectors as f32; // normalized-ish
        vectors.push(vec);
    }
    
    // 3. Build Index Manually (Simulating SegmentWriter)
    // We need to use HnswIvfIndex directly first or use a Writer if available.
    // The SegmentWriter logic is complex to mock entirely here without writing files.
    // However, HnswIvfIndex::build and save can be tested.
    
    use hyperstreamdb::core::index::hnsw_ivf::HnswIvfIndex;
    let algo = hyperstreamdb::core::manifest::IndexAlgorithm::hnsw();
    let index = HnswIvfIndex::build(vectors.clone(), hyperstreamdb::core::index::VectorMetric::L2, Some(10), Some(16), &algo)?;
    
    // 4. Save Index to Store
    // HnswIvfIndex::save writes to local filesystem currently (std::fs::File).
    // This is a limitation for pure in-memory testing of the Index struct itself without FS.
    // But since we are testing integration, we can use a temp dir.
    
    let temp_dir = tempfile::tempdir()?;
    let base_path = temp_dir.path().join("seg_test_native.embedding");
    let base_path_str = base_path.to_str().unwrap();
    
    index.save(base_path_str)?;
    
    // 5. Verify Files Exist
    assert!(base_path.with_file_name(format!("{}.centroids.parquet", base_path.file_name().unwrap().to_str().unwrap())).exists());
    
    // 6. Test Loading via HybridReader
    // HybridReader uses ObjectStore for Parquet but local FS for HNSW index loading currently?
    // Let's check reader.rs: 
    // "For HNSW index loading, we need the absolute filesystem path"
    // So if we use a local Path for base_path in config, it should work.
    
    let config = SegmentConfig::new("", "seg_test_native");
        
    let _reader = HybridReader::new(config, store.clone(), "");
    
    // 7. Perform Search (Skip InMemory search as we invoke LocalFileSystem later)
    let query = vectors[50].clone(); // Should match itself
    let k = 5;
    
    // We skip searching with 'reader' (InMemory) because we haven't written the parquet file to 'store'.
    // let results = reader.vector_search_index("embedding", &query, k, None).await?;
    
    // 8. Assertions
    // Note: HybridReader returns (RecordBatch, distances)
    // But here we didn't write the main Parquet file, only the index files!
    // HybridReader::vector_search_index attempts to read rows from Parquet after getting row IDs from index.
    // This will fail if "seg_test_native.parquet" doesn't exist.
    
    // So we need to write a dummy Parquet file for the segment too.
    {
         use parquet::arrow::ArrowWriter;
         use arrow::array::Int32Array;
         use arrow::datatypes::{Schema, DataType, Field};
         
         // Helper to build list array
         // (Skipping full vector column creation for brevity, just IDs to verify loop)
         let id_array = Int32Array::from_iter_values(0..n_vectors);
         
         let schema = Arc::new(Schema::new(vec![
             Field::new("id", DataType::Int32, false),
         ]));
         
         let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(id_array)])?;
         
         let pq_path = temp_dir.path().join("seg_test_native.parquet");
         let file = std::fs::File::create(&pq_path)?;
         let mut writer = ArrowWriter::try_new(file, schema, None)?;
         writer.write(&batch)?;
         writer.close()?;
         
         // Also verify the reader can find it via ObjectStore?
         // HybridReader uses `store` for parquet. We need `store` to point to temp dir?
         // object_store::local::LocalFileSystem
    }
    
    // Re-create reader with LocalFileSystem store
    use object_store::local::LocalFileSystem;
    let local_store = Arc::new(LocalFileSystem::new_with_prefix(temp_dir.path())?);
     
    // Config base path should be empty if store is rooted? 
    // Or if config.base_path is absolute, reader handles it. 
    // Let's use absolute path in config and LocalFileSystem rooted at temp_dir
    let config = SegmentConfig::new("", "seg_test_native");
    let reader = HybridReader::new(config, local_store, "");

    let query_val = hyperstreamdb::core::index::VectorValue::Float32(query);
    let results = reader.vector_search_index("embedding", &query_val, k, None, hyperstreamdb::core::index::VectorMetric::L2, None, None).await?;
    
    assert!(!results.is_empty(), "Should return results");
    
    let (batch, dists) = &results[0];
    assert!(batch.num_rows() > 0);
    assert_eq!(dists.len(), batch.num_rows());
    
    // Check if we found the query vector (ID 50)
    let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let mut found = false;
    for (i, &dist) in dists.iter().enumerate() {
        if ids.value(i) == 50 {
            found = true;
            println!("Found ID 50 with distance {}", dist);
            break;
        }
    }
    assert!(found, "Should find the exact match (ID 50)");

    Ok(())
}

/// Regression test for: https://github.com/rla3rd/hyperstreamdb/issues/blob_type_puffin_dispatch
///
/// Bug: When a ManifestEntry's IndexFile had `blob_type = Some("hnsw_tq8")`, the reader
/// branched into `load_puffin_async()`, which attempted to GET a single-file Puffin container
/// (e.g. `seg_*.embedding.tq8`).  That file never exists — the writer always produces the
/// multi-file layout (.centroids.parquet, .cluster_N.hnsw.graph/data, .cluster_N.mapping.parquet).
///
/// Fix: `search_hnsw_ivf` now unconditionally calls `load_async_with_cache_key`, which
/// discovers component files by prefix, regardless of `blob_type`.
#[tokio::test]
async fn test_tq8_index_loaded_via_multifile_not_puffin() -> Result<()> {
    use hyperstreamdb::core::index::hnsw_ivf::HnswIvfIndex;
    use hyperstreamdb::core::index::{VectorMetric, VectorValue};
    use hyperstreamdb::core::manifest::{IndexAlgorithm, IndexFile};
    use object_store::local::LocalFileSystem;
    use arrow::array::{Float32Builder, FixedSizeListBuilder, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};

    let dim = 32usize;
    let n_vectors = 200usize;

    // Build synthetic vectors: vector i has value (i as f32 / n) in every dimension.
    let vectors: Vec<Vec<f32>> = (0..n_vectors)
        .map(|i| vec![i as f32 / n_vectors as f32; dim])
        .collect();

    // Build a TQ8 index (this is what sets blob_type = Some("hnsw_tq8") in the manifest).
    let algo = IndexAlgorithm::HnswTq8 {
        metric: "l2".to_string(),
        complexity: 32,
        quality: 8,
    };
    let index = HnswIvfIndex::build(
        vectors.clone(),
        VectorMetric::L2,
        Some(4),
        Some(16),
        &algo,
    )?;

    // Save using the multi-file layout (same as the production writer path).
    let temp_dir = tempfile::tempdir()?;
    let seg_id = "seg_regression_tq8";
    let base_path = temp_dir.path().join(format!("{}.embedding.tq8", seg_id));
    let base_path_str = base_path.to_str().unwrap();
    index.save(base_path_str)?;

    // Verify the component files exist (but NOT a bare .tq8 container).
    assert!(
        temp_dir.path().join(format!("{}.embedding.tq8.centroids.parquet", seg_id)).exists(),
        "centroids.parquet must exist"
    );
    assert!(
        !temp_dir.path().join(format!("{}.embedding.tq8", seg_id)).exists(),
        "no single-file Puffin container should exist — the bug depended on finding this file"
    );

    // Write a minimal Parquet data file so HybridReader can return row data.
    {
        use parquet::arrow::ArrowWriter;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dim as i32,
                ),
                false,
            ),
        ]));

        let id_array = Arc::new(Int32Array::from_iter_values(0..n_vectors as i32));

        let mut list_builder =
            FixedSizeListBuilder::new(Float32Builder::new(), dim as i32);
        for vec in &vectors {
            for &v in vec {
                list_builder.values().append_value(v);
            }
            list_builder.append(true);
        }
        let embedding_array = Arc::new(list_builder.finish());

        let batch = arrow::record_batch::RecordBatch::try_new(
            schema.clone(),
            vec![id_array, embedding_array],
        )?;

        let pq_path = temp_dir.path().join(format!("{}.parquet", seg_id));
        let file = std::fs::File::create(&pq_path)?;
        let mut writer = ArrowWriter::try_new(file, schema, None)?;
        writer.write(&batch)?;
        writer.close()?;
    }

    // Build a SegmentConfig that mirrors what the production path produces:
    // blob_type is set, file_path points to the base (no extension), index_type = "vector".
    let index_file = IndexFile {
        file_path: format!("{}.embedding.tq8", seg_id),
        index_type: "vector".to_string(),
        column_name: Some("embedding".to_string()),
        blob_type: Some("hnsw_tq8".to_string()), // ← this is what triggered the bug
        offset: None,
        length: None,
    };

    let mut config = hyperstreamdb::SegmentConfig::new("", seg_id);
    config.index_files.push(index_file);

    let local_store = Arc::new(LocalFileSystem::new_with_prefix(temp_dir.path())?);
    let reader = hyperstreamdb::core::reader::HybridReader::new(config, local_store, "");

    // Query for vector 42 — it should be the nearest neighbour of itself.
    let query = VectorValue::Float32(vectors[42].clone());
    let results = reader
        .vector_search_index("embedding", &query, 5, None, VectorMetric::L2, None, None)
        .await?;

    // Before the fix this returned an empty vec (flat-scan fallback on error).
    assert!(
        !results.is_empty(),
        "Should return results — empty means the index fell back to flat scan (Puffin load bug)"
    );

    // Collect all returned row IDs across batches.
    let mut found_42 = false;
    for (batch, _dists) in &results {
        assert!(batch.num_rows() > 0, "Result batch should not be empty");
        let ids = batch
            .column_by_name("id")
            .expect("id column must be present")
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        for i in 0..ids.len() {
            if ids.value(i) == 42 {
                found_42 = true;
            }
        }
    }
    // TQ8 is a lossy quantized index; exact self-recall is not guaranteed at k=5 with
    // tiny synthetic data, but at k=5 of 200 vectors with L2 the correct result must appear.
    assert!(found_42, "ID 42 should appear in the top-5 results");

    Ok(())
}

