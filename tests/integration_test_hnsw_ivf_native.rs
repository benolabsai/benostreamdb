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
    
    let index = HnswIvfIndex::build(vectors.clone(), hyperstreamdb::core::index::VectorMetric::L2, Some(10), Some(16), false)?;
    
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
         let id_array = Int32Array::from_iter_values(0..n_vectors as i32);
         
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

    let results = reader.vector_search_index("embedding", &query, k, None, hyperstreamdb::core::index::VectorMetric::L2).await?;
    
    assert!(!results.is_empty(), "Should return results");
    
    let (batch, dists) = &results[0];
    assert!(batch.num_rows() > 0);
    assert_eq!(dists.len(), batch.num_rows());
    
    // Check if we found the query vector (ID 50)
    let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let mut found = false;
    for i in 0..ids.len() {
        if ids.value(i) == 50 {
            found = true;
            println!("Found ID 50 with distance {}", dists[i]);
            break;
        }
    }
    assert!(found, "Should find the exact match (ID 50)");

    Ok(())
}
