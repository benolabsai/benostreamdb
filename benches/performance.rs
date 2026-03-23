use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use hyperstreamdb::{SegmentConfig, core::{segment::HybridSegmentWriter, reader::HybridReader, compaction::{Compactor, CompactionOptions}, planner::FilterExpr, index::VectorMetric}};
use arrow::record_batch::RecordBatch;
use arrow::datatypes::{Schema, Field, DataType};
use arrow::array::{Int32Array, Float32Array, StringArray, FixedSizeListArray};
use std::sync::Arc;
use object_store::local::LocalFileSystem;
use rand::prelude::*;
use rand_distr::{Distribution, Normal, Zipf};
use hnsw_rs::prelude::*;
use std::fs::File;
use std::io::BufReader;
use hnsw_rs::hnswio::{load_description, load_hnsw};

/// Benchmark: Ingest throughput
fn bench_ingest(c: &mut Criterion) {
    let mut group = c.benchmark_group("ingest");
    
    for batch_size in [1_000, 10_000, 100_000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &size| {
                b.iter(|| {
                    let batch = create_test_batch(size);
                    let tmp_dir = tempfile::tempdir().unwrap();
                    let path = tmp_dir.path().to_str().unwrap();
                    let config = SegmentConfig::new(path, "test_seg");
                    let writer = HybridSegmentWriter::new(config);
                    writer.write_batch(&batch).unwrap();
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark: Query latency with index
fn bench_query_indexed(c: &mut Criterion) {
    // Setup: Create segment with index
    let batch = create_test_batch(100_000);
    let tmp_dir = tempfile::tempdir().unwrap();
    let path = tmp_dir.path().to_str().unwrap();
    let writer_config = SegmentConfig::new(path, "query_test")
        .with_columns_to_index(vec!["id".to_string()]);
    let writer = HybridSegmentWriter::new(writer_config);
    writer.write_batch(&batch).unwrap();
    
    // For Reader: Use relative path logic since store is rooted at tmp_dir
    // If we passed absolute path to reader config, it would be appended to store prefix
    let reader_config = SegmentConfig::new("", "query_test");
    let store = Arc::new(LocalFileSystem::new_with_prefix(path).unwrap());
    let reader = HybridReader::new(reader_config, store, path);
    
    let filter = hyperstreamdb::core::planner::QueryFilter::parse("id > 0").unwrap();
    
    c.bench_function("query_indexed", |b| {
        b.to_async(tokio::runtime::Runtime::new().unwrap())
            .iter(|| async {
                black_box(reader.query_index_first(&filter, None).await.unwrap())
            });
    });
}

/// Benchmark: Vector search (Load + Index Search)
/// Simulates finding 10 nearest neighbors in a 100k vector segment
fn bench_vector_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_search");
    
    // Setup: Create segment with vectors
    // Use a smaller size for bench setup to avoid timeout, or pre-generate?
    // Criterion setup is outside the measurement loop.
    let vec_dim = 128;
    let num_rows = 10_000; // Real world would be larger, but keeping it manageable for `cargo bench`
    
    let batch = create_vector_batch(num_rows, vec_dim);
    let tmp_dir = tempfile::tempdir().unwrap();
    let base_path = tmp_dir.path().to_str().unwrap();
    let config = SegmentConfig::new(base_path, "vec_bench")
        .with_columns_to_index(vec!["embedding".to_string()]);
    
    let writer = HybridSegmentWriter::new(config);
    writer.write_batch(&batch).unwrap();
    
    // Generate a random query vector
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();
    let query_vec: Vec<f32> = (0..vec_dim).map(|_| normal.sample(&mut rng)).collect();

    let reader_config = SegmentConfig::new("", "vec_bench");
    let store = Arc::new(LocalFileSystem::new_with_prefix(base_path).unwrap());
    let reader = HybridReader::new(reader_config, store, base_path);
    
    for k in [10, 100] {
        group.bench_with_input(
            BenchmarkId::from_parameter(k),
            &k,
            |b, &k| {
                b.to_async(tokio::runtime::Runtime::new().unwrap())
                    .iter(|| async {
                        black_box(reader.vector_search_index("embedding", &query_vec, k, None, VectorMetric::L2).await.unwrap())
                    });
            },
        );
    }
    
    group.finish();
}

/// Benchmark: Hybrid Search (Scalar Filter + Vector)
/// Scenario: "Find images where category='security' AND similarity(vec) > X"
fn bench_hybrid_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("hybrid_search");
    
    let vec_dim = 128;
    let num_rows = 10_000;
    
    // Create data with skewed category
    let batch = create_hybrid_batch(num_rows, vec_dim);
    let tmp_dir = tempfile::tempdir().unwrap();
    let base_path = tmp_dir.path().to_str().unwrap();
    let config = SegmentConfig::new(base_path, "hybrid_bench")
        .with_columns_to_index(vec!["embedding".to_string(), "id".to_string()]);
    
    let writer = HybridSegmentWriter::new(config);
    writer.write_batch(&batch).unwrap();
    
    let mut rng = rand::thread_rng();
    let query_vec: Vec<f32> = (0..vec_dim).map(|_| rng.gen()).collect();

    let reader_config = SegmentConfig::new("", "hybrid_bench");
    let store = Arc::new(LocalFileSystem::new_with_prefix(base_path).unwrap());
    let reader = HybridReader::new(reader_config, store, base_path);
    
    let filter = hyperstreamdb::core::planner::QueryFilter::parse("id > 0").unwrap();

    group.bench_function("hybrid_filter_50_percent", |b| {
        b.to_async(tokio::runtime::Runtime::new().unwrap())
            .iter(|| async {
                 // Pre-filter strategy (HybridReader way)
                 let expr = FilterExpr::DataFusion(filter.to_expr());
                 black_box(reader.vector_search_index("embedding", &query_vec, 10, Some(&expr), VectorMetric::L2).await.unwrap())
            });
    });
    
    group.finish();
}

/// Benchmark: High Selectivity Read (Mocking High Cardinality)
/// Scenario: Filter matches only 1 row out of 100,000 (e.g. ID lookup)
fn bench_high_selectivity(c: &mut Criterion) {
    // Generate data where only ONE id > 0
    let size = 100_000;
    
    // Only index i == 50000 will be > 0 (set to 1). Others 0.
    // Writer config: "if v > 0 { insert }"
    let mut ids = vec![0; size];
    ids[50000] = 1;
    let ids_array = Int32Array::from(ids);
    
    // Fill other cols
    let values_array = Int32Array::from(vec![0; size]);
    let categories: Vec<String> = (0..size).map(|_| "x".to_string()).collect();
    let cat_array = StringArray::from(categories);
    
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]);
    
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(ids_array),
            Arc::new(values_array),
            Arc::new(cat_array),
        ],
    ).unwrap();

    let tmp_dir = tempfile::tempdir().unwrap();
    let path = tmp_dir.path().to_str().unwrap();
    let config = SegmentConfig::new(path, "high_selectivity")
        .with_columns_to_index(vec!["id".to_string()]);
    let writer = HybridSegmentWriter::new(config);
    writer.write_batch(&batch).unwrap();
    
    // For Reader: Use relative path logic
    let reader_config = SegmentConfig::new("", "high_selectivity");
    let store = Arc::new(LocalFileSystem::new_with_prefix(path).unwrap());
    let reader = HybridReader::new(reader_config, store, path);
    
    let filter = hyperstreamdb::core::planner::QueryFilter::parse("id > 0").unwrap();

    c.bench_function("read_single_row_via_index", |b| {
        b.to_async(tokio::runtime::Runtime::new().unwrap())
            .iter(|| async {
                let batches = reader.query_index_first(&filter, None).await.unwrap();
                // Should return 1 row
                assert_eq!(batches[0].num_rows(), 1);
            });
    });
}

fn create_vector_batch(rows: usize, dim: usize) -> RecordBatch {
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();
    
    let mut values = Vec::with_capacity(rows * dim);
    for _ in 0..rows {
        for _ in 0..dim {
            values.push(normal.sample(&mut rng));
        }
    }
    
    let values_array = Float32Array::from(values);
    let vectors_array = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32,
        Arc::new(values_array),
        None
    ).unwrap();
    
    let schema = Schema::new(vec![
        Field::new("embedding", DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32
        ), false),
    ]);
    
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(vectors_array)]).unwrap()
}

fn create_hybrid_batch(rows: usize, dim: usize) -> RecordBatch {
    let mut rng = rand::thread_rng();
    
    // 1. ID Column (random integers 0..rows)
    // Use Zipf distribution to simulate skewed access patterns/metadata
    let zipf = Zipf::new(1000, 1.5).unwrap(); // Skewed distribution
    let ids: Vec<i32> = (0..rows).map(|_| zipf.sample(&mut rng) as i32).collect();
    
    // 2. Vector Column
    let mut values = Vec::with_capacity(rows * dim);
    for _ in 0..rows {
        for _ in 0..dim {
            values.push(rng.gen::<f32>());
        }
    }
    let values_array = Float32Array::from(values);
    let vectors_array = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32,
        Arc::new(values_array),
        None
    ).unwrap();
    
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("embedding", DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32
        ), false),
    ]);
    
    RecordBatch::try_new(
        Arc::new(schema), 
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(vectors_array)
        ]
    ).unwrap()
}

fn create_test_batch(size: usize) -> RecordBatch {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]);
    
    let ids: Vec<i32> = (0..size as i32).collect();
    let values: Vec<i32> = (0..size as i32).map(|i| i * 2).collect();
    let categories: Vec<String> = (0..size).map(|i| format!("cat_{}", i % 10)).collect();
    
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(Int32Array::from(values)),
            Arc::new(StringArray::from(categories)),
        ],
    ).unwrap()
}


/// Benchmark: Compaction
/// Measures time to compact 10 small segments into 1
fn bench_compaction(c: &mut Criterion) {
    let mut group = c.benchmark_group("compaction");
    // Configure for throughput (MB/s)? Or just latency?
    // Latency is fine for now.
    
    group.sample_size(10); // Compaction is slow, don't run 100 times
    
    group.bench_function("compact_10_segments", |b| {
        b.to_async(tokio::runtime::Runtime::new().unwrap())
            .iter_custom(|iters| async move {
                let mut total_duration = std::time::Duration::new(0, 0);
                
                for _ in 0..iters {
                    // Setup per iteration (Costly setup, but necessary for destructive test)
                    // We need fresh segments every time because compaction changes them.
                    
                    let tmp_dir = tempfile::tempdir().unwrap();
                    let path = tmp_dir.path().to_str().unwrap();
                    
                    // Create 10 segments of 1000 rows each
                    for i in 0..10 {
                        let batch = create_test_batch(1000);
                        let config = SegmentConfig::new(path, &format!("seg_{}", i));
                        let writer = HybridSegmentWriter::new(config);
                        writer.write_batch(&batch).unwrap();
                    }
                    
                    let uri = format!("file://{}", path);
                    let options = CompactionOptions {
                        target_file_size_bytes: 100 * 1024 * 1024,
                        min_file_size_bytes: 1024 * 1024 * 1024, // Compact everything < 1GB
                        strategy: "binpack".to_string(),
                        max_concurrent_bins: 4,
                        clustering: None,
                    };
                    let compactor = Compactor::new(&uri, options).unwrap();
                    
                    let start = std::time::Instant::now();
                    compactor.rewrite_data_files().await.unwrap();
                    total_duration += start.elapsed();
                }
                total_duration
            });
    });
    
    group.finish();
}

criterion_group!(benches, bench_ingest, bench_query_indexed, bench_vector_search, bench_hybrid_search, bench_high_selectivity, bench_compaction);
criterion_main!(benches);
