use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hyperstreamdb::{Table, VectorSearchParams};
use arrow::record_batch::RecordBatch;
use arrow::datatypes::{Schema, Field, DataType};
use arrow::array::{Int64Array, Float64Array, Int32Array, Float32Array, FixedSizeListArray};
use std::sync::Arc;
use tempfile::TempDir;
use tokio::runtime::Runtime;

/// Utility to create a standard test batch for ingest/query
fn create_test_batch(num_rows: usize) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("category", DataType::Int32, false),
    ]));

    let ids = Int64Array::from_iter_values(0..num_rows as i64);
    let values = Float64Array::from_iter_values((0..num_rows).map(|i| i as f64));
    let categories = Int32Array::from_iter_values((0..num_rows).map(|i| (i % 10) as i32));

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(ids),
            Arc::new(values),
            Arc::new(categories),
        ],
    ).unwrap()
}

/// Utility to create a vector batch for HNSW benchmarks
fn create_vector_batch(num_rows: usize, dim: usize) -> RecordBatch {
    let inner_field = Arc::new(Field::new("item", DataType::Float32, true));
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("embedding", DataType::FixedSizeList(inner_field.clone(), dim as i32), false),
    ]));

    let ids = Int64Array::from_iter_values(0..num_rows as i64);
    
    let mut values = Vec::with_capacity(num_rows * dim);
    for i in 0..num_rows * dim {
        values.push(i as f32 / 1000.0);
    }
    let values_array = Float32Array::from(values);
    
    let embedding = FixedSizeListArray::try_new(
        inner_field,
        dim as i32,
        Arc::new(values_array),
        None,
    ).unwrap();

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(ids),
            Arc::new(embedding),
        ],
    ).unwrap()
}

/// Benchmark: Ingest throughput (In-memory)
fn bench_ingest_in_memory(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let num_rows = 100_000;
    let batch_size = 10_000;
    let batch = create_test_batch(batch_size);

    c.bench_function("ingest_in_memory", |b| {
        b.to_async(&rt).iter_custom(|iters| {
            let batch = batch.clone();
            async move {
                let mut total_duration = std::time::Duration::ZERO;
                for _ in 0..iters {
                    let tmp_dir = TempDir::new().unwrap();
                    let table_uri = format!("file://{}", tmp_dir.path().to_str().unwrap());
                    let table = Table::new_async(table_uri).await.unwrap();
                    
                    let start = std::time::Instant::now();
                    for _ in 0..(num_rows / batch_size) {
                        table.write_async(vec![batch.clone()]).await.unwrap();
                    }
                    table.commit_async().await.unwrap();
                    total_duration += start.elapsed();
                }
                total_duration
            }
        });
    });
}

/// Benchmark: Indexed Query Latency
fn bench_query_indexed(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let num_rows = 100_000;
    let batch = create_test_batch(num_rows);
    
    // Setup table
    let tmp_dir = TempDir::new().unwrap();
    let table_uri = format!("file://{}", tmp_dir.path().to_str().unwrap());
    
    let table = rt.block_on(async {
        let mut t = Table::new_async(table_uri).await.unwrap();
        t.add_index_columns_async(vec!["id".to_string()]).await.unwrap(); 
        t.write_async(vec![batch]).await.unwrap();
        t.commit_async().await.unwrap();
        t.wait_for_background_tasks_async().await.unwrap();
        t
    });

    c.bench_function("query_indexed_in_memory", |b| {
        b.to_async(&rt).iter(|| {
            let table = table.clone();
            async move {
                black_box(table.read_async(Some("id > 50000"), None, None).await.unwrap())
            }
        });
    });
}

/// Benchmark: Vector Search Latency
fn bench_vector_search_in_memory(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let num_vectors = 10_000;
    let dim = 128;
    let batch = create_vector_batch(num_vectors, dim);
    
    // Setup table
    let tmp_dir = TempDir::new().unwrap();
    let table_uri = format!("file://{}", tmp_dir.path().to_str().unwrap());
    
    let table = rt.block_on(async {
        let mut t = Table::new_async(table_uri).await.unwrap();
        t.add_index_columns_async(vec!["embedding".to_string()]).await.unwrap();
        t.write_async(vec![batch]).await.unwrap();
        t.commit_async().await.unwrap();
        t.wait_for_background_tasks_async().await.unwrap();
        t
    });

    let vs_params = VectorSearchParams::new(
        "embedding",
        (0..dim).map(|i| i as f32 / 100.0).collect(),
        10,
    );

    c.bench_function("vector_search_in_memory", |b| {
        let vs_params = vs_params.clone();
        b.to_async(&rt).iter(|| {
            let vs_params = vs_params.clone();
            let table = table.clone();
            async move {
                black_box(table.read_async(None, Some(vs_params), None).await.unwrap())
            }
        });
    });
}

/// Benchmark: Compaction Speed
fn bench_compaction_in_memory(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let num_segments = 20;
    let rows_per_segment = 5_000;
    let batch = create_test_batch(rows_per_segment);

    c.bench_function("compaction_in_memory", |b| {
        b.to_async(&rt).iter_custom(|iters| {
            let batch = batch.clone();
            async move {
                let mut total_duration = std::time::Duration::ZERO;
                for _ in 0..iters {
                    let tmp_dir = TempDir::new().unwrap();
                    let table_uri = format!("file://{}", tmp_dir.path().to_str().unwrap());
                    let table = Table::new_async(table_uri).await.unwrap();
                    
                    for _ in 0..num_segments {
                        table.write_async(vec![batch.clone()]).await.unwrap();
                        table.commit_async().await.unwrap();
                    }
                    
                    let start = std::time::Instant::now();
                    table.compact(None).unwrap();
                    total_duration += start.elapsed();
                }
                total_duration
            }
        });
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_ingest_in_memory, bench_query_indexed, bench_vector_search_in_memory, bench_compaction_in_memory
);
criterion_main!(benches);
