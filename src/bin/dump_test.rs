use arrow::ipc::reader::FileReader;
use hyperstreamdb::core::index::hnsw_ivf::HnswIvfIndex;
use hyperstreamdb::core::index::VectorMetric;
use hyperstreamdb::core::manifest::IndexAlgorithm;
use std::fs::File;

fn main() {
    let dim = 32usize;
    let n_vectors = 200usize;
    let vectors: Vec<Vec<f32>> = (0..n_vectors)
        .map(|i| vec![i as f32 / n_vectors as f32; dim])
        .collect();

    let algo = IndexAlgorithm::HnswTq8 {
        metric: "l2".to_string(),
        complexity: 32,
        quality: 8,
    };
    let index = HnswIvfIndex::build(
        vectors,
        VectorMetric::L2,
        Some(4),
        Some(16),
        &algo,
        0,
    ).unwrap();

    let local_path = "/tmp/dump_test_index";
    std::fs::create_dir_all("/tmp").unwrap();
    index.save(local_path).unwrap();

    // Now let's read the graph file
    let graph_path = format!("{}.cluster_0.hnsw.graph", local_path);
    println!("Reading: {}", graph_path);
    let f = File::open(graph_path).unwrap();
    let mut reader = FileReader::try_new(f, None).unwrap();
    let schema = reader.schema();
    println!("Schema of generated file: {:#?}", schema);
}
