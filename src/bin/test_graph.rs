use hyperstreamdb::core::index::hnsw_ivf::HnswIvfIndex;
use hyperstreamdb::core::index::VectorMetric;
use hyperstreamdb::core::manifest::IndexAlgorithm;
use hyperstreamdb::core::index::VectorValue;

fn main() {
    let dim = 16;
    let n_vectors = 1000;
    let mut vectors = Vec::new();
    for i in 0..n_vectors {
        let mut vec = vec![0.0; dim];
        vec[0] = i as f32 / n_vectors as f32;
        vectors.push(vec);
    }
    
    let algo: IndexAlgorithm = serde_json::from_str(r#"{"type": "hnsw", "metric": "L2", "complexity": 16, "quality": 100, "build_device": "Cpu", "search_device": "Cpu"}"#).unwrap();

    let mut index = HnswIvfIndex::build(
        vectors.clone(),
        VectorMetric::L2,
        Some(10), // n_lists
        Some(16), // m
        &algo,
        0,
    ).unwrap();
    
    std::fs::create_dir_all("/tmp/hyperstreamdb_test_graph").unwrap();
    index.save("/tmp/hyperstreamdb_test_graph").unwrap();
    
    let loaded_index = HnswIvfIndex::load("/tmp/hyperstreamdb_test_graph").unwrap();
    
    let mut query = vec![0.0; dim];
    query[0] = 50.0 / 1000.0;
    
    let res = loaded_index.search(&VectorValue::Float32(query), 5, 2, None);
    println!("ARROW IPC SEARCH:");
    println!("{:?}", res);
}
