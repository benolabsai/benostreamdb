use hyperstreamdb::core::index::hnsw_rs::hnsw::Hnsw;
use hyperstreamdb::core::index::hnsw_rs::dist::DistL2;
use hyperstreamdb::core::index::hnsw_rs::arrow_hnsw::ArrowHnsw;
use hyperstreamdb::core::index::hnsw_rs::arrow_ipc::dump_arrow_ipc;

#[test]
fn scratch_test() {
    let mut hnsw = Hnsw::new(10, 100, 16, 200, DistL2);
    let v1 = vec![0.5f32, 0.0, 0.0, 0.0];
    let v2 = vec![0.1f32, 0.1, 0.1, 0.1];
    hnsw.insert((&v1, 50));
    hnsw.insert((&v2, 51));
    let buffer = dump_arrow_ipc(&hnsw).unwrap();
    
    let arrow_hnsw = ArrowHnsw::load_from_bytes(&buffer, DistL2).unwrap();
    println!("Vector 0: {:?}", arrow_hnsw.get_vector(0));
    println!("Vector 1: {:?}", arrow_hnsw.get_vector(1));
    
    let query = [0.5f32, 0.0, 0.0, 0.0];
    let neighbors = arrow_hnsw.search(&query, 5, 20, None);
    println!("Neighbors: {:?}", neighbors);
}
