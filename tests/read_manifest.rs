use benostreamdb::core::manifest::manager::ManifestManager;
use tokio::runtime::Runtime;

fn main() {
    let rt = Runtime::new().unwrap();
    rt.block_on(async {
        let uri = "/tmp/pytest-of-ralbright/pytest-15/test_shortest_path0/test_local_graph";
        let store = benostreamdb::core::storage::create_object_store(uri).unwrap();
        let store = std::sync::Arc::new(store);
        let mm = ManifestManager::new(store, "", uri);
        let (manifest, _) = mm.load_latest().await.unwrap();
        let entries = mm.load_all_entries(&manifest).await.unwrap();
        for e in entries {
            println!("Entry: {}", e.file_path);
            for idx in e.index_files {
                println!(
                    "  idx: {} (col={:?}, type={})",
                    idx.file_path, idx.column_name, idx.index_type
                );
            }
        }
    });
}
