// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Diagnostic: open a table and report whether its manifest entries resolve to
//! readable data files at the current location.

use benostreamdb::core::manifest::ManifestManager;
use benostreamdb::core::reader::HybridReader;
use benostreamdb::core::storage::create_object_store;
use benostreamdb::SegmentConfig;
use futures::StreamExt;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let uri = std::env::args()
        .nth(1)
        .expect("usage: check_table_paths <uri>");
    let store = create_object_store(&uri)?;
    let mgr = ManifestManager::new(store.clone(), "", &uri);
    let (_m, entries, _) = mgr.load_latest_full().await?;
    println!("entries: {}", entries.len());
    for e in entries.iter().take(3) {
        println!("  file_path: {}", e.file_path);
    }
    let e = &entries[0];
    let seg = e
        .file_path
        .split('/')
        .next_back()
        .unwrap()
        .strip_suffix(".parquet")
        .unwrap();
    let config = SegmentConfig::new(&uri, seg).with_parquet_path(e.file_path.clone());
    let reader = HybridReader::new(config, store, &uri);
    let mut s = reader.stream_row_groups(None, None).await?;
    let mut n = 0usize;
    while let Some(b) = s.next().await {
        n += b?.num_rows();
    }
    println!("rows read from first segment: {}", n);
    Ok(())
}
