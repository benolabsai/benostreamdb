// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Rewrite a moved table's manifest entry paths to the table's current location.
//!
//! Iceberg manifests store absolute data-file URIs. Moving a table on disk
//! leaves those URIs pointing at the old location, so the engine can no longer
//! resolve them (`resolve_object_path` only relativizes paths that start with
//! the current table URI). This tool loads the latest manifest, rewrites every
//! entry's `file_path` (and any delete-file paths) from the old prefix to the
//! current table URI, and commits the result as a new manifest version with the
//! entries inline — dropping the stale manifest-list pointer so the old
//! absolute paths are no longer read.
//!
//! Usage:
//! ```text
//! cargo run --release --bin fix_table_paths -- file:///new/location/nodes
//! ```

use benostreamdb::core::manifest::ManifestManager;
use benostreamdb::core::storage::create_object_store;
use object_store::path::Path;

/// Longest common directory prefix of a set of paths, ending in `/`.
///
/// Returns an empty string when the paths are already relative (nothing to
/// strip) or share no directory.
fn common_dir_prefix<'a>(paths: impl Iterator<Item = &'a str>) -> String {
    let mut it = paths;
    let Some(first) = it.next() else {
        return String::new();
    };
    if !first.contains("://") && !first.starts_with('/') {
        return String::new();
    }
    let mut prefix = first.to_string();
    for p in it {
        while !prefix.is_empty() && !p.starts_with(&prefix) {
            prefix.pop();
        }
        if prefix.is_empty() {
            break;
        }
    }
    match prefix.rfind('/') {
        Some(pos) => prefix[..=pos].to_string(),
        None => String::new(),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let uri = std::env::args()
        .nth(1)
        .expect("usage: fix_table_paths <table-uri>");

    let store = create_object_store(&uri)?;
    let mgr = ManifestManager::new(store.clone(), "", &uri);
    let (mut manifest, entries, ver) = mgr.load_latest_full().await?;

    let old_prefix = common_dir_prefix(entries.iter().map(|e| e.file_path.as_str()));
    let new_prefix = format!("{}/", uri.trim_end_matches('/'));
    println!("manifest version: {ver}");
    println!("entries: {}", entries.len());
    println!("old prefix: {old_prefix}");
    println!("new prefix: {new_prefix}");

    let rewrite = |p: &str| -> String {
        if old_prefix.is_empty() {
            p.to_string()
        } else if let Some(rest) = p.strip_prefix(&old_prefix) {
            format!("{new_prefix}{rest}")
        } else {
            p.to_string()
        }
    };

    let mut rewritten = entries;
    let mut changed = 0usize;
    for e in &mut rewritten {
        let np = rewrite(&e.file_path);
        if np != e.file_path {
            changed += 1;
        }
        e.file_path = np;
        for f in &mut e.index_files {
            f.file_path = rewrite(&f.file_path);
        }
        for d in &mut e.delete_files {
            d.file_path = rewrite(&d.file_path);
        }
    }
    for d in &mut manifest.delete_files {
        d.file_path = rewrite(&d.file_path);
    }
    println!("rewrote {changed} entry path(s)");

    // Commit as a new version with the entries inline. Clearing
    // `manifest_list_path` is essential: otherwise `load_all_entries` would
    // also read the old avro list and return both the old and new paths.
    manifest.entries = rewritten;
    manifest.manifest_list_path = None;
    manifest.prev_version = Some(ver);
    manifest.version = ver + 1;
    manifest.timestamp_ms = chrono::Utc::now().timestamp_millis();

    let path = Path::from(format!("_manifest/v{}.json", ver + 1));
    let bytes = serde_json::to_vec_pretty(&manifest)?;
    use object_store::{PutMode, PutOptions};
    let opts = PutOptions {
        mode: PutMode::Create,
        ..Default::default()
    };
    store.put_opts(&path, bytes.into(), opts).await?;
    println!("wrote {path}");

    Ok(())
}
