// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Relocate a BenoStreamDB / Iceberg table to a new on-disk location.
//!
//! Iceberg stores **absolute** paths in three places, so moving a table on disk
//! without rewriting all of them leaves both the engine and external readers
//! (PyIceberg, Trino, Spark) unable to resolve the data files:
//!
//! 1. the table metadata JSON — `location`, `snapshots[].manifest-list`,
//!    `metadata-log[].metadata-file`;
//! 2. the BenoStream manifest JSON — `entries[].file_path` (and delete files);
//! 3. the Avro manifest list (`snap-*.avro`) and manifest files (`*-m*.avro`) —
//!    `manifest_path` and `data_file.file_path`.
//!
//! This tool rewrites every absolute path under the old table location to the
//! new one, in place, and commits new metadata / manifest versions.
//!
//! Usage:
//! ```text
//! cargo run --release --bin relocate_table -- <new-table-uri> [--old-prefix <old-uri>]
//! ```
//!
//! The old prefix is auto-detected from the table metadata `location` (or the
//! common directory prefix of the manifest entries) when not supplied.
//!
//! Catalog: if the table is registered in a catalog (REST / Glue / Hive /
//! Unity / Nessie / JDBC), pass `--catalog-type` (plus `--catalog-url`,
//! `--namespace`, `--table`) and the tool will also repoint the catalog's
//! metadata-location at the new metadata file via a `set-metadata-location`
//! commit. Catalog credentials/endpoints are read from the environment by the
//! catalog implementations, exactly as the engine does.

use anyhow::{Context, Result};
use benostreamdb::core::catalog::{create_catalog_async, CatalogType};
use benostreamdb::core::manifest::ManifestManager;
use benostreamdb::core::storage::create_object_store;
use object_store::path::Path;
use object_store::ObjectStore;
use std::collections::HashMap;
use std::sync::Arc;

/// Recursively rewrite every string field that starts with `old` to `new`.
/// Returns whether anything changed.
fn rewrite_avro_value(v: &mut apache_avro::types::Value, old: &str, new: &str) -> bool {
    use apache_avro::types::Value;
    match v {
        Value::String(s) => {
            if let Some(rest) = s.strip_prefix(old) {
                *s = format!("{new}{rest}");
                true
            } else {
                false
            }
        }
        Value::Record(fields) => {
            let mut changed = false;
            for (_, val) in fields.iter_mut() {
                changed |= rewrite_avro_value(val, old, new);
            }
            changed
        }
        Value::Array(items) => {
            let mut changed = false;
            for item in items.iter_mut() {
                changed |= rewrite_avro_value(item, old, new);
            }
            changed
        }
        Value::Map(m) => {
            let mut changed = false;
            for val in m.values_mut() {
                changed |= rewrite_avro_value(val, old, new);
            }
            changed
        }
        Value::Union(_, inner) => rewrite_avro_value(inner, old, new),
        _ => false,
    }
}

/// Rewrite the absolute paths embedded in one Avro file, in place.
async fn rewrite_avro_file(
    store: &Arc<dyn ObjectStore>,
    path: &Path,
    old: &str,
    new: &str,
) -> Result<bool> {
    let bytes = store.get(path).await?.bytes().await?;
    let reader = apache_avro::Reader::new(&bytes[..])?;
    let schema = reader.writer_schema().clone();
    let mut records = Vec::new();
    let mut changed = false;
    for value in reader {
        let mut v = value?;
        changed |= rewrite_avro_value(&mut v, old, new);
        records.push(v);
    }
    if !changed {
        return Ok(false);
    }
    let mut writer = apache_avro::Writer::new(&schema, Vec::new());
    for r in records {
        writer.append(r)?;
    }
    let out = writer.into_inner()?;
    store.put(path, out.into()).await?;
    Ok(true)
}

/// Find the highest `v{N}.metadata.json` under `metadata/`.
async fn latest_metadata_version(store: &Arc<dyn ObjectStore>) -> Result<Option<u64>> {
    use futures::StreamExt;
    let mut stream = store.list(Some(&Path::from("metadata")));
    let mut max = None;
    while let Some(meta) = stream.next().await {
        let meta = meta?;
        let name = meta.location.filename().unwrap_or_default().to_string();
        if let Some(rest) = name.strip_prefix('v') {
            if let Some(num) = rest.strip_suffix(".metadata.json") {
                if let Ok(v) = num.parse::<u64>() {
                    max = Some(max.map_or(v, |m: u64| m.max(v)));
                }
            }
        }
    }
    Ok(max)
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let new_uri = args
        .get(1)
        .context("usage: relocate_table <new-table-uri> [--old-prefix <old-uri>]")?
        .clone();
    let arg_value = |flag: &str| -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let old_prefix_arg = arg_value("--old-prefix");
    let catalog_type_arg = arg_value("--catalog-type");
    let catalog_url = arg_value("--catalog-url");
    let catalog_token = arg_value("--catalog-token");
    let namespace = arg_value("--namespace");
    let table_name = arg_value("--table");

    let store = create_object_store(&new_uri)?;
    let new_prefix = format!("{}/", new_uri.trim_end_matches('/'));

    // ---- 1. Iceberg table metadata -------------------------------------
    let meta_ver = latest_metadata_version(&store)
        .await?
        .context("no metadata/v*.metadata.json found")?;
    let meta_path = Path::from(format!("metadata/v{meta_ver}.metadata.json"));
    let meta_bytes = store.get(&meta_path).await?.bytes().await?;
    let mut meta: serde_json::Value = serde_json::from_slice(&meta_bytes)?;

    let old_prefix = old_prefix_arg
        .or_else(|| {
            meta.get("location")
                .and_then(|v| v.as_str())
                .map(|s| format!("{}/", s.trim_end_matches('/')))
        })
        .context("could not determine old prefix; pass --old-prefix")?;

    println!("old prefix: {old_prefix}");
    println!("new prefix: {new_prefix}");

    let rewrite = |p: &str| -> String {
        match p.strip_prefix(&old_prefix) {
            Some(rest) => format!("{new_prefix}{rest}"),
            None => p.to_string(),
        }
    };

    // location
    if let Some(loc) = meta.get("location").and_then(|v| v.as_str()) {
        let nl = rewrite(loc);
        meta["location"] = serde_json::Value::String(nl);
    }
    // snapshots[].manifest-list
    if let Some(snaps) = meta.get_mut("snapshots").and_then(|v| v.as_array_mut()) {
        for s in snaps.iter_mut() {
            if let Some(ml) = s.get("manifest-list").and_then(|v| v.as_str()) {
                let nl = rewrite(ml);
                s["manifest-list"] = serde_json::Value::String(nl);
            }
        }
    }
    // metadata-log[].metadata-file
    if let Some(log) = meta.get_mut("metadata-log").and_then(|v| v.as_array_mut()) {
        for e in log.iter_mut() {
            if let Some(mf) = e.get("metadata-file").and_then(|v| v.as_str()) {
                let nl = rewrite(mf);
                e["metadata-file"] = serde_json::Value::String(nl);
            }
        }
    }

    let new_meta_ver = meta_ver + 1;
    let new_meta_path = Path::from(format!("metadata/v{new_meta_ver}.metadata.json"));
    store
        .put(&new_meta_path, serde_json::to_vec_pretty(&meta)?.into())
        .await?;
    store
        .put(
            &Path::from("metadata/version-hint.text"),
            new_meta_ver.to_string().into(),
        )
        .await?;
    println!("wrote {new_meta_path} (version-hint -> {new_meta_ver})");

    // ---- 2. BenoStream manifest ----------------------------------------
    // Iceberg requires absolute paths in the manifest, so rewrite to the new
    // absolute location. The engine relativizes entries against the table URI
    // on load (see `ManifestManager::load_all_entries`), so the writer still
    // sees store-relative paths internally.
    let mgr = ManifestManager::new(store.clone(), "", &new_uri);
    let (mut manifest, entries, ver) = mgr.load_latest_full().await?;
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
    println!("rewrote {changed} manifest entry path(s)");

    manifest.entries = rewritten;
    manifest.manifest_list_path = None;
    manifest.prev_version = Some(ver);
    manifest.version = ver + 1;
    manifest.timestamp_ms = chrono::Utc::now().timestamp_millis();
    let new_manifest_path = Path::from(format!("_manifest/v{}.json", ver + 1));
    store
        .put(
            &new_manifest_path,
            serde_json::to_vec_pretty(&manifest)?.into(),
        )
        .await?;
    println!("wrote {new_manifest_path}");

    // ---- 3. Avro manifest list + manifest files ------------------------
    use futures::StreamExt;
    let mut avro_stream = store.list(Some(&Path::from("_manifest")));
    let mut avro_paths = Vec::new();
    while let Some(meta) = avro_stream.next().await {
        let meta = meta?;
        if meta.location.to_string().ends_with(".avro") {
            avro_paths.push(meta.location);
        }
    }
    let mut avro_changed = 0usize;
    for p in &avro_paths {
        if rewrite_avro_file(&store, p, &old_prefix, &new_prefix).await? {
            avro_changed += 1;
        }
    }
    println!("rewrote {avro_changed}/{} avro file(s)", avro_paths.len());

    // ---- 4. Catalog metadata-location pointer --------------------------
    if let Some(ct) = catalog_type_arg {
        let catalog_type: CatalogType = ct.parse()?;
        let namespace = namespace.context("--namespace is required with --catalog-type")?;
        let table_name = table_name.context("--table is required with --catalog-type")?;

        let mut config: HashMap<String, String> = HashMap::new();
        if let Some(url) = catalog_url {
            config.insert("url".to_string(), url);
        }
        if let Some(token) = catalog_token {
            config.insert("token".to_string(), token);
        }

        let catalog = create_catalog_async(catalog_type, config).await?;
        let new_metadata_location = format!("{new_prefix}metadata/v{new_meta_ver}.metadata.json");
        let updates = vec![serde_json::json!({
            "action": "set-metadata-location",
            "metadata-location": new_metadata_location,
        })];
        catalog
            .commit_table(&namespace, &table_name, updates)
            .await?;
        println!("catalog pointer updated -> {new_metadata_location}");
    } else {
        println!(
            "no --catalog-type given; if this table is registered in a catalog, \
             repoint it manually to {new_prefix}metadata/v{new_meta_ver}.metadata.json"
        );
    }

    println!("relocate complete");
    Ok(())
}
