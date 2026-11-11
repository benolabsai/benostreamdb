// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
//
// Verifies a BenoStreamDB table with the upstream `apache/iceberg-rust` reader.
//
// Run with:
//   BSDB_ORACLE_URI=file:///tmp/tbl BSDB_ORACLE_SUITE=v3 \
//   BSDB_ORACLE_EXPECT_IDS=1,2 \
//   cargo test --manifest-path tests/iceberg_rust_oracle/Cargo.toml -- --nocapture
//
// Nothing here imports `benostreamdb`: the oracle shares no code with the writer,
// which is the point. If the table only makes sense to BenoStreamDB, this fails.

use anyhow::{Context, Result};
use benostreamdb_iceberg_oracle::{OracleTarget, Suite};
use futures::StreamExt;
use iceberg::io::FileIOBuilder;
use iceberg::spec::FormatVersion;
use iceberg::table::StaticTable;
use iceberg::TableIdent;
use std::path::{Path, PathBuf};

/// Highest-versioned `metadata/v<N>.metadata.json` under the table root.
fn latest_metadata_file(table_root: &Path) -> Result<PathBuf> {
    let metadata_dir = table_root.join("metadata");
    let entries = std::fs::read_dir(&metadata_dir).with_context(|| {
        format!(
            "no `metadata/` directory under {} — is BSDB_ORACLE_URI a table root?",
            table_root.display()
        )
    })?;

    let mut best: Option<(u64, PathBuf)> = None;
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        // Only v<N>.metadata.json; ignore version-hint.text, snap-*.avro, etc.
        let Some(rest) = name.strip_prefix('v') else {
            continue;
        };
        let Some(num) = rest.strip_suffix(".metadata.json") else {
            continue;
        };
        let Ok(version) = num.parse::<u64>() else {
            continue;
        };
        if best.as_ref().is_none_or(|(v, _)| version > *v) {
            best = Some((version, path));
        }
    }

    best.map(|(_, p)| p)
        .context("no v<N>.metadata.json found in metadata/")
}

fn local_path_from_uri(uri: &str) -> Result<PathBuf> {
    let trimmed = uri
        .strip_prefix("file://")
        .or_else(|| uri.strip_prefix("file:"))
        .unwrap_or(uri);
    Ok(PathBuf::from(trimmed))
}

async fn open_table(target: &OracleTarget) -> Result<iceberg::table::Table> {
    let root = local_path_from_uri(&target.uri)?;
    let metadata_path = latest_metadata_file(&root)?;

    let file_io = FileIOBuilder::new(std::sync::Arc::new(iceberg::io::LocalFsStorageFactory)).build();
    let ident = TableIdent::from_strs(["oracle", "table"])?;

    let static_table =
        StaticTable::from_metadata_file(metadata_path.to_str().context("non-UTF-8 path")?, ident, file_io)
            .await
            .context("iceberg-rust could not open the table from its metadata file")?;

    Ok(static_table.into_table())
}

fn expected_format_version(suite: Suite) -> FormatVersion {
    match suite {
        Suite::V2 => FormatVersion::V2,
        Suite::V3 => FormatVersion::V3,
    }
}

#[tokio::test]
async fn upstream_reader_agrees_with_the_writer() -> Result<()> {
    let Ok(target) = OracleTarget::from_env() else {
        // Not configured: this harness is opt-in (it needs network for crates and
        // an externally-produced table). Say so loudly rather than passing silently.
        eprintln!(
            "SKIPPED: set BSDB_ORACLE_URI (and optionally BSDB_ORACLE_SUITE / \
             BSDB_ORACLE_EXPECT_IDS) to run the iceberg-rust oracle"
        );
        return Ok(());
    };

    let table = open_table(&target).await?;
    let metadata = table.metadata();

    // 1. The format version an external reader infers must match the suite.
    assert_eq!(
        metadata.format_version(),
        expected_format_version(target.suite),
        "external reader inferred the wrong format version"
    );

    // 2. V3 row lineage must be visible to the external reader. `_row_id` /
    // `_last_updated_sequence_number` are *metadata* columns and are deliberately
    // absent from `current_schema()`; the lineage is carried by the table-level
    // `next-row-id` and by each snapshot's row range.
    if target.suite == Suite::V3 {
        assert!(
            metadata.next_row_id() > 0,
            "V3 table metadata must carry a non-zero `next-row-id`"
        );
        let snapshot = metadata
            .current_snapshot()
            .context("V3 table has no current snapshot")?;
        assert!(
            snapshot.first_row_id().is_some(),
            "V3 snapshot must carry a row range (`first-row-id` + `added-rows`);              without it an external reader cannot assign stable `_row_id` values"
        );
    }

    // 3. The rows an external engine reads must be exactly the expected set.
    let scan = table.scan().build()?;
    let mut stream = scan.to_arrow().await?;

    let mut ids: Vec<i32> = Vec::new();
    while let Some(batch) = stream.next().await {
        if let Some(col) = batch?.column_by_name("id") {
            let arr = col
                .as_any()
                .downcast_ref::<arrow_array::Int32Array>()
                .context("`id` is not Int32 in the reader's schema")?;
            ids.extend(arr.iter().flatten());
        }
    }
    ids.sort_unstable();

    let mut expected = target.expected_ids.clone();
    expected.sort_unstable();

    assert_eq!(
        ids, expected,
        "upstream reader saw a different row set than the writer intended. A mismatch \
         here means either the manifest/bounds are wrong or a deletion (V3 deletion \
         vector) was not applied by the external reader."
    );

    Ok(())
}
