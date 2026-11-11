// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
//
// Independent Iceberg reader oracle for BenoStreamDB conformance testing.
//
// This crate deliberately does **not** depend on `benostreamdb`. It opens a
// table tree with the upstream `apache/iceberg-rust` crate and asserts what an
// external engine must be able to see: exact row values, schema field ids,
// bounds, `format-version`, and the row-lineage columns. Sharing no code with
// the writer is the entire point — a reader built from the writer's own
// assumptions proves nothing.
//
// It lives in its own workspace root because `iceberg` pins different `arrow` /
// `object_store` versions than the engine.

use anyhow::{Context, Result};

/// Which verification pass to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Suite {
    /// V2 table written by BenoStreamDB: rows, field ids, bounds.
    V2,
    /// V3 table: row lineage (`first_row_id` / `next_row_id`) plus surviving
    /// rows after a deletion vector.
    V3,
}

impl Suite {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "v2" => Ok(Suite::V2),
            "v3" => Ok(Suite::V3),
            other => anyhow::bail!("unknown oracle suite `{other}` (expected `v2` or `v3`)"),
        }
    }
}

/// The table the oracle should verify.
///
/// Supplied through environment variables so the harness depends only on an
/// on-disk (or object-store) artifact, never on BenoStreamDB's types:
///
/// * `BSDB_ORACLE_URI`          — table location, e.g. `file:///tmp/tbl`
/// * `BSDB_ORACLE_SUITE`        — `v2` or `v3` (default `v2`)
/// * `BSDB_ORACLE_EXPECT_IDS`   — comma-separated `id` values that must survive,
///                                in ascending order
pub struct OracleTarget {
    pub uri: String,
    pub suite: Suite,
    pub expected_ids: Vec<i32>,
}

impl OracleTarget {
    pub fn from_env() -> Result<Self> {
        let uri = std::env::var("BSDB_ORACLE_URI").context(
            "BSDB_ORACLE_URI is not set; point it at a BenoStreamDB table to verify \
             (see tests/iceberg_rust_oracle/README.md)",
        )?;
        let suite = Suite::parse(&std::env::var("BSDB_ORACLE_SUITE").unwrap_or_else(|_| "v2".into()))?;
        let expected_ids = std::env::var("BSDB_ORACLE_EXPECT_IDS")
            .unwrap_or_default()
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().parse::<i32>())
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("BSDB_ORACLE_EXPECT_IDS must be comma-separated integers")?;
        Ok(Self {
            uri,
            suite,
            expected_ids,
        })
    }
}
