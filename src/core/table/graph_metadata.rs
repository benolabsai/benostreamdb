// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Table-type metadata: `table_type` (`node` / `edge` / `table`) and the
//! endpoint column names, stored in the table's `Manifest.properties`.
//!
//! This is the declarative edge/node-table convention. It lets the graph
//! functions and the connectors resolve a table's endpoints from **metadata**
//! rather than guessing from column names, and it is what an MCP agent uses to
//! discover the graph (`Session.list_graph_tables`).
//!
//! Property keys (all optional; absent means "auto-detect"):
//!   * `table_type`   — `node` | `edge` | `table` (default `table`)
//!   * `src_col`      — edge source column
//!   * `dst_col`      — edge target column
//!   * `relation_col` — edge relation/predicate column
//!   * `weight_col`   — edge weight column
//!   * `id_col`       — node id column
//!   * `label_col`    — node label/name column

use std::collections::HashMap;

use anyhow::Result;
use arrow::datatypes::Schema;

use super::Table;
use crate::core::manifest::IndexAlgorithm;

pub const PROP_TABLE_TYPE: &str = "table_type";
pub const PROP_SRC_COL: &str = "src_col";
pub const PROP_DST_COL: &str = "dst_col";
pub const PROP_RELATION_COL: &str = "relation_col";
pub const PROP_WEIGHT_COL: &str = "weight_col";
pub const PROP_ID_COL: &str = "id_col";
pub const PROP_LABEL_COL: &str = "label_col";

/// The role a table plays in the graph model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TableType {
    /// A node/entity table (`id_col`, optional `label_col`).
    Node,
    /// An edge/relationship table (`src_col`, `dst_col`, optional relation/weight).
    Edge,
    /// A plain table (the default).
    #[default]
    Table,
}

impl TableType {
    pub fn as_str(self) -> &'static str {
        match self {
            TableType::Node => "node",
            TableType::Edge => "edge",
            TableType::Table => "table",
        }
    }

    /// Parse a `table_type` value; unknown values fall back to `Table`.
    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "node" | "nodes" | "vertex" | "vertices" => TableType::Node,
            "edge" | "edges" | "relationship" | "relationships" => TableType::Edge,
            _ => TableType::Table,
        }
    }
}

/// Resolved graph metadata for a table.
#[derive(Debug, Clone, Default)]
pub struct GraphMetadata {
    pub table_type: TableType,
    pub source_column: Option<String>,
    pub target_column: Option<String>,
    pub relation_column: Option<String>,
    pub weight_column: Option<String>,
    pub id_column: Option<String>,
    pub label_column: Option<String>,
}

impl GraphMetadata {
    /// Build from a table's properties.
    pub fn from_properties(props: &HashMap<String, String>) -> Self {
        let get = |k: &str| props.get(k).filter(|v| !v.is_empty()).cloned();
        GraphMetadata {
            table_type: props
                .get(PROP_TABLE_TYPE)
                .map(|v| TableType::parse(v))
                .unwrap_or_default(),
            source_column: get(PROP_SRC_COL),
            target_column: get(PROP_DST_COL),
            relation_column: get(PROP_RELATION_COL),
            weight_column: get(PROP_WEIGHT_COL),
            id_column: get(PROP_ID_COL),
            label_column: get(PROP_LABEL_COL),
        }
    }

    /// True when the table is declared as an edge table.
    pub fn is_edge(&self) -> bool {
        self.table_type == TableType::Edge
    }

    /// True when the table is declared as a node table.
    pub fn is_node(&self) -> bool {
        self.table_type == TableType::Node
    }
}

/// The standard edge-endpoint column candidates (name-based fallback).
pub const SOURCE_COLUMN_CANDIDATES: [&str; 5] = ["source", "src", "src_id", "from", "u"];
pub const TARGET_COLUMN_CANDIDATES: [&str; 5] = ["target", "dst", "dst_id", "to", "v"];

/// First candidate present in the schema, else `default`.
fn auto_detect(schema: &Schema, candidates: &[&str], default: &str) -> String {
    candidates
        .iter()
        .find(|c| schema.column_with_name(c).is_some())
        .map(|s| s.to_string())
        .unwrap_or_else(|| default.to_string())
}

impl Table {
    /// The table's declared type (`node` / `edge` / `table`).
    pub async fn table_type_async(&self) -> Result<TableType> {
        Ok(self.graph_metadata_async().await?.table_type)
    }

    /// The table's declared type (synchronous).
    pub fn table_type(&self) -> Result<TableType> {
        self.runtime().block_on(self.table_type_async())
    }

    /// The table's resolved graph metadata.
    pub async fn graph_metadata_async(&self) -> Result<GraphMetadata> {
        Ok(GraphMetadata::from_properties(
            &self.properties_async().await?,
        ))
    }

    /// The table's resolved graph metadata (synchronous).
    pub fn graph_metadata(&self) -> Result<GraphMetadata> {
        self.runtime().block_on(self.graph_metadata_async())
    }

    /// Resolve the `(source, target)` endpoint columns for this table.
    ///
    /// Precedence: explicit arguments → `src_col`/`dst_col` metadata → the
    /// standard name candidates (`source`/`src`/… , `target`/`dst`/…).
    pub async fn resolve_endpoint_columns(
        &self,
        source_column: Option<&str>,
        target_column: Option<&str>,
    ) -> (String, String) {
        let meta = self.graph_metadata_async().await.unwrap_or_default();
        let schema = self.arrow_schema();
        let src = source_column
            .map(String::from)
            .or(meta.source_column)
            .unwrap_or_else(|| auto_detect(&schema, &SOURCE_COLUMN_CANDIDATES, "source"));
        let tgt = target_column
            .map(String::from)
            .or(meta.target_column)
            .unwrap_or_else(|| auto_detect(&schema, &TARGET_COLUMN_CANDIDATES, "target"));
        (src, tgt)
    }

    /// The `(source, target)` endpoint columns, metadata-first (synchronous).
    pub fn edge_endpoints(&self) -> Result<(String, String)> {
        Ok(self
            .runtime()
            .block_on(self.resolve_endpoint_columns(None, None)))
    }

    /// Ensure the forward and reverse CSR graph indexes implied by an edge-table
    /// declaration exist.
    ///
    /// Declaring `table_type = 'edge'` with `src_col`/`dst_col` (via
    /// `SET TBLPROPERTIES` or Python `set_property`/`set_properties`) is the
    /// declarative form of "this is a graph"; this materialises the two overlays
    /// the traversal fast paths need:
    ///   * **forward** CSR keyed on the source column (`src → dst`), and
    ///   * **reverse** CSR keyed on the target column (`dst → src`).
    ///
    /// Idempotent: an index already present (matched by its `src_column` /
    /// `dst_column`) is left untouched, so re-declaring the metadata is cheap.
    /// Returns the number of indexes added. A no-op (returns `0`) for non-edge
    /// tables, for a self-loop declaration (`src == dst`), or when the endpoint
    /// columns are not in the schema yet.
    pub async fn ensure_edge_indexes_async(&self) -> Result<usize> {
        let meta = self.graph_metadata_async().await?;
        if !meta.is_edge() {
            return Ok(0);
        }
        let (src, dst) = self.resolve_endpoint_columns(None, None).await;
        if src == dst {
            return Ok(0);
        }
        let schema = self.arrow_schema();
        if schema.column_with_name(&src).is_none() || schema.column_with_name(&dst).is_none() {
            return Ok(0);
        }

        // The manifest schema is the source of truth for configured indexes.
        let manifest = self.manifest().await?;
        let has_csr = |col: &str, s: &str, d: &str| -> bool {
            manifest
                .schemas
                .last()
                .and_then(|sch| sch.fields.iter().find(|f| f.name == col))
                .map(|f| {
                    f.indexes.iter().any(|ix| {
                        matches!(
                            ix,
                            IndexAlgorithm::CsrGraph { src_column, dst_column }
                                if src_column == s && dst_column == d
                        )
                    })
                })
                .unwrap_or(false)
        };

        let mut added = 0;
        if !has_csr(&src, &src, &dst) {
            self.add_index(
                src.clone(),
                IndexAlgorithm::CsrGraph {
                    src_column: src.clone(),
                    dst_column: dst.clone(),
                },
            )
            .await?;
            added += 1;
        }
        if !has_csr(&dst, &dst, &src) {
            self.add_index(
                dst.clone(),
                IndexAlgorithm::CsrGraph {
                    src_column: dst.clone(),
                    dst_column: src.clone(),
                },
            )
            .await?;
            added += 1;
        }
        if added > 0 {
            tracing::info!(
                table = %self.uri,
                "edge declaration materialised {} CSR graph index(es) (forward '{}', reverse '{}')",
                added,
                src,
                dst
            );
        }
        Ok(added)
    }

    /// Synchronous wrapper for [`Table::ensure_edge_indexes_async`].
    pub fn ensure_edge_indexes(&self) -> Result<usize> {
        self.runtime().block_on(self.ensure_edge_indexes_async())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_type_parses() {
        assert_eq!(TableType::parse("edge"), TableType::Edge);
        assert_eq!(TableType::parse("EDGES"), TableType::Edge);
        assert_eq!(TableType::parse("nodes"), TableType::Node);
        assert_eq!(TableType::parse("vertex"), TableType::Node);
        assert_eq!(TableType::parse("garbage"), TableType::Table);
        assert_eq!(TableType::parse(""), TableType::Table);
    }

    #[test]
    fn metadata_from_properties() {
        let mut p = HashMap::new();
        p.insert(PROP_TABLE_TYPE.to_string(), "edge".to_string());
        p.insert(PROP_SRC_COL.to_string(), "u".to_string());
        p.insert(PROP_DST_COL.to_string(), "v".to_string());
        let m = GraphMetadata::from_properties(&p);
        assert!(m.is_edge());
        assert!(!m.is_node());
        assert_eq!(m.source_column.as_deref(), Some("u"));
        assert_eq!(m.target_column.as_deref(), Some("v"));
    }

    #[test]
    fn empty_properties_default_to_plain_table() {
        let m = GraphMetadata::from_properties(&HashMap::new());
        assert_eq!(m.table_type, TableType::Table);
        assert!(!m.is_edge());
        assert!(!m.is_node());
        assert!(m.source_column.is_none());
    }

    #[test]
    fn blank_values_are_treated_as_absent() {
        let mut p = HashMap::new();
        p.insert(PROP_SRC_COL.to_string(), String::new());
        let m = GraphMetadata::from_properties(&p);
        assert!(m.source_column.is_none());
    }

    #[tokio::test]
    async fn edge_declaration_materialises_forward_and_reverse_csr() {
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let dir = std::env::temp_dir().join(format!("bsdb_edge_idx_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let schema = Arc::new(Schema::new(vec![
            Field::new("source", DataType::UInt64, false),
            Field::new("target", DataType::UInt64, false),
        ]));
        let table = Table::create_async(dir.to_str().unwrap().to_string(), schema)
            .await
            .unwrap();

        // A plain table is a no-op.
        assert_eq!(table.ensure_edge_indexes_async().await.unwrap(), 0);

        let mut props = HashMap::new();
        props.insert(PROP_TABLE_TYPE.to_string(), "edge".to_string());
        props.insert(PROP_SRC_COL.to_string(), "source".to_string());
        props.insert(PROP_DST_COL.to_string(), "target".to_string());
        table.set_properties_async(props).await.unwrap();

        // Forward (source) + reverse (target) CSR indexes.
        assert_eq!(table.ensure_edge_indexes_async().await.unwrap(), 2);
        // Idempotent: re-declaring adds nothing.
        assert_eq!(table.ensure_edge_indexes_async().await.unwrap(), 0);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
