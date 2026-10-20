// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Table-property keys and validation shared by every connector.
//!
//! Embedding provenance lives on the table (Iceberg table properties), not in a
//! connector-specific side channel, so Python, SQL/Flight SQL, the MCP server,
//! Spark, and Trino all see and enforce the same contract.
//!
//! Model identity is only knowable from the declared property: a vector column
//! does not carry the model that produced it, and re-embedding every row to
//! prove it is not a write-path check. What the engine *can* enforce is:
//! - the declared model/dimension cannot be changed once the table has data
//!   (unless the table had no provenance, i.e. it is being adopted), and
//! - vectors written through the engine have the declared dimension.

use anyhow::{bail, Result};
use arrow::array::{Array, LargeListArray, LargeStringArray, ListArray, StringArray};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use std::collections::HashMap;

/// Table properties recording which embedding space produced the vectors.
pub const EMBEDDING_MODEL_PROP: &str = "benostream.embedding.model";
pub const EMBEDDING_DIM_PROP: &str = "benostream.embedding.dim";
pub const EMBEDDING_PROVIDER_PROP: &str = "benostream.embedding.provider";
pub const EMBEDDING_ENDPOINT_PROP: &str = "benostream.embedding.endpoint";
pub const EMBEDDING_QUERY_PREFIX_PROP: &str = "benostream.embedding.query_prefix";

/// Dimension produced by a known embedding model.
///
/// This is a curated catalogue, not a downloaded model registry: it lets the
/// engine derive `benostream.embedding.dim` from `benostream.embedding.model`
/// for the common local and hosted models without loading anything. Unknown or
/// custom models, and MRL-truncated vectors (e.g. bge-large at 512), still
/// declare the dimension explicitly.
pub fn known_embedding_dim(model: &str) -> Option<usize> {
    let model = model.trim().to_ascii_lowercase();
    let model = model
        .strip_prefix("sentence-transformers/")
        .unwrap_or(&model);
    match model {
        "baai/bge-small-en-v1.5" | "bge-small-en-v1.5" => Some(384),
        "baai/bge-base-en-v1.5" | "bge-base-en-v1.5" => Some(768),
        "baai/bge-large-en-v1.5" | "bge-large-en-v1.5" => Some(1024),
        "baai/bge-m3" | "bge-m3" => Some(1024),
        "all-minilm" | "all-minilm-l6-v2" => Some(384),
        "intfloat/multilingual-e5-small" | "multilingual-e5-small" => Some(384),
        "intfloat/multilingual-e5-base" | "multilingual-e5-base" => Some(768),
        "intfloat/multilingual-e5-large" | "multilingual-e5-large" => Some(1024),
        "nomic-ai/nomic-embed-text-v1.5" | "nomic-embed-text-v1.5" => Some(768),
        "nomic-embed-text" => Some(768),
        "mixedbread-ai/mxbai-embed-large-v1" | "mxbai-embed-large" | "mxbai-embed-large-v1" => {
            Some(1024)
        }
        "jinaai/jina-embeddings-v2-base-code" | "jina-embeddings-v2-base-code" => Some(768),
        "text-embedding-3-small" => Some(1536),
        "text-embedding-3-large" => Some(3072),
        "text-embedding-ada-002" => Some(1536),
        _ => None,
    }
}

/// Fill in `benostream.embedding.dim` from a known model id when the caller did
/// not provide one. Never overwrites an explicit dimension (MRL truncation and
/// custom models are legitimate).
pub(crate) fn normalize_embedding_properties(properties: &mut HashMap<String, String>) {
    if properties.contains_key(EMBEDDING_DIM_PROP) {
        return;
    }
    let Some(model) = properties.get(EMBEDDING_MODEL_PROP) else {
        return;
    };
    if let Some(dim) = known_embedding_dim(model) {
        properties.insert(EMBEDDING_DIM_PROP.to_string(), dim.to_string());
    }
}

/// Reject changes to the declared embedding space once the table has data.
///
/// `proposed` is the complete property map that would be committed. A table
/// with no `benostream.embedding.model` yet is allowed to adopt provenance even
/// when it already holds data: that is the documented repair path for legacy
/// tables. Once a model is declared, changing or removing it (or a declared
/// dimension) requires a new table and a re-ingest.
pub(crate) fn validate_embedding_property_transition(
    current: &HashMap<String, String>,
    proposed: &HashMap<String, String>,
    has_data: bool,
) -> Result<()> {
    if !has_data {
        return Ok(());
    }

    // Legacy tables have no declared model. Adoption is user-asserted and is
    // the only way to make existing vectors participate in the checks.
    let Some(current_model) = current.get(EMBEDDING_MODEL_PROP) else {
        return Ok(());
    };

    if proposed.get(EMBEDDING_MODEL_PROP).map(String::as_str) != Some(current_model.as_str()) {
        bail!(
            "refusing to change `{EMBEDDING_MODEL_PROP}` on a non-empty table: stored vectors were produced by '{current_model}'. \
             Re-ingest into a new table with the new model instead of mutating the property."
        );
    }

    let current_dim = current.get(EMBEDDING_DIM_PROP);
    let proposed_dim = proposed.get(EMBEDDING_DIM_PROP);
    if current_dim != proposed_dim && current_dim.is_some() {
        bail!(
            "refusing to change `{EMBEDDING_DIM_PROP}` on a non-empty table: stored vectors use dimension '{}', requested '{}'. \
             Re-ingest into a new table instead of mutating the property.",
            current_dim.map(String::as_str).unwrap_or("<none>"),
            proposed_dim.map(String::as_str).unwrap_or("<none>"),
        );
    }

    Ok(())
}

/// Validate that vectors in `batches` match the declared
/// `benostream.embedding.dim`.
///
/// This runs in the engine's central write path, so Python, SQL/Flight SQL, the
/// MCP server, and bulk ingest all get the same dimension guard. Tables with no
/// declared dimension are unaffected (legacy behavior), and a `Field` that is
/// not a recognizable vector column is ignored.
///
/// The declared `benostream.embedding.dim` wins. When it is absent, the
/// dimension is derived from a known `benostream.embedding.model` (BGE, E5,
/// Nomic, OpenAI, ...), or from a `FixedSizeList` vector column in the Arrow
/// schema. Unknown/custom models and MRL-truncated vectors must declare the
/// dimension explicitly.
///
/// Default cost is O(fields): only `FixedSizeList` dimensions are checked.
/// `List` and stringified vectors require the opt-in
/// `BSDB_VALIDATE_EMBEDDINGS=strict` mode, which scans every row. Model
/// identity is intentionally never checked: a vector carries no model id.
pub(crate) fn validate_embedding_batches(
    table_uri: &str,
    properties: &HashMap<String, String>,
    batches: &[RecordBatch],
) -> Result<()> {
    validate_embedding_batches_inner(table_uri, properties, batches, value_scan_enabled())
}

fn validate_embedding_batches_inner(
    table_uri: &str,
    properties: &HashMap<String, String>,
    batches: &[RecordBatch],
    scan_values: bool,
) -> Result<()> {
    let declared_dim = properties
        .get(EMBEDDING_DIM_PROP)
        .and_then(|value| value.trim().parse::<usize>().ok());
    let model_dim = properties
        .get(EMBEDDING_MODEL_PROP)
        .and_then(|model| known_embedding_dim(model));
    let schema_dim = batches.iter().find_map(|batch| {
        batch
            .schema()
            .fields()
            .iter()
            .find_map(|field| match field.data_type() {
                DataType::FixedSizeList(child, size)
                    if is_vector_column_name(field.name()) && is_float_type(child.data_type()) =>
                {
                    Some(*size as usize)
                }
                _ => None,
            })
    });
    let Some(expected_dim) = declared_dim.or(model_dim).or(schema_dim) else {
        return Ok(());
    };
    for batch in batches {
        for (index, field) in batch.schema().fields().iter().enumerate() {
            if !is_vector_column_name(field.name()) {
                continue;
            }
            let array = batch.column(index);
            match field.data_type() {
                DataType::FixedSizeList(child, size) if is_float_type(child.data_type()) => {
                    if *size as usize != expected_dim {
                        bail!(
                            "embedding dimension mismatch in '{table_uri}' column '{}': expected {expected_dim}, but the column is a fixed list of {size}",
                            field.name()
                        );
                    }
                }
                DataType::List(child) if is_float_type(child.data_type()) => {
                    if !scan_values {
                        continue;
                    }
                    let list = array.as_any().downcast_ref::<ListArray>().ok_or_else(|| {
                        anyhow::anyhow!("column '{}' is not a ListArray", field.name())
                    })?;
                    validate_list_lengths(
                        table_uri,
                        field.name(),
                        expected_dim,
                        list.len(),
                        |row| list.is_null(row),
                        |row| list.value(row).len(),
                    )?;
                }
                DataType::LargeList(child) if is_float_type(child.data_type()) => {
                    if !scan_values {
                        continue;
                    }
                    let list =
                        array
                            .as_any()
                            .downcast_ref::<LargeListArray>()
                            .ok_or_else(|| {
                                anyhow::anyhow!("column '{}' is not a LargeListArray", field.name())
                            })?;
                    validate_list_lengths(
                        table_uri,
                        field.name(),
                        expected_dim,
                        list.len(),
                        |row| list.is_null(row),
                        |row| list.value(row).len(),
                    )?;
                }
                DataType::Utf8 => {
                    if !scan_values {
                        continue;
                    }
                    let values = array
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or_else(|| {
                            anyhow::anyhow!("column '{}' is not a StringArray", field.name())
                        })?;
                    validate_string_vectors(table_uri, field.name(), expected_dim, values)?;
                }
                DataType::LargeUtf8 => {
                    if !scan_values {
                        continue;
                    }
                    let values = array
                        .as_any()
                        .downcast_ref::<LargeStringArray>()
                        .ok_or_else(|| {
                            anyhow::anyhow!("column '{}' is not a LargeStringArray", field.name())
                        })?;
                    validate_string_vectors(table_uri, field.name(), expected_dim, values)?;
                }
                _ => {}
            }
        }
    }

    Ok(())
}

fn validate_list_lengths<F, G>(
    table_uri: &str,
    column: &str,
    expected_dim: usize,
    len: usize,
    is_null: F,
    value_len: G,
) -> Result<()>
where
    F: Fn(usize) -> bool,
    G: Fn(usize) -> usize,
{
    for row in 0..len {
        if is_null(row) {
            continue;
        }
        let found = value_len(row);
        if found != expected_dim {
            bail!(
                "embedding dimension mismatch in '{table_uri}' column '{column}' row {row}: expected {expected_dim}, found {found}"
            );
        }
    }
    Ok(())
}

fn validate_string_vectors(
    table_uri: &str,
    column: &str,
    expected_dim: usize,
    values: &dyn Array,
) -> Result<()> {
    for row in 0..values.len() {
        if values.is_null(row) {
            continue;
        }
        let value = if let Some(strings) = values.as_any().downcast_ref::<StringArray>() {
            strings.value(row)
        } else if let Some(strings) = values.as_any().downcast_ref::<LargeStringArray>() {
            strings.value(row)
        } else {
            return Ok(());
        };
        let Some(found) = string_vector_dim(value) else {
            // Not a vector literal (plain text stored in a vector-named column).
            continue;
        };
        if found != expected_dim {
            bail!(
                "embedding dimension mismatch in '{table_uri}' column '{column}' row {row}: expected {expected_dim}, found {found}"
            );
        }
    }
    Ok(())
}

/// Cheap pre-check so non-vector ingestion never reads table properties or
/// walks the validation path at all.
pub(crate) fn batches_have_vector_columns(batches: &[RecordBatch]) -> bool {
    batches.iter().any(|batch| {
        batch.schema().fields().iter().any(|field| {
            is_vector_column_name(field.name())
                && matches!(
                    field.data_type(),
                    DataType::FixedSizeList(..)
                        | DataType::List(..)
                        | DataType::LargeList(..)
                        | DataType::Utf8
                        | DataType::LargeUtf8
                )
        })
    })
}

fn is_vector_column_name(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    name == "embedding"
        || name == "embeddings"
        || name == "vector"
        || name == "vectors"
        || name.ends_with("_embedding")
        || name.ends_with("_embeddings")
        || name.ends_with("_vector")
        || name.ends_with("_vectors")
}

fn is_float_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Float16 | DataType::Float32 | DataType::Float64
    )
}

/// Dimension of a stringified vector (`"[0.1, 0.2, ...]"`).
fn string_vector_dim(value: &str) -> Option<usize> {
    let trimmed = value.trim();
    let inner = trimmed.strip_prefix('[')?.strip_suffix(']')?.trim();
    if inner.is_empty() {
        return Some(0);
    }
    Some(
        inner
            .split(',')
            .filter(|part| !part.trim().is_empty())
            .count(),
    )
}

/// Per-row list/string scans are opt-in. The default only checks
/// `FixedSizeList` dimensions, which is O(fields) and catches schema mismatches
/// without touching every value.
fn value_scan_enabled() -> bool {
    let Ok(value) = std::env::var("BSDB_VALIDATE_EMBEDDINGS") else {
        return false;
    };
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "on" | "strict" | "yes"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{FixedSizeListArray, Float32Array, StringArray};
    use arrow::datatypes::Field;

    fn props(model: Option<&str>, dim: Option<&str>) -> HashMap<String, String> {
        let mut props = HashMap::new();
        if let Some(model) = model {
            props.insert(EMBEDDING_MODEL_PROP.to_string(), model.to_string());
        }
        if let Some(dim) = dim {
            props.insert(EMBEDDING_DIM_PROP.to_string(), dim.to_string());
        }
        props
    }

    #[test]
    fn known_models_resolve_to_dimensions() {
        assert_eq!(known_embedding_dim("BAAI/bge-small-en-v1.5"), Some(384));
        assert_eq!(
            known_embedding_dim("sentence-transformers/all-MiniLM-L6-v2"),
            Some(384)
        );
        assert_eq!(known_embedding_dim("BAAI/bge-large-en-v1.5"), Some(1024));
        assert_eq!(known_embedding_dim("all-minilm"), Some(384));
        assert_eq!(known_embedding_dim("mxbai-embed-large"), Some(1024));
        assert_eq!(known_embedding_dim("text-embedding-3-small"), Some(1536));
        assert_eq!(known_embedding_dim("vendor/custom-embed"), None);
    }

    #[test]
    fn dimension_is_filled_from_the_model_without_overwriting_explicit_mrl() {
        let mut derived = props(Some("BAAI/bge-small-en-v1.5"), None);
        normalize_embedding_properties(&mut derived);
        assert_eq!(
            derived.get(EMBEDDING_DIM_PROP).map(String::as_str),
            Some("384")
        );

        // MRL truncation: bge-large at 512 must keep the explicit 512.
        let mut truncated = props(Some("BAAI/bge-large-en-v1.5"), Some("512"));
        normalize_embedding_properties(&mut truncated);
        assert_eq!(
            truncated.get(EMBEDDING_DIM_PROP).map(String::as_str),
            Some("512")
        );
    }

    #[test]
    fn model_dimension_validates_string_vectors_without_a_property() {
        let schema =
            arrow::datatypes::Schema::new(vec![Field::new("embedding", DataType::Utf8, false)]);
        let good_vector = format!(
            "[{}]",
            std::iter::repeat_n("0.0", 384)
                .collect::<Vec<_>>()
                .join(", ")
        );
        let good = RecordBatch::try_new(
            std::sync::Arc::new(schema.clone()),
            vec![std::sync::Arc::new(StringArray::from(vec![good_vector]))],
        )
        .unwrap();
        let bad = RecordBatch::try_new(
            std::sync::Arc::new(schema),
            vec![std::sync::Arc::new(StringArray::from(vec!["[1.0, 2.0]"]))],
        )
        .unwrap();

        let model_only = props(Some("BAAI/bge-small-en-v1.5"), None);
        assert!(validate_embedding_batches_inner("file:///t", &model_only, &[good], true).is_ok());
        assert!(validate_embedding_batches_inner("file:///t", &model_only, &[bad], true).is_err());
    }

    #[test]
    fn property_change_is_rejected_once_data_exists() {
        let current = props(Some("model-a"), Some("3"));
        let proposed = props(Some("model-b"), Some("3"));
        assert!(validate_embedding_property_transition(&current, &proposed, true).is_err());
        assert!(validate_embedding_property_transition(&current, &proposed, false).is_ok());
    }

    #[test]
    fn same_property_is_allowed_and_legacy_can_be_adopted() {
        let current = props(Some("model-a"), Some("3"));
        let same = props(Some("model-a"), Some("3"));
        assert!(validate_embedding_property_transition(&current, &same, true).is_ok());

        let legacy = HashMap::new();
        let adopted = props(Some("model-a"), Some("3"));
        assert!(validate_embedding_property_transition(&legacy, &adopted, true).is_ok());
    }

    #[test]
    fn removing_provenance_is_rejected_once_data_exists() {
        let current = props(Some("model-a"), Some("3"));
        let removed = HashMap::new();
        assert!(validate_embedding_property_transition(&current, &removed, true).is_err());
    }

    #[test]
    fn string_vector_dimension_is_checked() {
        let schema =
            arrow::datatypes::Schema::new(vec![Field::new("embedding", DataType::Utf8, false)]);
        let good = RecordBatch::try_new(
            std::sync::Arc::new(schema.clone()),
            vec![std::sync::Arc::new(StringArray::from(vec![
                "[1.0, 2.0, 3.0]",
                "[4.0, 5.0, 6.0]",
            ]))],
        )
        .unwrap();
        assert!(validate_embedding_batches_inner(
            "file:///t",
            &props(None, Some("3")),
            &[good],
            true
        )
        .is_ok());

        let bad = RecordBatch::try_new(
            std::sync::Arc::new(schema),
            vec![std::sync::Arc::new(StringArray::from(vec!["[1.0, 2.0]"]))],
        )
        .unwrap();
        assert!(validate_embedding_batches_inner(
            "file:///t",
            &props(None, Some("3")),
            &[bad],
            true
        )
        .is_err());
    }

    #[test]
    fn schema_dimension_validates_other_vector_columns_without_a_property() {
        let list = FixedSizeListArray::new(
            std::sync::Arc::new(Field::new("item", DataType::Float32, false)),
            3,
            std::sync::Arc::new(Float32Array::from(vec![1.0, 2.0, 3.0])),
            None,
        );
        let schema = arrow::datatypes::Schema::new(vec![
            Field::new(
                "embedding",
                DataType::FixedSizeList(
                    std::sync::Arc::new(Field::new("item", DataType::Float32, false)),
                    3,
                ),
                false,
            ),
            Field::new("vector", DataType::Utf8, false),
        ]);
        let batch = RecordBatch::try_new(
            std::sync::Arc::new(schema),
            vec![
                std::sync::Arc::new(list),
                std::sync::Arc::new(StringArray::from(vec!["[1.0, 2.0]"])),
            ],
        )
        .unwrap();

        // No `benostream.embedding.dim` property: the fixed-size column's
        // dimension (3) still catches the 2-d string vector.
        assert!(
            validate_embedding_batches_inner("file:///t", &HashMap::new(), &[batch], true).is_err()
        );
    }

    #[test]
    fn fixed_size_list_dimension_is_checked() {
        let values = Float32Array::from(vec![1.0, 2.0, 3.0]);
        let array = FixedSizeListArray::new(
            std::sync::Arc::new(Field::new("item", DataType::Float32, false)),
            3,
            std::sync::Arc::new(values),
            None,
        );
        let schema = arrow::datatypes::Schema::new(vec![Field::new(
            "embedding",
            DataType::FixedSizeList(
                std::sync::Arc::new(Field::new("item", DataType::Float32, false)),
                3,
            ),
            false,
        )]);
        let batch = RecordBatch::try_new(
            std::sync::Arc::new(schema),
            vec![std::sync::Arc::new(array)],
        )
        .unwrap();
        assert!(validate_embedding_batches("file:///t", &props(None, Some("3")), &[batch]).is_ok());

        let values = Float32Array::from(vec![1.0, 2.0]);
        let array = FixedSizeListArray::new(
            std::sync::Arc::new(Field::new("item", DataType::Float32, false)),
            2,
            std::sync::Arc::new(values),
            None,
        );
        let schema = arrow::datatypes::Schema::new(vec![Field::new(
            "embedding",
            DataType::FixedSizeList(
                std::sync::Arc::new(Field::new("item", DataType::Float32, false)),
                2,
            ),
            false,
        )]);
        let batch = RecordBatch::try_new(
            std::sync::Arc::new(schema),
            vec![std::sync::Arc::new(array)],
        )
        .unwrap();
        assert!(
            validate_embedding_batches("file:///t", &props(None, Some("3")), &[batch]).is_err()
        );
    }
}
