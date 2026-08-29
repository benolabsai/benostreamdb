// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Ingestion-time schema and array inference for the search API.
//!
//! Maps OpenSearch / Elasticsearch 7.10 dynamic-mapping rules onto Arrow
//! types: JSON booleans become `Boolean`, integral numbers become `Int64`
//! (ES `long`), other numbers become `Float64` (ES `double`), strings
//! become `Utf8` (ES `text`), objects become nested `Struct`s with keys in
//! alphabetical order, and uniform numeric vectors become fixed-size
//! `Float32` lists. [`merge_datatypes`] / [`merge_schemas`] evolve an
//! existing schema as new documents arrive, and [`value_to_array`]
//! materializes a column of JSON values into the inferred Arrow type.

use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, BooleanArray, FixedSizeListArray, Float32Array, Float64Array, Int64Array, NullArray,
    StringArray, StructArray,
};
use arrow::buffer::NullBuffer;
use arrow::datatypes::{DataType, Field, FieldRef, Fields, Schema, SchemaRef};
use serde_json::Value;

/// Errors that can occur while inferring Arrow types from JSON or merging
/// schemas.
#[derive(Debug)]
pub enum InferError {
    /// The top-level JSON value is not an object, so no schema can be
    /// inferred for it.
    NotAnObject,
    /// Two values of the same field have incompatible kinds.
    Conflict {
        /// The field name; empty when the conflict is raised directly by
        /// [`merge_datatypes`] (the caller rewrites it).
        field: String,
        /// The ES name of the existing kind, e.g. `"long"`.
        existing: String,
        /// The ES name of the incoming kind, e.g. `"text"`.
        incoming: String,
    },
    /// A value cannot be represented in the target Arrow type.
    Unsupported {
        /// The field name; may be empty for internal failures.
        field: String,
        /// Human-readable explanation of the problem.
        reason: String,
    },
}

impl fmt::Display for InferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotAnObject => write!(f, "value is not a JSON object"),
            Self::Conflict {
                field,
                existing,
                incoming,
            } => {
                if field.is_empty() {
                    write!(f, "type conflict: existing {existing}, incoming {incoming}")
                } else {
                    write!(
                        f,
                        "field '{field}': type conflict: existing {existing}, incoming {incoming}"
                    )
                }
            }
            Self::Unsupported { field, reason } => {
                if field.is_empty() {
                    write!(f, "unsupported value: {reason}")
                } else {
                    write!(f, "field '{field}': {reason}")
                }
            }
        }
    }
}

impl std::error::Error for InferError {}

/// Coarse kind of a JSON value, mirroring ES dynamic mapping types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum JsonKind {
    Bool,
    Int,
    Float,
    Str,
    Obj,
    Arr,
}

/// Classify a JSON value; integral numbers are `Int`, all other numbers
/// are `Float`.
fn json_kind(v: &Value) -> JsonKind {
    match v {
        Value::Bool(_) => JsonKind::Bool,
        Value::Number(n) if n.is_i64() || n.is_u64() => JsonKind::Int,
        Value::Number(_) => JsonKind::Float,
        Value::String(_) => JsonKind::Str,
        Value::Object(_) => JsonKind::Obj,
        Value::Array(_) => JsonKind::Arr,
        // Callers filter JSON nulls before classifying; treat a leaked
        // null as a string so this function is total.
        Value::Null => JsonKind::Str,
    }
}

/// ES 7.10 dynamic mapping name for a JSON kind.
fn es_name(kind: JsonKind) -> &'static str {
    match kind {
        JsonKind::Bool => "boolean",
        JsonKind::Int => "long",
        JsonKind::Float => "double",
        JsonKind::Str => "text",
        JsonKind::Obj => "object",
        JsonKind::Arr => "array",
    }
}

/// Whether two JSON kinds can share a single column.
fn kinds_compatible(a: JsonKind, b: JsonKind) -> bool {
    a == b
        || matches!(
            (a, b),
            (JsonKind::Int, JsonKind::Float) | (JsonKind::Float, JsonKind::Int)
        )
}

/// Infer the Arrow data type for a field from the JSON values observed
/// across a batch of documents.
///
/// `None` entries and JSON `null`s carry no type information and are
/// ignored; if nothing remains, `Ok(None)` is returned and callers should
/// default to [`DataType::Utf8`].
///
/// Arrays become `FixedSizeList(Float32, dim)` when every non-empty array
/// has the same length of at least 2 and numeric elements only. Objects
/// become a `Struct` whose keys are the union across values in
/// alphabetical order, recursing per key. Scalars promote `Int64` +
/// `Float64` to `Float64`; any other mix is a [`InferError::Conflict`].
pub fn infer_datatype(
    field: &str,
    values: &[Option<&Value>],
) -> Result<Option<DataType>, InferError> {
    let values: Vec<&Value> = values
        .iter()
        .filter_map(|v| match v {
            Some(v) if !v.is_null() => Some(*v),
            _ => None,
        })
        .collect();

    if values.is_empty() {
        return Ok(None);
    }

    if values.iter().all(|v| v.is_array()) {
        return infer_vector(field, &values);
    }

    if values.iter().all(|v| v.is_object()) {
        let keys: BTreeSet<&str> = values
            .iter()
            .filter_map(|v| v.as_object())
            .flat_map(|o| o.keys())
            .map(|k| k.as_str())
            .collect();
        let mut fields: Vec<Field> = Vec::with_capacity(keys.len());
        for k in keys {
            let child: Vec<Option<&Value>> = values
                .iter()
                .map(|v| v.as_object().and_then(|o| o.get(k)))
                .collect();
            let dt = infer_datatype(k, &child)?.unwrap_or(DataType::Utf8);
            fields.push(Field::new(k, dt, true));
        }
        return Ok(Some(DataType::Struct(Fields::from(fields))));
    }

    // Scalar values (or any non-uniform mix of kinds).
    let mut kinds: Vec<JsonKind> = Vec::new();
    for v in &values {
        let k = json_kind(v);
        if !kinds.contains(&k) {
            kinds.push(k);
        }
    }
    match kinds.as_slice() {
        [JsonKind::Bool] => Ok(Some(DataType::Boolean)),
        [JsonKind::Int] => Ok(Some(DataType::Int64)),
        [JsonKind::Float] => Ok(Some(DataType::Float64)),
        [JsonKind::Str] => Ok(Some(DataType::Utf8)),
        [JsonKind::Int, JsonKind::Float] | [JsonKind::Float, JsonKind::Int] => {
            Ok(Some(DataType::Float64))
        }
        _ => {
            // Distinct kinds in first-seen order: the first kind is the
            // "existing" one, the first kind incompatible with it is the
            // "incoming" one (Int/Float is the only cross-kind merge and
            // was handled above).
            let existing = kinds[0];
            let incoming = kinds
                .iter()
                .skip(1)
                .find(|k| !kinds_compatible(existing, **k))
                .expect("a non-compatible kind exists in this branch");
            Err(InferError::Conflict {
                field: field.to_string(),
                existing: es_name(existing).to_string(),
                incoming: es_name(*incoming).to_string(),
            })
        }
    }
}

/// Infer a fixed-size vector type from a slice whose values are all
/// arrays.
fn infer_vector(field: &str, values: &[&Value]) -> Result<Option<DataType>, InferError> {
    let non_empty: Vec<&Vec<Value>> = values
        .iter()
        .filter_map(|v| v.as_array())
        .filter(|a| !a.is_empty())
        .collect();

    if non_empty.is_empty() {
        return Ok(None);
    }

    let dim = non_empty[0].len();
    if non_empty.iter().any(|a| a.len() != dim) {
        return Err(InferError::Unsupported {
            field: field.to_string(),
            reason: "ragged vector lengths".to_string(),
        });
    }
    if dim < 2 {
        return Err(InferError::Unsupported {
            field: field.to_string(),
            reason: format!("vector length {dim} < 2"),
        });
    }
    for a in &non_empty {
        if a.iter().any(|e| !e.is_number()) {
            return Err(InferError::Unsupported {
                field: field.to_string(),
                reason: "non-numeric vector element".to_string(),
            });
        }
    }

    Ok(Some(DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32,
    )))
}

/// Infer an Arrow schema from a single JSON document.
///
/// Every top-level key becomes a nullable field in alphabetical order
/// (serde_json without `preserve_order` stores objects as sorted maps).
/// A non-object document is an [`InferError::NotAnObject`].
pub fn infer_schema(doc: &Value) -> Result<SchemaRef, InferError> {
    let obj = doc.as_object().ok_or(InferError::NotAnObject)?;
    let mut fields: Vec<Field> = Vec::with_capacity(obj.len());
    for (k, v) in obj {
        let dt = infer_datatype(k, &[Some(v)])?.unwrap_or(DataType::Utf8);
        fields.push(Field::new(k, dt, true));
    }
    Ok(Arc::new(Schema::new(fields)))
}

/// Merge the Arrow types of a single field seen in two schemas.
///
/// Identical types are returned unchanged; `Int64` and `Float64` promote
/// to `Float64`; fixed-size lists must agree on dimension (the conflict
/// reports an empty field name so [`merge_schemas`] can rewrite it);
/// structs union their children (base children keep their order, then
/// incoming-only children in incoming order). Anything else conflicts.
pub fn merge_datatypes(base: &DataType, incoming: &DataType) -> Result<DataType, InferError> {
    if base == incoming {
        return Ok(base.clone());
    }
    match (base, incoming) {
        (DataType::Int64, DataType::Float64) | (DataType::Float64, DataType::Int64) => {
            Ok(DataType::Float64)
        }
        (DataType::FixedSizeList(b_item, b_dim), DataType::FixedSizeList(i_item, i_dim))
            if matches!(b_item.data_type(), DataType::Float32)
                && matches!(i_item.data_type(), DataType::Float32) =>
        {
            if b_dim == i_dim {
                Ok(base.clone())
            } else {
                Err(InferError::Conflict {
                    field: String::new(),
                    existing: base.to_string(),
                    incoming: incoming.to_string(),
                })
            }
        }
        (DataType::Struct(b_fields), DataType::Struct(i_fields)) => {
            let mut merged: Vec<FieldRef> = b_fields.iter().cloned().collect();
            for f in i_fields.iter() {
                match merged.iter_mut().find(|m| m.name() == f.name()) {
                    Some(slot) => {
                        let dt = merge_datatypes(slot.data_type(), f.data_type())?;
                        *slot = Arc::new(Field::new(f.name(), dt, true));
                    }
                    None => merged.push(f.clone()),
                }
            }
            Ok(DataType::Struct(Fields::from(merged)))
        }
        _ => Err(InferError::Conflict {
            field: String::new(),
            existing: base.to_string(),
            incoming: incoming.to_string(),
        }),
    }
}

/// Merge two schemas into one.
///
/// Base fields keep their order; incoming-only fields are appended in
/// incoming order; shared fields are merged with [`merge_datatypes`]. A
/// conflict raised from inside a merged field (empty field name) is
/// rewritten to name the actual column.
pub fn merge_schemas(base: &Schema, incoming: &Schema) -> Result<Schema, InferError> {
    let mut fields: Vec<FieldRef> = base.fields().iter().cloned().collect();
    for f in incoming.fields().iter() {
        match fields.iter_mut().find(|m| m.name() == f.name()) {
            Some(slot) => {
                let dt = merge_datatypes(slot.data_type(), f.data_type())
                    .map_err(|err| rewrite_field(err, f.name()))?;
                *slot = Arc::new(Field::new(f.name(), dt, true));
            }
            None => fields.push(f.clone()),
        }
    }
    Ok(Schema::new(fields))
}

/// Re-point a field-less [`InferError::Conflict`] at the column being
/// merged.
fn rewrite_field(mut err: InferError, field: &str) -> InferError {
    if let InferError::Conflict { field: f, .. } = &mut err {
        if f.is_empty() {
            *f = field.to_string();
        }
    }
    err
}

/// Materialize a batch of JSON values (one per row) into an Arrow array
/// of `datatype`.
///
/// `None` entries (and JSON nulls) become nulls in the resulting array.
/// For `Struct` and `FixedSizeList` inputs the field references are
/// reused from `datatype` so the result stays `DataType`-equal to it,
/// which is what `RecordBatch::try_new` checks.
pub fn value_to_array(
    field_name: &str,
    datatype: &DataType,
    values: &[Option<Value>],
) -> Result<ArrayRef, InferError> {
    let validity: Vec<bool> = values.iter().map(|v| v.is_some()).collect();
    let nulls = if validity.contains(&false) {
        Some(NullBuffer::from(validity))
    } else {
        None
    };

    match datatype {
        DataType::Utf8 => {
            for v in values.iter().flatten() {
                if !v.is_string() {
                    return Err(InferError::Unsupported {
                        field: field_name.to_string(),
                        reason: format!("expected string, got {}", es_name(json_kind(v))),
                    });
                }
            }
            let strs: Vec<Option<&str>> = values
                .iter()
                .map(|v| v.as_ref().and_then(Value::as_str))
                .collect();
            Ok(Arc::new(StringArray::from(strs)))
        }
        DataType::Int64 => {
            for v in values.iter().flatten() {
                if !v.is_number() {
                    return Err(InferError::Unsupported {
                        field: field_name.to_string(),
                        reason: format!("expected number, got {}", es_name(json_kind(v))),
                    });
                }
            }
            let vals: Vec<i64> = values
                .iter()
                .map(|v| match v.as_ref().and_then(Value::as_number) {
                    Some(n) => n
                        .as_i64()
                        .or_else(|| n.as_u64().map(|u| u as i64))
                        .unwrap_or(0),
                    None => 0,
                })
                .collect();
            Ok(Arc::new(Int64Array::from_iter_values_with_nulls(
                vals.iter().copied(),
                nulls,
            )))
        }
        DataType::Float64 => {
            for v in values.iter().flatten() {
                if !v.is_number() {
                    return Err(InferError::Unsupported {
                        field: field_name.to_string(),
                        reason: format!("expected number, got {}", es_name(json_kind(v))),
                    });
                }
            }
            let vals: Vec<f64> = values
                .iter()
                .map(|v| match v.as_ref().and_then(Value::as_number) {
                    Some(n) => n.as_f64().unwrap_or(0.0),
                    None => 0.0,
                })
                .collect();
            Ok(Arc::new(Float64Array::from_iter_values_with_nulls(
                vals.iter().copied(),
                nulls,
            )))
        }
        DataType::Boolean => {
            for v in values.iter().flatten() {
                if !v.is_boolean() {
                    return Err(InferError::Unsupported {
                        field: field_name.to_string(),
                        reason: format!("expected boolean, got {}", es_name(json_kind(v))),
                    });
                }
            }
            let vals: Vec<Option<bool>> = values
                .iter()
                .map(|v| v.as_ref().and_then(Value::as_bool))
                .collect();
            Ok(Arc::new(BooleanArray::from(vals)))
        }
        DataType::Null => Ok(Arc::new(NullArray::new(values.len()))),
        DataType::Struct(fields) => {
            let mut arrays: Vec<ArrayRef> = Vec::with_capacity(fields.len());
            for child in fields.iter() {
                let child_values: Vec<Option<Value>> = values
                    .iter()
                    .map(|v| match v {
                        Some(obj) => {
                            if !obj.is_object() {
                                return Err(InferError::Unsupported {
                                    field: field_name.to_string(),
                                    reason: format!(
                                        "expected object, got {}",
                                        es_name(json_kind(obj))
                                    ),
                                });
                            }
                            Ok(obj.as_object().and_then(|o| o.get(child.name())).cloned())
                        }
                        None => Ok(None),
                    })
                    .collect::<Result<Vec<Option<Value>>, InferError>>()?;
                arrays.push(value_to_array(
                    child.name(),
                    child.data_type(),
                    &child_values,
                )?);
            }
            let arr = StructArray::try_new(fields.clone(), arrays, nulls).map_err(|e| {
                InferError::Unsupported {
                    field: field_name.to_string(),
                    reason: format!("failed to build struct array: {e}"),
                }
            })?;
            Ok(Arc::new(arr))
        }
        DataType::FixedSizeList(item, dim) if matches!(item.data_type(), DataType::Float32) => {
            let dim = (*dim) as usize;
            let mut flat: Vec<f32> = Vec::with_capacity(values.len() * dim);
            for v in values {
                match v {
                    None => {
                        flat.extend(std::iter::repeat_n(0.0, dim));
                    }
                    Some(row) => {
                        let row = row.as_array().ok_or_else(|| InferError::Unsupported {
                            field: field_name.to_string(),
                            reason: format!("expected vector, got {}", es_name(json_kind(row))),
                        })?;
                        if row.len() != dim {
                            return Err(InferError::Unsupported {
                                field: field_name.to_string(),
                                reason: format!("vector length {}, expected {}", row.len(), dim),
                            });
                        }
                        for e in row {
                            let e = e.as_number().ok_or_else(|| InferError::Unsupported {
                                field: field_name.to_string(),
                                reason: format!(
                                    "non-numeric vector element (got {})",
                                    es_name(json_kind(e))
                                ),
                            })?;
                            flat.push(e.as_f64().unwrap_or(0.0) as f32);
                        }
                    }
                }
            }
            let fsl = FixedSizeListArray::try_new(
                item.clone(),
                dim as i32,
                Arc::new(Float32Array::from(flat)),
                nulls,
            )
            .map_err(|e| InferError::Unsupported {
                field: field_name.to_string(),
                reason: format!("failed to build fixed-size list: {e}"),
            })?;
            Ok(Arc::new(fsl))
        }
        _ => Err(InferError::Unsupported {
            field: field_name.to_string(),
            reason: format!("unsupported data type: {datatype}"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Array;
    use arrow::record_batch::RecordBatch;
    use serde_json::json;

    #[test]
    fn infer_scalar_kinds() {
        let b = json!(true);
        let i = json!(42);
        let f = json!(2.5);
        let s = json!("abc");
        assert_eq!(
            infer_datatype("flag", &[Some(&b)]).unwrap(),
            Some(DataType::Boolean)
        );
        assert_eq!(
            infer_datatype("n", &[Some(&i)]).unwrap(),
            Some(DataType::Int64)
        );
        assert_eq!(
            infer_datatype("r", &[Some(&f)]).unwrap(),
            Some(DataType::Float64)
        );
        assert_eq!(
            infer_datatype("s", &[Some(&s)]).unwrap(),
            Some(DataType::Utf8)
        );
        // A string `_id` infers as text/Utf8 like any other string.
        assert_eq!(
            infer_datatype("_id", &[Some(&json!("id-123"))]).unwrap(),
            Some(DataType::Utf8)
        );
    }

    #[test]
    fn infer_int_float_mix_promotes_to_float64() {
        let i = json!(1);
        let f = json!(2.5);
        assert_eq!(
            infer_datatype("x", &[Some(&i), Some(&f)]).unwrap(),
            Some(DataType::Float64)
        );
        assert_eq!(
            infer_datatype("x", &[Some(&f), Some(&i)]).unwrap(),
            Some(DataType::Float64)
        );
    }

    #[test]
    fn infer_string_int_conflict_reports_field() {
        let s = json!("a");
        let i = json!(1);
        match infer_datatype("col", &[Some(&s), Some(&i)]) {
            Err(InferError::Conflict {
                field,
                existing,
                incoming,
            }) => {
                assert_eq!(field, "col");
                assert_eq!(existing, "text");
                assert_eq!(incoming, "long");
            }
            other => panic!("expected conflict, got {other:?}"),
        }
    }

    #[test]
    fn infer_no_non_null_values_is_none() {
        assert_eq!(infer_datatype("x", &[]).unwrap(), None);
        assert_eq!(infer_datatype("x", &[None, None]).unwrap(), None);
        // JSON nulls carry no type information either.
        let n = json!(null);
        assert_eq!(infer_datatype("x", &[Some(&n), None]).unwrap(), None);
    }

    #[test]
    fn infer_schema_rejects_non_objects() {
        assert!(matches!(
            infer_schema(&json!(5)),
            Err(InferError::NotAnObject)
        ));
        assert!(matches!(
            infer_schema(&json!([1, 2])),
            Err(InferError::NotAnObject)
        ));
    }

    #[test]
    fn infer_schema_is_alphabetical_and_nullable() {
        let schema = infer_schema(&json!({"b": 1, "a": "x"})).unwrap();
        assert_eq!(schema.fields().len(), 2);
        assert_eq!(schema.field(0).name(), "a");
        assert_eq!(schema.field(0).data_type(), &DataType::Utf8);
        assert!(schema.field(0).is_nullable());
        assert_eq!(schema.field(1).name(), "b");
        assert_eq!(schema.field(1).data_type(), &DataType::Int64);
        assert!(schema.field(1).is_nullable());
    }

    #[test]
    fn infer_uniform_numeric_vectors() {
        let a = json!([1.0, 2.0, 3.0]);
        let b = json!([4, 5, 6]);
        let expected =
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 3);
        assert_eq!(
            infer_datatype("v", &[Some(&a), Some(&b)]).unwrap(),
            Some(expected)
        );
    }

    #[test]
    fn infer_ragged_vectors_unsupported() {
        let a = json!([1.0, 2.0]);
        let b = json!([3.0, 4.0, 5.0]);
        match infer_datatype("v", &[Some(&a), Some(&b)]) {
            Err(InferError::Unsupported { field, reason }) => {
                assert_eq!(field, "v");
                assert_eq!(reason, "ragged vector lengths");
            }
            other => panic!("expected unsupported, got {other:?}"),
        }
    }

    #[test]
    fn infer_short_vector_unsupported() {
        let a = json!([5]);
        match infer_datatype("v", &[Some(&a)]).unwrap_err() {
            InferError::Unsupported { field, reason } => {
                assert_eq!(field, "v");
                assert_eq!(reason, "vector length 1 < 2");
            }
            other => panic!("expected unsupported, got {other:?}"),
        }
    }

    #[test]
    fn infer_empty_arrays_default_to_utf8() {
        let empty = json!([]);
        assert_eq!(
            infer_datatype("v", &[Some(&empty), Some(&empty)]).unwrap(),
            None
        );
        let schema = infer_schema(&json!({"v": []})).unwrap();
        assert_eq!(schema.field(0).data_type(), &DataType::Utf8);
    }

    #[test]
    fn infer_nested_struct() {
        let doc = json!({"o": {"y": 2, "x": "s"}});
        let o = &doc["o"];
        let dt = infer_datatype("o", &[Some(o)]).unwrap().expect("struct");
        let fields = match &dt {
            DataType::Struct(fields) => fields,
            other => panic!("expected struct, got {other:?}"),
        };
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].name(), "x");
        assert_eq!(fields[0].data_type(), &DataType::Utf8);
        assert_eq!(fields[1].name(), "y");
        assert_eq!(fields[1].data_type(), &DataType::Int64);

        // Union across docs: absent children (and children that are null
        // in every doc) default to Utf8.
        let other = json!({"y": 3, "n": null});
        let dt = infer_datatype("o", &[Some(o), Some(&other)])
            .unwrap()
            .expect("struct");
        let fields = match &dt {
            DataType::Struct(fields) => fields,
            child => panic!("expected struct, got {child:?}"),
        };
        let names: Vec<&str> = fields.iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, vec!["n", "x", "y"]);
        assert_eq!(fields[0].data_type(), &DataType::Utf8);
        assert_eq!(fields[1].data_type(), &DataType::Utf8);
        assert_eq!(fields[2].data_type(), &DataType::Int64);
    }

    #[test]
    fn merge_schemas_promotes_and_appends() {
        let base = Schema::new(vec![
            Field::new("x", DataType::Int64, true),
            Field::new("y", DataType::Int64, true),
        ]);
        let incoming = Schema::new(vec![
            Field::new("y", DataType::Float64, true),
            Field::new("z", DataType::Int64, true),
        ]);
        let merged = merge_schemas(&base, &incoming).unwrap();
        assert_eq!(merged.fields().len(), 3);
        assert_eq!(merged.field(0).name(), "x");
        assert_eq!(merged.field(0).data_type(), &DataType::Int64);
        assert_eq!(merged.field(1).name(), "y");
        assert_eq!(merged.field(1).data_type(), &DataType::Float64);
        assert_eq!(merged.field(2).name(), "z");
        assert_eq!(merged.field(2).data_type(), &DataType::Int64);

        // A conflict inside a merged field (text vs double) must name
        // the actual column.
        let base = Schema::new(vec![Field::new("y", DataType::Utf8, true)]);
        let incoming = Schema::new(vec![Field::new("y", DataType::Float64, true)]);
        match merge_schemas(&base, &incoming).unwrap_err() {
            InferError::Conflict {
                field,
                existing,
                incoming,
            } => {
                assert_eq!(field, "y");
                assert_eq!(existing, "Utf8");
                assert_eq!(incoming, "Float64");
            }
            other => panic!("expected conflict, got {other:?}"),
        }
    }

    #[test]
    fn merge_fsl_dimension_mismatch_conflicts() {
        let base_fsl =
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 3);
        let inc_fsl =
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4);
        match merge_datatypes(&base_fsl, &inc_fsl).unwrap_err() {
            InferError::Conflict {
                field,
                existing,
                incoming,
            } => {
                // `merge_datatypes` reports no field; the caller rewrites it.
                assert_eq!(field, "");
                assert_eq!(existing, base_fsl.to_string());
                assert_ne!(existing, incoming);
            }
            other => panic!("expected conflict, got {other:?}"),
        }

        let schema_base = Schema::new(vec![Field::new("vec", base_fsl.clone(), true)]);
        let schema_inc = Schema::new(vec![Field::new("vec", inc_fsl.clone(), true)]);
        match merge_schemas(&schema_base, &schema_inc).unwrap_err() {
            InferError::Conflict { field, .. } => assert_eq!(field, "vec"),
            other => panic!("expected conflict, got {other:?}"),
        }
    }

    #[test]
    fn merge_fsl_same_dimension_unchanged() {
        let base =
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 3);
        let incoming =
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 3);
        assert_eq!(merge_datatypes(&base, &incoming).unwrap(), base);
    }

    #[test]
    fn merge_int_float_promotes_both_orders() {
        assert_eq!(
            merge_datatypes(&DataType::Int64, &DataType::Float64).unwrap(),
            DataType::Float64
        );
        assert_eq!(
            merge_datatypes(&DataType::Float64, &DataType::Int64).unwrap(),
            DataType::Float64
        );
    }

    #[test]
    fn merge_structs_union_children() {
        let base = DataType::Struct(Fields::from(vec![
            Arc::new(Field::new("x", DataType::Int64, true)),
            Arc::new(Field::new("y", DataType::Int64, true)),
        ]));
        let incoming = DataType::Struct(Fields::from(vec![
            Arc::new(Field::new("y", DataType::Float64, true)),
            Arc::new(Field::new("z", DataType::Utf8, true)),
        ]));
        let merged = merge_datatypes(&base, &incoming).unwrap();
        let fields = match &merged {
            DataType::Struct(fields) => fields,
            other => panic!("expected struct, got {other:?}"),
        };
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0].name(), "x");
        assert_eq!(fields[0].data_type(), &DataType::Int64);
        assert_eq!(fields[1].name(), "y");
        assert_eq!(fields[1].data_type(), &DataType::Float64);
        assert_eq!(fields[2].name(), "z");
        assert_eq!(fields[2].data_type(), &DataType::Utf8);
        assert!(fields.iter().all(|f| f.is_nullable()));
    }

    #[test]
    fn value_to_array_int64_with_null() {
        let arr = value_to_array("n", &DataType::Int64, &[Some(json!(1)), None]).unwrap();
        let arr = arr.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr.null_count(), 1);
        assert_eq!(arr.value(0), 1);
        assert!(arr.is_valid(0));
        assert!(arr.is_null(1));
    }

    #[test]
    fn value_to_array_u64_max_wraps_to_minus_one() {
        let arr = value_to_array("n", &DataType::Int64, &[Some(json!(u64::MAX))]).unwrap();
        let arr = arr.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(arr.value(0), -1);
        assert_eq!(arr.null_count(), 0);
    }

    #[test]
    fn value_to_array_float64_with_null() {
        let arr = value_to_array(
            "r",
            &DataType::Float64,
            &[Some(json!(1.5)), None, Some(json!(2))],
        )
        .unwrap();
        let arr = arr.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.value(0), 1.5);
        assert!(arr.is_null(1));
        assert_eq!(arr.value(2), 2.0);
    }

    #[test]
    fn value_to_array_utf8_nulls_and_type_check() {
        let arr = value_to_array("s", &DataType::Utf8, &[Some(json!("a")), None]).unwrap();
        let arr = arr.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr.value(0), "a");
        assert!(arr.is_null(1));

        match value_to_array(
            "s",
            &DataType::Utf8,
            &[Some(json!("ok")), None, Some(json!(1))],
        ) {
            Err(InferError::Unsupported { reason, .. }) => {
                assert_eq!(reason, "expected string, got long");
            }
            other => panic!("expected unsupported, got {other:?}"),
        }
    }

    #[test]
    fn value_to_array_boolean() {
        let arr = value_to_array(
            "b",
            &DataType::Boolean,
            &[Some(json!(true)), None, Some(json!(false))],
        )
        .unwrap();
        let arr = arr.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(arr.value(0));
        assert!(arr.is_null(1));
        assert!(!arr.value(2));

        match value_to_array("b", &DataType::Boolean, &[Some(json!("no"))]) {
            Err(InferError::Unsupported { reason, .. }) => {
                assert_eq!(reason, "expected boolean, got text");
            }
            other => panic!("expected unsupported, got {other:?}"),
        }
    }

    #[test]
    fn value_to_array_fixed_size_list() {
        let fsl = DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 3);
        let arr = value_to_array("v", &fsl, &[Some(json!([1.0, 2.0, 3.0])), None]).unwrap();
        let fsl_arr = arr.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
        assert_eq!(fsl_arr.len(), 2);
        assert_eq!(fsl_arr.value_length(), 3);
        assert!(fsl_arr.is_null(1));
        let flat = fsl_arr
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        assert_eq!(
            flat,
            &Float32Array::from(vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
        );

        match value_to_array("v", &fsl, &[Some(json!([1.0, 2.0]))]).unwrap_err() {
            InferError::Unsupported { reason, .. } => {
                assert!(reason.contains("expected 3"), "reason was: {reason}");
            }
            other => panic!("expected unsupported, got {other:?}"),
        }
    }

    #[test]
    fn value_to_array_struct_reuses_field_refs() {
        let dt = DataType::Struct(Fields::from(vec![
            Arc::new(Field::new("x", DataType::Utf8, true)),
            Arc::new(Field::new("y", DataType::Int64, true)),
        ]));
        let arr = value_to_array("o", &dt, &[Some(json!({"y": 2, "x": "s"})), None]).unwrap();
        let sa = arr.as_any().downcast_ref::<StructArray>().unwrap();
        assert_eq!(sa.len(), 2);
        assert_eq!(sa.null_count(), 1);
        let x = sa.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(x.value(0), "s");
        assert!(x.is_null(1));
        let y = sa.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(y.value(0), 2);
        assert!(y.is_null(1));

        // Smoke: a RecordBatch built from the same field references
        // accepts the struct array (DataType equality check passes).
        let schema = Arc::new(Schema::new(vec![Arc::new(Field::new(
            "o",
            dt.clone(),
            true,
        ))]));
        assert!(RecordBatch::try_new(schema, vec![arr]).is_ok());
    }

    #[test]
    fn value_to_array_null_datatype() {
        let arr = value_to_array("n", &DataType::Null, &[None, Some(json!(1))]).unwrap();
        let arr = arr.as_any().downcast_ref::<NullArray>().unwrap();
        assert_eq!(arr.len(), 2);
    }
}
