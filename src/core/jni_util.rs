// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Pure-Rust helpers shared by the JNI bridge.
//!
//! These are split out of `ffi.rs` (which is gated on the `java` feature) so
//! the untrusted-input parsing they perform — the `[{name, type, nullable}]`
//! schema JSON and the Arrow type-name mapping — can be fuzzed without a JVM.
//! The JNI entry points call straight into these.

use arrow::datatypes::{DataType, Field, Schema};

use std::sync::Arc;

fn format_datatype(dt: &DataType) -> String {
    match dt {
        DataType::List(field) => format!("List({})", format_datatype(field.data_type())),
        DataType::Struct(fields) => {
            let mut s = String::from("Struct(");
            for (i, f) in fields.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                s.push_str(f.name());
                s.push_str(": ");
                s.push_str(&format_datatype(f.data_type()));
            }
            s.push(')');
            s
        }
        _ => format!("{}", dt),
    }
}

/// Serialize an Arrow schema to a compact JSON array of `{name, type, nullable}`.
pub fn schema_to_json(schema: &Schema) -> String {
    let fields: Vec<serde_json::Value> = schema
        .fields()
        .iter()
        .map(|f| {
            serde_json::json!({
                "name": f.name(),
                "type": format_datatype(f.data_type()),
                "nullable": f.is_nullable(),
            })
        })
        .collect();
    serde_json::Value::Array(fields).to_string()
}

/// Map an Arrow type name (as produced by [`schema_to_json`]) back to a `DataType`.
pub fn arrow_type_from_str(s: &str) -> DataType {
    if s.starts_with("List(") && s.ends_with(")") {
        let inner = &s[5..s.len() - 1];
        return DataType::List(Arc::new(Field::new(
            "item",
            arrow_type_from_str(inner),
            true,
        )));
    }
    if s.starts_with("Struct(") && s.ends_with(")") {
        let inner = &s[7..s.len() - 1];
        let mut fields = Vec::new();
        for part in inner.split(", ") {
            if let Some((name, ty)) = part.split_once(": ") {
                fields.push(Field::new(name, arrow_type_from_str(ty), true));
            }
        }
        return DataType::Struct(arrow::datatypes::Fields::from(fields));
    }

    use DataType::*;
    match s {
        "Int8" => Int8,
        "Int16" => Int16,
        "Int32" => Int32,
        "Int64" => Int64,
        "UInt8" => UInt8,
        "UInt16" => UInt16,
        "UInt32" => UInt32,
        "UInt64" => UInt64,
        "Float16" => Float16,
        "Float32" => Float32,
        "Float64" => Float64,
        "Boolean" => Boolean,
        "Date32" => Date32,
        "Date64" => Date64,
        "Utf8" => Utf8,
        "LargeUtf8" => LargeUtf8,
        _ => Utf8,
    }
}

/// Parse the `[{name, type, nullable}]` JSON produced by [`schema_to_json`].
///
/// Malformed input returns an error; it must never panic (the JNI `createTable`
/// entry point feeds this straight from a Java-supplied string).
pub fn schema_from_json(json: &str) -> anyhow::Result<Schema> {
    let fields: Vec<serde_json::Value> = serde_json::from_str(json)?;
    let mut arrow_fields = Vec::with_capacity(fields.len());
    for f in fields {
        let name = f.get("name").and_then(|v| v.as_str()).unwrap_or("col");
        let ty = f.get("type").and_then(|v| v.as_str()).unwrap_or("Utf8");
        let nullable = f.get("nullable").and_then(|v| v.as_bool()).unwrap_or(true);
        arrow_fields.push(Field::new(name, arrow_type_from_str(ty), nullable));
    }
    Ok(Schema::new(arrow_fields))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_json_round_trips() {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]);
        let json = schema_to_json(&schema);
        let parsed = schema_from_json(&json).expect("round-trip");
        assert_eq!(parsed.fields().len(), 2);
        assert_eq!(parsed.field(0).name(), "id");
        assert_eq!(parsed.field(0).data_type(), &DataType::Int32);
        assert_eq!(parsed.field(1).name(), "name");
    }

    #[test]
    fn malformed_schema_json_errors_not_panics() {
        assert!(schema_from_json("not json").is_err());
        assert!(schema_from_json("{}").is_err());
        assert!(schema_from_json("").is_err());
        // A JSON array of non-objects is tolerated (fields default).
        assert!(schema_from_json("[1, 2, 3]").is_ok());
    }

    #[test]
    fn unknown_type_falls_back_to_utf8() {
        assert_eq!(arrow_type_from_str("Nonsense"), DataType::Utf8);
    }
}
