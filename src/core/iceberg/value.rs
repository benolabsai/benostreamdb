// Copyright (c) 2026 Richard Albright. All rights reserved.

use apache_avro::types::Value as AvroValue;
use base64::Engine;

/// Convert an Avro value to a JSON value
pub fn avro_to_json(val: AvroValue) -> serde_json::Value {
    match val {
        AvroValue::Null => serde_json::Value::Null,
        AvroValue::Boolean(b) => serde_json::json!(b),
        AvroValue::Int(i) => serde_json::json!(i),
        AvroValue::Long(l) => serde_json::json!(l),
        AvroValue::Float(f) => serde_json::json!(f),
        AvroValue::Double(d) => serde_json::json!(d),
        AvroValue::String(s) => serde_json::json!(s),
        AvroValue::Bytes(b) => serde_json::json!(b),
        AvroValue::Union(_, b) => avro_to_json(*b),
        _ => serde_json::Value::Null,
    }
}

/// Decode Iceberg binary value bounds using type information
pub fn decode_iceberg_value(
    type_json: &serde_json::Value,
    bytes: &[u8],
) -> crate::core::manifest::ManifestValue {
    use crate::core::manifest::ManifestValue;

    let type_str = if let Some(s) = type_json.as_str() {
        s
    } else if let Some(obj) = type_json.as_object() {
        obj.get("type").and_then(|t| t.as_str()).unwrap_or("unknown")
    } else {
        "unknown"
    };

    match type_str {
        "boolean" => {
            if !bytes.is_empty() {
                ManifestValue::Boolean(bytes[0] != 0)
            } else {
                ManifestValue::Null
            }
        }
        "int" | "date" => {
            if bytes.len() >= 4 {
                ManifestValue::Int32(i32::from_le_bytes(bytes[0..4].try_into().unwrap_or_default()))
            } else {
                ManifestValue::Null
            }
        }
        "long" | "timestamp" | "timestamptz" => {
            if bytes.len() >= 8 {
                ManifestValue::Int64(i64::from_le_bytes(bytes[0..8].try_into().unwrap_or_default()))
            } else {
                ManifestValue::Null
            }
        }
        "float" => {
            if bytes.len() >= 4 {
                ManifestValue::Float32(f32::from_le_bytes(bytes[0..4].try_into().unwrap_or_default()))
            } else {
                ManifestValue::Null
            }
        }
        "double" => {
            if bytes.len() >= 8 {
                ManifestValue::Float64(f64::from_le_bytes(bytes[0..8].try_into().unwrap_or_default()))
            } else {
                ManifestValue::Null
            }
        }
        "string" | "uuid" => {
            if let Ok(s) = std::str::from_utf8(bytes) {
                ManifestValue::String(s.to_string())
            } else {
                ManifestValue::String(base64::engine::general_purpose::STANDARD.encode(bytes))
            }
        }
        "binary" | "fixed" => {
            ManifestValue::String(base64::engine::general_purpose::STANDARD.encode(bytes))
        }
        _ => {
            // Best effort fallback
            parse_avro_value_bytes(bytes)
        }
    }
}

/// Parse Avro serialized bytes without type hint (best-effort inference)
pub fn parse_avro_value_bytes(bytes: &[u8]) -> crate::core::manifest::ManifestValue {
    // Iceberg stores min/max bounds as the serialized binary of the type.
    // Without strict type info, we attempt best-effort inference from byte length.
    parse_avro_value_bytes_with_type(bytes, None)
}

/// Parse Avro serialized bytes with an optional type hint for accurate decoding.
/// When `type_hint` is provided, uses typed decoding; otherwise falls back to length-based inference.
pub fn parse_avro_value_bytes_with_type(
    bytes: &[u8],
    type_hint: Option<&str>,
) -> crate::core::manifest::ManifestValue {
    use crate::core::manifest::ManifestValue;

    // If a type hint is available, use the typed decoder for correctness
    if let Some(type_name) = type_hint {
        let type_json = serde_json::json!(type_name);
        return decode_iceberg_value(&type_json, bytes);
    }

    // Best-effort inference from byte length (legacy fallback)
    if bytes.len() == 4 {
        // Could be int or float — prefer int as it's more common in partition bounds
        let val = i32::from_le_bytes(bytes.try_into().unwrap_or_default());
        ManifestValue::Int64(val as i64)
    } else if bytes.len() == 8 {
        // Could be long or double — prefer long for consistency
        let val = i64::from_le_bytes(bytes.try_into().unwrap_or_default());
        ManifestValue::Int64(val)
    } else {
        // String or Binary
        if let Ok(s) = std::str::from_utf8(bytes) {
            ManifestValue::String(s.to_string())
        } else {
             ManifestValue::String(base64::engine::general_purpose::STANDARD.encode(bytes))
        }
    }
}

/// Convert a JSON value to an Avro value
pub fn json_to_avro_value(v: &serde_json::Value) -> AvroValue {
    match v {
        serde_json::Value::Null => AvroValue::Null,
        serde_json::Value::Bool(b) => AvroValue::Boolean(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                if i >= (i32::MIN as i64) && i <= (i32::MAX as i64) {
                    AvroValue::Int(i as i32)
                } else {
                    AvroValue::Long(i)
                }
            } else if let Some(f) = n.as_f64() {
                AvroValue::Double(f)
            } else {
                AvroValue::Null
            }
        },
        serde_json::Value::String(s) => AvroValue::String(s.clone()),
        _ => AvroValue::Null
    }
}
