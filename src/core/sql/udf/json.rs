// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! PostgreSQL `json` scalar UDFs.
//!
//! JSON is stored as a `Utf8` string column (text, not a decomposed binary
//! representation), so these use PostgreSQL's `json`-type names rather than
//! `jsonb_*`. The session dialect is PostgreSQL, so the semantics follow
//! PostgreSQL (not MySQL's `JSON_*`):
//!
//! ```sql
//! SELECT json_extract_path_text(payload, 'user', 'id') FROM events;
//! SELECT json_extract_path(payload, 'tags', '0') FROM events;
//! SELECT * FROM events WHERE json_contains(payload, '{"level":"error"}');
//! SELECT * FROM events WHERE json_exists(payload, 'error');
//! SELECT json_typeof(payload) FROM events;
//! SELECT * FROM events WHERE json_path_exists(payload, '$.items[*].id');
//! SELECT json_path_query(payload, '$.items[*].id') FROM events;
//! ```
//!
//! `json_extract_path`, `json_extract_path_text`, and `json_typeof` are the
//! standard PostgreSQL `json` functions. `json_contains`, `json_exists`,
//! `json_path_exists`, and `json_path_query` are extensions that mirror the
//! `jsonb` containment/path semantics on the text representation.
//!
//! Path syntax for the `json_path_*` functions: an optional `$` root, then
//! `.key` / bare `key` segments, `[index]` subscripts, and `[*]` / `.*`
//! wildcards (all array elements or object values). `json_extract_path` takes
//! the path as variadic text elements, matching PostgreSQL.

use arrow::array::{Array, ArrayRef, BooleanBuilder, ListBuilder, StringArray, StringBuilder};
use arrow::datatypes::{DataType, Field};
use datafusion::error::Result;
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
};
use serde_json::Value;
use std::any::Any;
use std::sync::Arc;

macro_rules! impl_dyn_traits {
    ($name:ident) => {
        impl PartialEq for $name {
            fn eq(&self, _other: &Self) -> bool {
                true
            }
        }
        impl Eq for $name {}
        impl std::hash::Hash for $name {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                std::any::type_name::<Self>().hash(state);
            }
        }
    };
}

/// One step of a parsed JSON path.
#[derive(Debug, PartialEq, Eq)]
enum PathStep {
    /// A key (object) or an index (array) — resolved contextually, matching
    /// PostgreSQL's `json_extract_path`.
    Key(String),
    /// `[*]` / `.*` — every array element or object value.
    Wildcard,
}

/// Parse a JSON path into steps. Accepts `$.a.b[0]`, `a.b[0]`, `$[0]`,
/// `$.a[*]`, `$.*`, and quoted subscripts (`$["a b"]`).
fn parse_path(path: &str) -> Vec<PathStep> {
    let mut steps = Vec::new();
    let mut chars = path.chars().peekable();
    if chars.peek() == Some(&'$') {
        chars.next();
    }
    let mut buf = String::new();
    let flush = |buf: &mut String, steps: &mut Vec<PathStep>| {
        if !buf.is_empty() {
            steps.push(PathStep::Key(std::mem::take(buf)));
        }
    };
    while let Some(c) = chars.next() {
        match c {
            '.' => flush(&mut buf, &mut steps),
            '[' => {
                flush(&mut buf, &mut steps);
                let mut idx = String::new();
                for c2 in chars.by_ref() {
                    if c2 == ']' {
                        break;
                    }
                    idx.push(c2);
                }
                let idx = idx.trim().trim_matches(|c| c == '"' || c == '\'');
                if idx == "*" {
                    steps.push(PathStep::Wildcard);
                } else {
                    steps.push(PathStep::Key(idx.to_string()));
                }
            }
            '*' => {
                // `.*` wildcard (a bare `*` segment).
                flush(&mut buf, &mut steps);
                steps.push(PathStep::Wildcard);
            }
            _ => buf.push(c),
        }
    }
    flush(&mut buf, &mut steps);
    steps
}

/// Resolve one step against a value: object key, or array index (a numeric
/// key), matching PostgreSQL's `json_extract_path`.
fn step_value<'a>(v: &'a Value, key: &str) -> Option<&'a Value> {
    match v {
        Value::Object(o) => o.get(key),
        Value::Array(a) => key.parse::<usize>().ok().and_then(|i| a.get(i)),
        _ => None,
    }
}

/// Evaluate a path, returning every match (wildcards can yield many).
fn eval_path<'a>(root: &'a Value, steps: &[PathStep]) -> Vec<&'a Value> {
    let mut current = vec![root];
    for step in steps {
        let mut next = Vec::new();
        for v in current {
            match step {
                PathStep::Key(k) => {
                    if let Some(child) = step_value(v, k) {
                        next.push(child);
                    }
                }
                PathStep::Wildcard => match v {
                    Value::Array(a) => next.extend(a.iter()),
                    Value::Object(o) => next.extend(o.values()),
                    _ => {}
                },
            }
        }
        current = next;
    }
    current
}

/// First match of a path, if any.
fn resolve_one<'a>(root: &'a Value, steps: &[PathStep]) -> Option<&'a Value> {
    eval_path(root, steps).into_iter().next()
}

/// Whether `candidate` is contained in `haystack` (PostgreSQL `@>`): recursive
/// subset for objects, element membership for arrays, equality for scalars.
fn contains(haystack: &Value, candidate: &Value) -> bool {
    match (haystack, candidate) {
        (Value::Object(h), Value::Object(c)) => {
            c.iter().all(|(k, v)| h.get(k).is_some_and(|hv| contains(hv, v)))
        }
        (Value::Array(h), Value::Array(c)) => {
            c.iter().all(|cv| h.iter().any(|hv| contains(hv, cv)))
        }
        (Value::Array(h), _) => h.iter().any(|hv| contains(hv, candidate)),
        _ => haystack == candidate,
    }
}

/// PostgreSQL `jsonb_exists` (`?`): a top-level object key, or an array element.
fn exists(haystack: &Value, key: &str) -> bool {
    match haystack {
        Value::Object(o) => o.contains_key(key),
        Value::Array(a) => a.iter().any(|v| v.as_str() == Some(key)),
        _ => false,
    }
}

/// PostgreSQL `json_typeof`.
fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Text form for `json_extract_path_text`: strings are unquoted, everything
/// else is its JSON text.
fn as_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn as_string_array(value: &ColumnarValue, rows: usize) -> Result<ArrayRef> {
    value.clone().into_array(rows)
}

fn string_at(arr: &ArrayRef, i: usize) -> Result<Option<&str>> {
    // A `NULL` literal arrives as a `NullArray` (or a typed null), so check
    // nullability before downcasting to `StringArray`.
    if arr.is_null(i) {
        return Ok(None);
    }
    match arr.as_any().downcast_ref::<StringArray>() {
        Some(s) => Ok(Some(s.value(i))),
        // A non-string, non-null value is treated as absent rather than
        // erroring, so JSON functions stay total.
        None => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// json_extract_path(json, VARIADIC path...) -> json (Utf8)
// json_extract_path_text(json, VARIADIC path...) -> text (Utf8)
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct JsonExtractPathUDF {
    name: &'static str,
    as_text: bool,
    signature: Signature,
}
impl_dyn_traits!(JsonExtractPathUDF);
impl JsonExtractPathUDF {
    pub fn new() -> Self {
        Self {
            name: "json_extract_path",
            as_text: false,
            signature: Signature::variadic_any(Volatility::Immutable),
        }
    }
    pub fn text() -> Self {
        Self {
            name: "json_extract_path_text",
            as_text: true,
            signature: Signature::variadic_any(Volatility::Immutable),
        }
    }
}
impl Default for JsonExtractPathUDF {
    fn default() -> Self {
        Self::new()
    }
}
impl ScalarUDFImpl for JsonExtractPathUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        self.name
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Utf8)
    }
    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let rows = args.number_rows;
        if args.args.is_empty() {
            return Ok(ColumnarValue::Array(Arc::new(StringArray::new_null(rows))));
        }
        let json_arr = as_string_array(&args.args[0], rows)?;
        let path_arrays: Vec<ArrayRef> = args.args[1..]
            .iter()
            .map(|a| as_string_array(a, rows))
            .collect::<Result<_>>()?;

        let mut out = StringBuilder::with_capacity(rows, rows * 16);
        for i in 0..rows {
            let json = string_at(&json_arr, i)?;
            // A null path element makes the whole result null.
            let mut elems = Vec::with_capacity(path_arrays.len());
            let mut null_elem = false;
            for a in &path_arrays {
                match string_at(a, i)? {
                    Some(s) => elems.push(s.to_string()),
                    None => null_elem = true,
                }
            }
            match (json, null_elem) {
                (Some(json), false) => match serde_json::from_str::<Value>(json) {
                    Ok(v) => {
                        let steps: Vec<PathStep> = elems.into_iter().map(PathStep::Key).collect();
                        match resolve_one(&v, &steps) {
                            Some(val) if self.as_text => out.append_value(as_text(val)),
                            Some(val) => out.append_value(val.to_string()),
                            None => out.append_null(),
                        }
                    }
                    Err(_) => out.append_null(),
                },
                _ => out.append_null(),
            }
        }
        Ok(ColumnarValue::Array(Arc::new(out.finish())))
    }
}

// ---------------------------------------------------------------------------
// json_contains(json, json) -> Boolean
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct JsonContainsUDF {
    signature: Signature,
}
impl_dyn_traits!(JsonContainsUDF);
impl Default for JsonContainsUDF {
    fn default() -> Self {
        Self::new()
    }
}
impl JsonContainsUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::any(2, Volatility::Immutable),
        }
    }
}
impl ScalarUDFImpl for JsonContainsUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "json_contains"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Boolean)
    }
    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let rows = args.number_rows;
        let json_arr = as_string_array(&args.args[0], rows)?;
        let cand_arr = as_string_array(&args.args[1], rows)?;

        let mut out = BooleanBuilder::with_capacity(rows);
        for i in 0..rows {
            match (string_at(&json_arr, i)?, string_at(&cand_arr, i)?) {
                (Some(json), Some(cand)) => {
                    match (
                        serde_json::from_str::<Value>(json),
                        serde_json::from_str::<Value>(cand),
                    ) {
                        (Ok(haystack), Ok(candidate)) => {
                            out.append_value(contains(&haystack, &candidate))
                        }
                        _ => out.append_value(false),
                    }
                }
                _ => out.append_null(),
            }
        }
        Ok(ColumnarValue::Array(Arc::new(out.finish())))
    }
}

// ---------------------------------------------------------------------------
// json_exists(json, text) -> Boolean
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct JsonExistsUDF {
    signature: Signature,
}
impl_dyn_traits!(JsonExistsUDF);
impl Default for JsonExistsUDF {
    fn default() -> Self {
        Self::new()
    }
}
impl JsonExistsUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::any(2, Volatility::Immutable),
        }
    }
}
impl ScalarUDFImpl for JsonExistsUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "json_exists"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Boolean)
    }
    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let rows = args.number_rows;
        let json_arr = as_string_array(&args.args[0], rows)?;
        let key_arr = as_string_array(&args.args[1], rows)?;

        let mut out = BooleanBuilder::with_capacity(rows);
        for i in 0..rows {
            match (string_at(&json_arr, i)?, string_at(&key_arr, i)?) {
                (Some(json), Some(key)) => match serde_json::from_str::<Value>(json) {
                    Ok(v) => out.append_value(exists(&v, key)),
                    Err(_) => out.append_value(false),
                },
                _ => out.append_null(),
            }
        }
        Ok(ColumnarValue::Array(Arc::new(out.finish())))
    }
}

// ---------------------------------------------------------------------------
// json_typeof(json) -> text
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct JsonTypeofUDF {
    signature: Signature,
}
impl_dyn_traits!(JsonTypeofUDF);
impl Default for JsonTypeofUDF {
    fn default() -> Self {
        Self::new()
    }
}
impl JsonTypeofUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}
impl ScalarUDFImpl for JsonTypeofUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "json_typeof"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Utf8)
    }
    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let rows = args.number_rows;
        let json_arr = as_string_array(&args.args[0], rows)?;

        let mut out = StringBuilder::with_capacity(rows, rows * 8);
        for i in 0..rows {
            match string_at(&json_arr, i)? {
                Some(json) => match serde_json::from_str::<Value>(json) {
                    Ok(v) => out.append_value(type_name(&v)),
                    Err(_) => out.append_null(),
                },
                None => out.append_null(),
            }
        }
        Ok(ColumnarValue::Array(Arc::new(out.finish())))
    }
}

// ---------------------------------------------------------------------------
// json_path_exists(json, jsonpath) -> Boolean
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct JsonPathExistsUDF {
    signature: Signature,
}
impl_dyn_traits!(JsonPathExistsUDF);
impl Default for JsonPathExistsUDF {
    fn default() -> Self {
        Self::new()
    }
}
impl JsonPathExistsUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::any(2, Volatility::Immutable),
        }
    }
}
impl ScalarUDFImpl for JsonPathExistsUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "json_path_exists"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Boolean)
    }
    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let rows = args.number_rows;
        let json_arr = as_string_array(&args.args[0], rows)?;
        let path_arr = as_string_array(&args.args[1], rows)?;

        let mut out = BooleanBuilder::with_capacity(rows);
        for i in 0..rows {
            match (string_at(&json_arr, i)?, string_at(&path_arr, i)?) {
                (Some(json), Some(path)) => match serde_json::from_str::<Value>(json) {
                    Ok(v) => out.append_value(!eval_path(&v, &parse_path(path)).is_empty()),
                    Err(_) => out.append_value(false),
                },
                _ => out.append_null(),
            }
        }
        Ok(ColumnarValue::Array(Arc::new(out.finish())))
    }
}

// ---------------------------------------------------------------------------
// json_path_query(json, jsonpath) -> List<Utf8> (all matches)
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct JsonPathQueryUDF {
    signature: Signature,
}
impl_dyn_traits!(JsonPathQueryUDF);
impl Default for JsonPathQueryUDF {
    fn default() -> Self {
        Self::new()
    }
}
impl JsonPathQueryUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::any(2, Volatility::Immutable),
        }
    }
}
impl ScalarUDFImpl for JsonPathQueryUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "json_path_query"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::List(Arc::new(Field::new(
            "item",
            DataType::Utf8,
            true,
        ))))
    }
    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let rows = args.number_rows;
        let json_arr = as_string_array(&args.args[0], rows)?;
        let path_arr = as_string_array(&args.args[1], rows)?;

        let mut builder = ListBuilder::new(StringBuilder::new());
        for i in 0..rows {
            match (string_at(&json_arr, i)?, string_at(&path_arr, i)?) {
                (Some(json), Some(path)) => match serde_json::from_str::<Value>(json) {
                    Ok(v) => {
                        for m in eval_path(&v, &parse_path(path)) {
                            builder.values().append_value(m.to_string());
                        }
                        builder.append(true);
                    }
                    Err(_) => builder.append(false),
                },
                _ => builder.append(false),
            }
        }
        Ok(ColumnarValue::Array(Arc::new(builder.finish())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_path_handles_root_dots_indices_and_wildcards() {
        assert_eq!(
            parse_path("$.a.b[0].c"),
            vec![
                PathStep::Key("a".into()),
                PathStep::Key("b".into()),
                PathStep::Key("0".into()),
                PathStep::Key("c".into()),
            ]
        );
        assert_eq!(
            parse_path("a.b"),
            vec![PathStep::Key("a".into()), PathStep::Key("b".into())]
        );
        assert_eq!(parse_path("$[2]"), vec![PathStep::Key("2".into())]);
        assert_eq!(parse_path("$[\"a b\"]"), vec![PathStep::Key("a b".into())]);
        assert_eq!(
            parse_path("$.items[*].id"),
            vec![
                PathStep::Key("items".into()),
                PathStep::Wildcard,
                PathStep::Key("id".into()),
            ]
        );
        assert_eq!(parse_path("$.*"), vec![PathStep::Wildcard]);
    }

    #[test]
    fn eval_path_walks_objects_arrays_and_wildcards() {
        let v: Value = serde_json::from_str(r#"{"a":{"b":[10,20,{"c":"x"}]}}"#).unwrap();
        assert_eq!(
            resolve_one(&v, &parse_path("$.a.b[2].c")),
            Some(&Value::String("x".into()))
        );
        assert_eq!(
            resolve_one(&v, &parse_path("$.a.b[1]")),
            Some(&Value::from(20))
        );
        assert_eq!(resolve_one(&v, &parse_path("$.a.missing")), None);
        assert_eq!(resolve_one(&v, &parse_path("$.a.b[9]")), None);

        let items: Value =
            serde_json::from_str(r#"{"items":[{"id":1},{"id":2},{"id":3}]}"#).unwrap();
        let ids = eval_path(&items, &parse_path("$.items[*].id"));
        assert_eq!(ids, vec![&Value::from(1), &Value::from(2), &Value::from(3)]);
    }

    #[test]
    fn contains_is_recursive_subset() {
        let hay: Value =
            serde_json::from_str(r#"{"level":"error","tags":["a","b"],"n":1}"#).unwrap();
        assert!(contains(
            &hay,
            &serde_json::from_str(r#"{"level":"error"}"#).unwrap()
        ));
        assert!(contains(
            &hay,
            &serde_json::from_str(r#"{"tags":["b"]}"#).unwrap()
        ));
        // A scalar candidate is not contained in an object target, but is
        // contained in an array target.
        assert!(!contains(&hay, &serde_json::from_str(r#""a""#).unwrap()));
        assert!(contains(
            &serde_json::from_str::<Value>(r#"["a","b"]"#).unwrap(),
            &serde_json::from_str(r#""a""#).unwrap()
        ));
        assert!(!contains(
            &hay,
            &serde_json::from_str(r#"{"level":"warn"}"#).unwrap()
        ));
    }

    #[test]
    fn exists_and_typeof() {
        let v: Value = serde_json::from_str(r#"{"a":1,"tags":["x","y"]}"#).unwrap();
        assert!(exists(&v, "a"));
        assert!(!exists(&v, "b"));
        assert!(exists(&v["tags"], "x"));
        assert_eq!(type_name(&v), "object");
        assert_eq!(type_name(&v["a"]), "number");
        assert_eq!(type_name(&v["tags"]), "array");
        assert_eq!(type_name(&Value::Null), "null");
    }
}
