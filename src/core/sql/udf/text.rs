// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Text-scoring scalar UDFs: `bm25_score(text, query)` and `tf_idf(text)`.
//!
//! These are **corpus-free** variants: a scalar UDF has no access to the
//! table's document statistics, so the IDF term is fixed at `1.0` and the
//! length-normalization term is `1.0` (no length penalty). What remains is the
//! BM25 term-frequency saturation, which is exactly what you want for ranking
//! rows against a single query:
//!
//! ```text
//! bm25_score(d, q) = \sum_{t in q} tf(t, d) * (k1 + 1) / (tf(t, d) + k1)
//! ```
//!
//! For corpus-aware BM25 (real IDF and `avgdl`), use the table's BM25 index
//! via the keyword-search path, which has the segment statistics.

use arrow::array::{Array, ArrayRef, Float32Builder, ListBuilder, StringArray};
use arrow::datatypes::DataType;
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
};
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

/// Lucene default term-frequency saturation parameter.
const K1: f32 = 1.2;

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

/// Lowercase whitespace tokenizer (matches the default analyzer's token set
/// closely enough for scoring).
fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|t| {
            t.trim_matches(|c: char| !c.is_alphanumeric())
                .to_ascii_lowercase()
        })
        .filter(|t| !t.is_empty())
        .collect()
}

/// Term frequencies of `text`, in first-appearance order.
fn term_frequencies(text: &str) -> Vec<(String, u32)> {
    let mut order: Vec<String> = Vec::new();
    let mut counts: HashMap<String, u32> = HashMap::new();
    for token in tokenize(text) {
        let entry = counts.entry(token.clone()).or_insert(0);
        if *entry == 0 {
            order.push(token);
        }
        *entry += 1;
    }
    order
        .into_iter()
        .map(|t| {
            let c = counts[&t];
            (t, c)
        })
        .collect()
}

fn as_string_array(value: &ColumnarValue, rows: usize) -> Result<ArrayRef> {
    value.clone().into_array(rows)
}

fn string_at(arr: &ArrayRef, i: usize) -> Result<Option<String>> {
    let s = arr
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| DataFusionError::Execution("expected Utf8 argument".to_string()))?;
    if s.is_null(i) {
        Ok(None)
    } else {
        Ok(Some(s.value(i).to_string()))
    }
}

// ---------------------------------------------------------------------------
// bm25_score(text, query) -> Float32
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct Bm25ScoreUDF {
    signature: Signature,
}
impl_dyn_traits!(Bm25ScoreUDF);
impl Default for Bm25ScoreUDF {
    fn default() -> Self {
        Self::new()
    }
}
impl Bm25ScoreUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::any(2, Volatility::Immutable),
        }
    }
}
impl ScalarUDFImpl for Bm25ScoreUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "bm25_score"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float32)
    }
    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let rows = args.number_rows;
        let text = as_string_array(&args.args[0], rows)?;
        let query = as_string_array(&args.args[1], rows)?;

        let mut out = Float32Builder::with_capacity(rows);
        for i in 0..rows {
            match (string_at(&text, i)?, string_at(&query, i)?) {
                (Some(doc), Some(q)) => {
                    let tf: HashMap<String, u32> = term_frequencies(&doc).into_iter().collect();
                    let mut score = 0.0f32;
                    for term in tokenize(&q) {
                        if let Some(&f) = tf.get(&term) {
                            let f = f as f32;
                            score += f * (K1 + 1.0) / (f + K1);
                        }
                    }
                    out.append_value(score);
                }
                _ => out.append_null(),
            }
        }
        Ok(ColumnarValue::Array(Arc::new(out.finish())))
    }
}

// ---------------------------------------------------------------------------
// tf_idf(text) -> List<Float32>
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct TfIdfUDF {
    signature: Signature,
}
impl_dyn_traits!(TfIdfUDF);
impl Default for TfIdfUDF {
    fn default() -> Self {
        Self::new()
    }
}
impl TfIdfUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}
impl ScalarUDFImpl for TfIdfUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "tf_idf"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::List(Arc::new(arrow::datatypes::Field::new(
            "item",
            DataType::Float32,
            true,
        ))))
    }
    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let rows = args.number_rows;
        let text = as_string_array(&args.args[0], rows)?;

        let mut builder = ListBuilder::new(Float32Builder::new());
        for i in 0..rows {
            match string_at(&text, i)? {
                Some(doc) => {
                    // Corpus-free: IDF is fixed at 1.0, so the weight is the
                    // normalized term frequency.
                    let freqs = term_frequencies(&doc);
                    let total: u32 = freqs.iter().map(|(_, c)| *c).sum();
                    let denom = total.max(1) as f32;
                    for (_, count) in freqs {
                        builder.values().append_value(count as f32 / denom);
                    }
                    builder.append(true);
                }
                None => builder.append(false),
            }
        }
        Ok(ColumnarValue::Array(Arc::new(builder.finish())))
    }
}
