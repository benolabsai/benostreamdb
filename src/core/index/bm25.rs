// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Okapi BM25 scoring primitives for keyword search.
//!
//! Scores use the standard BM25 formulation:
//!
//! ```text
//! score(q, d) = \sum_{t in q} IDF(t) * tf*(k1 + 1) / (tf + k1 * norm)
//! norm = 1 - b + b * (len(d) / avgdl)
//! ```
//!
//! IDF uses the Lucene-variant (positive, `+1`) formula:
//! `IDF(t) = ln((N - df + 0.5) / (df + 0.5) + 1)`.

/// Tunable BM25 parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bm25Params {
    /// Term-frequency saturation parameter (Lucene default: 1.2).
    pub k1: f32,
    /// Document length normalization factor (Lucene default: 0.75).
    pub b: f32,
}

impl Default for Bm25Params {
    fn default() -> Self {
        // Derived Default would give k1 = b = 0.0 (f32::default), which would
        // collapse every norm to zero and saturate every tf. Use Lucene's
        // canonical values instead.
        Self { k1: 1.2, b: 0.75 }
    }
}

/// Lucene-style (always non-negative) inverse document frequency.
///
/// `df` is the document frequency of the term; `n_docs` the total number of
/// documents in the segment.
pub fn idf(df: usize, n_docs: usize) -> f32 {
    let n = n_docs.max(1) as f32;
    let f = df as f32;
    ((n - f + 0.5) / (f + 0.5) + 1.0).ln()
}

/// Length normalization factor.
///
/// Degrades gracefully to 1.0 (no normalization) when `avg_doc_len` is zero
/// or non-positive, e.g. when a segment has no doc-length sidecar yet
/// (pre-M2 inverted files).
pub fn doc_len_norm(doc_len: u32, avg_doc_len: f32, b: f32) -> f32 {
    if avg_doc_len <= 0.0 || b <= 0.0 {
        1.0
    } else {
        1.0 - b + b * (doc_len as f32 / avg_doc_len)
    }
}

/// BM25 score for a single term in a single document.
///
/// `tf` is the raw term frequency (occurrences) in the document; `df` its
/// document frequency across the segment; `n_docs` the segment's document
/// count; `doc_len` the token count of the document; `avg_doc_len` the
/// segment's average token count per document.
pub fn term_score(
    tf: f32,
    df: usize,
    n_docs: usize,
    doc_len: u32,
    avg_doc_len: f32,
    params: &Bm25Params,
) -> f32 {
    let norm = doc_len_norm(doc_len, avg_doc_len, params.b);
    idf(df, n_docs) * (tf * (params.k1 + 1.0) / (tf + params.k1 * norm))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_lucene() {
        let p = Bm25Params::default();
        assert_eq!(p.k1, 1.2);
        assert_eq!(p.b, 0.75);
    }

    #[test]
    fn idf_is_positive_and_decreases_with_df() {
        let rare = idf(1, 100);
        let common = idf(50, 100);
        assert!(rare > 0.0);
        assert!(common > 0.0);
        assert!(rare > common);
    }

    #[test]
    fn idf_single_doc_segment_is_positive() {
        // The +1 variant keeps IDF positive even when the term appears in
        // every document of the segment.
        assert!(idf(1, 1) > 0.0);
        assert!(idf(10, 10) > 0.0);
    }

    #[test]
    fn term_score_monotonic_in_tf() {
        let p = Bm25Params::default();
        let s1 = term_score(1.0, 2, 100, 10, 10.0, &p);
        let s2 = term_score(5.0, 2, 100, 10, 10.0, &p);
        assert!(s2 > s1);
        let s3 = term_score(25.0, 2, 100, 10, 10.0, &p);
        assert!(s3 > s2);
    }

    #[test]
    fn longer_documents_are_penalized() {
        let p = Bm25Params::default();
        let short_doc = term_score(3.0, 2, 100, 5, 20.0, &p);
        let long_doc = term_score(3.0, 2, 100, 40, 20.0, &p);
        assert!(short_doc > long_doc);
    }

    #[test]
    fn b_zero_disables_length_norm() {
        let p = Bm25Params { k1: 1.2, b: 0.0 };
        let s_short = term_score(3.0, 2, 100, 5, 20.0, &p);
        let s_long = term_score(3.0, 2, 100, 40, 20.0, &p);
        assert_eq!(s_short, s_long);
    }

    #[test]
    fn missing_avg_doc_len_degrades_to_unnormalized() {
        let p = Bm25Params::default();
        let unnormalized = term_score(3.0, 2, 100, 42, 0.0, &p);
        // With norm == 1.0 the formula is idf * tf*(k1+1)/(tf + k1).
        let expected = idf(2, 100) * (3.0 * 2.2 / (3.0 + 1.2));
        assert!((unnormalized - expected).abs() < 1e-6);
    }

    #[test]
    fn doc_len_norm_identity_at_average_length() {
        let p = Bm25Params::default();
        assert!((doc_len_norm(20, 20.0, p.b) - 1.0).abs() < 1e-6);
    }
}
