// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Vector UDF modules decomposed from the original monolithic vector_udf module.
//!
//! Submodules:
//! - `distance`  — Distance UDFs (L2, Cosine, IP, L1, Hamming, Jaccard) and sparse distance helpers
//! - `transform` — Vector transform UDFs (add, sub, mul, concat, dims, norm, normalize, quantize, subvector, to_binary)
//! - `aggregate` — Aggregate UDFs (vector_sum, vector_avg) and accumulator types
//! - `sparse`    — Sparse vector conversion UDFs and sparse utility functions

pub mod aggregate;
pub mod distance;
pub mod json;
pub mod sparse;
pub mod text;
pub mod transform;
pub mod vector_stats;

use datafusion::logical_expr::{AggregateUDF, ScalarUDF};

// -- Re-exports from distance --

pub use distance::{
    dense_to_sparse, sparse_cosine_distance, sparse_inner_product_distance, sparse_l2_distance,
    sparse_to_dense, CosineDistUDF, HammingDistUDF, IPDistUDF, JaccardDistUDF, L1DistUDF,
    L2DistUDF,
};

// -- Re-exports from transform --

pub use transform::{
    BinaryQuantizeUDF, SubvectorUDF, VectorAddUDF, VectorConcatUDF, VectorDimsUDF, VectorMulUDF,
    VectorNormUDF, VectorNormalizeUDF, VectorSubUDF, VectorToBinaryUDF,
};

// -- Re-exports from aggregate --

pub use aggregate::{VectorAvgAccumulator, VectorAvgUDF, VectorSumAccumulator, VectorSumUDF};

// -- Re-exports from vector_stats --

pub use vector_stats::{
    CentroidAccumulator, CentroidUDF, VectorMaxAccumulator, VectorMaxUDF, VectorMedianAccumulator,
    VectorMedianUDF, VectorMinAccumulator, VectorMinUDF, VectorStddevAccumulator, VectorStddevUDF,
};

// -- Re-exports from text --

pub use text::{Bm25ScoreUDF, TfIdfUDF};

// -- Re-exports from sparse --

pub use sparse::{sparsevec_dims, sparsevec_nnz, SparseToVectorUDF, VectorToSparseUDF};

// -- Re-exports from json --

pub use json::{
    JsonContainsUDF, JsonExistsUDF, JsonExtractPathUDF, JsonPathExistsUDF, JsonPathQueryUDF,
    JsonTypeofUDF,
};

/// Returns all PostgreSQL `json` scalar UDFs for registration with DataFusion.
///
/// JSON is stored as text (not a decomposed binary representation), so these
/// use `json_*` names. `json_extract_path` / `json_extract_path_text` take a
/// variadic path; `json_contains` tests recursive containment (`@>`);
/// `json_exists` tests a top-level key/element; `json_typeof` names the type;
/// `json_path_exists` / `json_path_query` evaluate a jsonpath subset
/// (`$.a.b[0]`, `[*]`).
pub fn all_json_udfs() -> Vec<ScalarUDF> {
    vec![
        ScalarUDF::new_from_impl(JsonExtractPathUDF::new()),
        ScalarUDF::new_from_impl(JsonExtractPathUDF::text()),
        ScalarUDF::new_from_impl(JsonContainsUDF::new()),
        ScalarUDF::new_from_impl(JsonExistsUDF::new()),
        ScalarUDF::new_from_impl(JsonTypeofUDF::new()),
        ScalarUDF::new_from_impl(JsonPathExistsUDF::new()),
        ScalarUDF::new_from_impl(JsonPathQueryUDF::new()),
    ]
}

/// Returns all vector scalar UDFs for registration with DataFusion
pub fn all_vector_udfs() -> Vec<ScalarUDF> {
    vec![
        // Distance UDFs
        ScalarUDF::new_from_impl(L2DistUDF::new()),
        ScalarUDF::new_from_impl(CosineDistUDF::new()),
        ScalarUDF::new_from_impl(IPDistUDF::new()),
        ScalarUDF::new_from_impl(L1DistUDF::new()),
        ScalarUDF::new_from_impl(HammingDistUDF::new()),
        ScalarUDF::new_from_impl(JaccardDistUDF::new()),
        // Element-wise binary ops
        ScalarUDF::new_from_impl(VectorAddUDF::new()),
        ScalarUDF::new_from_impl(VectorSubUDF::new()),
        ScalarUDF::new_from_impl(VectorMulUDF::new()),
        ScalarUDF::new_from_impl(VectorConcatUDF::new()),
        // Utility UDFs
        ScalarUDF::new_from_impl(VectorDimsUDF::new()),
        ScalarUDF::new_from_impl(VectorNormUDF::new()),
        ScalarUDF::new_from_impl(VectorNormalizeUDF::new()),
        ScalarUDF::new_from_impl(BinaryQuantizeUDF::new()),
        ScalarUDF::new_from_impl(SubvectorUDF::new()),
        // Type casting UDFs
        ScalarUDF::new_from_impl(VectorToSparseUDF::new()),
        ScalarUDF::new_from_impl(SparseToVectorUDF::new()),
        ScalarUDF::new_from_impl(VectorToBinaryUDF::new()),
        // Text scoring UDFs
        ScalarUDF::new_from_impl(Bm25ScoreUDF::new()),
        ScalarUDF::new_from_impl(TfIdfUDF::new()),
    ]
}

/// Returns all vector aggregate UDFs for registration with DataFusion
pub fn all_vector_aggregates() -> Vec<AggregateUDF> {
    vec![
        AggregateUDF::new_from_impl(VectorSumUDF::new()),
        AggregateUDF::new_from_impl(VectorAvgUDF::new()),
        AggregateUDF::new_from_impl(CentroidUDF::new()),
        AggregateUDF::new_from_impl(VectorMinUDF::new()),
        AggregateUDF::new_from_impl(VectorMaxUDF::new()),
        AggregateUDF::new_from_impl(VectorStddevUDF::new()),
        AggregateUDF::new_from_impl(VectorMedianUDF::new()),
    ]
}
