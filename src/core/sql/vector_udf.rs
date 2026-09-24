// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Vector UDF module — decomposed into submodules.
//!
//! Submodules:
//! - `distance`  — Distance UDFs (L2, Cosine, IP, L1, Hamming, Jaccard) and sparse distance helpers
//! - `transform` — Vector transform UDFs (add, sub, mul, concat, dims, norm, normalize, quantize, subvector, to_binary)
//! - `aggregate` — Aggregate UDFs (vector_sum, vector_avg) and accumulator types
//! - `sparse`    — Sparse vector conversion UDFs and sparse utility functions

pub mod distance;
pub mod transform;
pub mod aggregate;
pub mod sparse;

use datafusion::logical_expr::{ScalarUDF, AggregateUDF};

// -- Re-exports from distance --

pub use distance::{
    dense_to_sparse,
    sparse_to_dense,
    sparse_l2_distance,
    sparse_cosine_distance,
    sparse_inner_product_distance,
    L2DistUDF,
    CosineDistUDF,
    IPDistUDF,
    L1DistUDF,
    HammingDistUDF,
    JaccardDistUDF,
};

// -- Re-exports from transform --

pub use transform::{
    VectorAddUDF,
    VectorSubUDF,
    VectorMulUDF,
    VectorConcatUDF,
    VectorDimsUDF,
    VectorNormUDF,
    VectorNormalizeUDF,
    BinaryQuantizeUDF,
    SubvectorUDF,
    VectorToBinaryUDF,
};

// -- Re-exports from aggregate --

pub use aggregate::{
    VectorSumUDF,
    VectorAvgUDF,
    VectorSumAccumulator,
    VectorAvgAccumulator,
};

// -- Re-exports from sparse --

pub use sparse::{
    VectorToSparseUDF,
    SparseToVectorUDF,
    sparsevec_dims,
    sparsevec_nnz,
};

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
    ]
}

/// Returns all vector aggregate UDFs for registration with DataFusion
pub fn all_vector_aggregates() -> Vec<AggregateUDF> {
    vec![
        AggregateUDF::new_from_impl(VectorSumUDF::new()),
        AggregateUDF::new_from_impl(VectorAvgUDF::new()),
    ]
}
