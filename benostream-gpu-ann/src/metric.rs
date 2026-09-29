// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
// Licensed under MIT OR Apache-2.0.

/// Vector distance metrics supported across all hardware backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Metric {
    /// Euclidean / L2 distance
    L2,
    /// Cosine distance (1.0 - cosine_similarity)
    Cosine,
    /// Inner product (dot product, negated for distance ranking where smaller = closer)
    InnerProduct,
    /// Manhattan / L1 distance
    L1,
    /// Hamming distance
    Hamming,
    /// Jaccard distance
    Jaccard,
}

impl Metric {
    /// Metric type ID used for uniform buffer configuration in shader pipelines.
    pub fn shader_type_id(&self) -> u32 {
        match self {
            Metric::L2 => 0,
            Metric::InnerProduct => 1,
            Metric::Cosine => 2,
            Metric::L1 => 3,
            Metric::Hamming => 4,
            Metric::Jaccard => 5,
        }
    }
}
