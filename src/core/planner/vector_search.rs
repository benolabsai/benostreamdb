// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Vector search parameters for query planning.

#[derive(Debug, Clone)]
pub struct VectorSearchParams {
    pub column: String,
    pub query: crate::core::index::VectorValue,
    pub k: usize,
    pub metric: crate::core::index::VectorMetric,
    pub ef_search: Option<usize>,
    pub probes: Option<usize>,
    /// Optimization: Metadata-only search (don't load vectors if stats alone guarantee match)
    pub stats_only: bool,
}

impl VectorSearchParams {
    /// Create new vector search parameters with default L2 metric.
    pub fn new(column: &str, query: crate::core::index::VectorValue, k: usize) -> Self {
        Self {
            column: column.to_string(),
            query,
            k,
            metric: crate::core::index::VectorMetric::L2,
            ef_search: None,
            probes: None,
            stats_only: false,
        }
    }

    /// Set the distance metric for vector search.
    pub fn with_metric(mut self, metric: crate::core::index::VectorMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Override the HNSW `ef_search` parameter (higher = more accurate, slower).
    pub fn with_ef_search(mut self, ef_search: usize) -> Self {
        self.ef_search = Some(ef_search);
        self
    }

    /// Override the IVF `probes` parameter (higher = more partitions searched, slower).
    pub fn with_probes(mut self, probes: usize) -> Self {
        self.probes = Some(probes);
        self
    }
}
