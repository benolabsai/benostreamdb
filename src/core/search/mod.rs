// Copyright (c) 2026 Richard Albright. All rights reserved.

pub mod rrf;

pub use rrf::{ScoredResult, ReciprocalRankFusion};
use crate::core::planner::VectorSearchParams;
use anyhow::Result;

/// Coordinates multi-path search (Hybrid Search)
pub struct HybridSearchCoordinator {
    rrf: ReciprocalRankFusion,
}

impl HybridSearchCoordinator {
    pub fn new() -> Self {
        Self {
            rrf: ReciprocalRankFusion::default(),
        }
    }

    /// Execute search across multiple indexes and fuse results using RRF.
    /// This runs the different search paths (Vector, Keyword) in parallel.
    pub async fn execute_hybrid(
        &self,
        table: &crate::core::table::Table,
        _filter: Option<&str>,
        vector_params: Option<VectorSearchParams>,
        keyword_params: Option<KeywordSearchParams>,
        limit: usize,
    ) -> Result<Vec<ScoredResult>> {
        let mut search_handles = Vec::new();

        // 1. Vector Search Path
        if let Some(vp) = vector_params {
            let table_clone = table.clone();
            let handle = tokio::spawn(async move {
                table_clone.execute_vector_search_as_scored(vp).await
            });
            search_handles.push(handle);
        }

        // 2. Keyword Search Path (BM25)
        if let Some(kp) = keyword_params {
            let table_clone = table.clone();
            let handle = tokio::spawn(async move {
                table_clone.execute_keyword_search_as_scored(kp).await
            });
            search_handles.push(handle);
        }

        // Collect results
        let mut ranked_lists = Vec::new();
        for handle in search_handles {
            match handle.await {
                Ok(Ok(list)) => ranked_lists.push(list),
                Ok(Err(e)) => tracing::error!("Search path failed: {}", e),
                Err(e) => tracing::error!("Search task panicked: {}", e),
            }
        }

        // Fusion
        if ranked_lists.is_empty() {
            return Ok(Vec::new());
        }

        if ranked_lists.len() == 1 {
            let mut list: Vec<ScoredResult> = ranked_lists.pop().unwrap();
            list.truncate(limit);
            return Ok(list);
        }

        Ok(self.rrf.fuse(ranked_lists, limit))
    }
}

/// Parameters for BM25 Keyword Search
#[derive(Debug, Clone)]
pub struct KeywordSearchParams {
    pub column: String,
    pub query: String,
}
