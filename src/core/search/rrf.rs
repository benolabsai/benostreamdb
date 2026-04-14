// Copyright (c) 2026 Richard Albright. All rights reserved.

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// A single search result entry from an index path
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredResult {
    pub segment_id: String,
    pub row_id: u32,
    pub score: f32, // Distance for HNSW, BM25 score for Keyword
}

/// Reciprocal Rank Fusion (RRF) Implementation
/// Formula: score = \sum_{rankers} \frac{1}{k + rank}
pub struct ReciprocalRankFusion {
    k: f32,
}

impl ReciprocalRankFusion {
    pub fn new(k: f32) -> Self {
        Self { k }
    }

    pub fn default() -> Self {
        Self::new(60.0)
    }

    /// Fuse multiple ranked lists into a single ranked list.
    /// Input is a vector of ranked lists, where each list is already sorted by its internal score.
    pub fn fuse(&self, ranked_lists: Vec<Vec<ScoredResult>>, limit: usize) -> Vec<ScoredResult> {
        let mut fused_scores: HashMap<(String, u32), f32> = HashMap::new();

        for list in ranked_lists {
            for (rank, result) in list.iter().enumerate() {
                let key = (result.segment_id.clone(), result.row_id);
                // RRF Rank starts at 1
                let rrf_score = 1.0 / (self.k + (rank as f32 + 1.0));
                *fused_scores.entry(key).or_insert(0.0) += rrf_score;
            }
        }

        let mut results: Vec<ScoredResult> = fused_scores.into_iter()
            .map(|((segment_id, row_id), score)| ScoredResult {
                segment_id,
                row_id,
                score, // This is now the fused RRF score
            })
            .collect();

        // Sort by fused score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        if results.len() > limit {
            results.truncate(limit);
        }
        
        results
    }
}
