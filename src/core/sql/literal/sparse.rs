// Copyright (c) 2026 Richard Albright. All rights reserved.

use datafusion::error::{DataFusionError, Result};

/// Parse a sparse vector literal string
///
/// # Arguments
/// * `input` - The sparse vector literal string, e.g., "{1:0.5, 10:0.3, 100:0.8}"
/// * `dim` - The total dimensionality of the sparse vector
///
/// # Returns
/// * `Result<crate::core::index::SparseVector>` - A SparseVector with indices, values, and dimension
///
/// # Examples
/// ```
/// use hyperstreamdb::core::sql::literal::sparse::parse_sparse_vector;
///
/// let result = parse_sparse_vector("{1:0.5, 10:0.3}", 1000).unwrap();
/// ```
pub fn parse_sparse_vector(input: &str, dim: usize) -> Result<crate::core::index::SparseVector> {
    let trimmed = input.trim();

    // Check for opening brace
    if !trimmed.starts_with('{') {
        return Err(DataFusionError::Plan(
            "Sparse vector literal must be enclosed in braces: '{...}'".to_string()
        ));
    }

    // Check for closing brace
    if !trimmed.ends_with('}') {
        return Err(DataFusionError::Plan(
            "Sparse vector literal must be enclosed in braces: '{...}'".to_string()
        ));
    }

    // Extract content between braces
    let content = &trimmed[1..trimmed.len() - 1].trim();

    // Handle empty sparse vector (all zeros)
    if content.is_empty() {
        return Ok(crate::core::index::SparseVector {
            indices: Vec::new(),
            values: Vec::new(),
            dim,
        });
    }

    // Parse index:value pairs
    let mut indices = Vec::new();
    let mut values = Vec::new();

    for (pair_idx, pair) in content.split(',').enumerate() {
        let pair = pair.trim();

        // Split by colon
        let parts: Vec<&str> = pair.split(':').collect();
        if parts.len() != 2 {
            return Err(DataFusionError::Plan(
                format!("Invalid sparse vector pair at position {}: expected 'index:value', got '{}'",
                    pair_idx, pair)
            ));
        }

        // Parse index
        let index = parts[0].trim().parse::<u32>().map_err(|_| {
            DataFusionError::Plan(
                format!("Invalid index at position {}: {}", pair_idx, parts[0])
            )
        })?;

        // Validate index is within bounds
        if index as usize >= dim {
            return Err(DataFusionError::Plan(
                format!("Index {} exceeds dimension {}", index, dim)
            ));
        }

        // Parse value
        let value = parts[1].trim().parse::<f32>().map_err(|_| {
            DataFusionError::Plan(
                format!("Invalid value at position {}: {}", pair_idx, parts[1])
            )
        })?;

        indices.push(index);
        values.push(value);
    }

    // Check for duplicate indices
    let mut sorted_indices = indices.clone();
    sorted_indices.sort_unstable();
    for i in 1..sorted_indices.len() {
        if sorted_indices[i] == sorted_indices[i - 1] {
            return Err(DataFusionError::Plan(
                format!("Duplicate index {} in sparse vector", sorted_indices[i])
            ));
        }
    }

    Ok(crate::core::index::SparseVector {
        indices,
        values,
        dim,
    })
}
