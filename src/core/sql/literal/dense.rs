// Copyright (c) 2026 Richard Albright. All rights reserved.

use arrow::array::Float32Array;
use arrow::datatypes::{DataType, Field};
use datafusion::error::{DataFusionError, Result};
use datafusion::scalar::ScalarValue;
use std::sync::Arc;

/// Parse a dense vector literal string into a DataFusion ScalarValue
///
/// Supports parsing vector literals in the format:
/// - `'[1,2,3]'::vector`
/// - `'[1.0, 2.0, 3.0]'::vector`
/// - `'[1,2,3]'::vector(3)`
///
/// Both integer and floating-point values are supported and converted to Float32.
///
/// # Arguments
/// * `input` - The vector literal string, e.g., ``[1,2,3]`` or ``[1.0, 2.0, 3.0]``
///
/// # Returns
/// * `Result<ScalarValue>` - A FixedSizeList ScalarValue containing Float32 elements
///
/// # Examples
/// ```
/// use hyperstreamdb::core::sql::literal::dense::parse_vector_literal;
///
/// let result = parse_vector_literal("[1,2,3]").unwrap();
/// let result = parse_vector_literal("[1.0, 2.0, 3.0]").unwrap();
/// ```
pub fn parse_vector_literal(input: &str) -> Result<ScalarValue> {
    let trimmed = input.trim();

    // Check for opening bracket
    if !trimmed.starts_with('[') {
        return Err(DataFusionError::Plan(
            "Vector literal must be enclosed in brackets: '[...]'".to_string(),
        ));
    }

    // Check for closing bracket
    if !trimmed.ends_with(']') {
        return Err(DataFusionError::Plan(
            "Vector literal must be enclosed in brackets: '[...]'".to_string(),
        ));
    }

    // Extract content between brackets
    let content = &trimmed[1..trimmed.len() - 1].trim();

    // Handle empty vector
    if content.is_empty() {
        return Err(DataFusionError::Plan(
            "Vector literal cannot be empty".to_string(),
        ));
    }

    // Split by comma and parse each element
    let mut values = Vec::new();
    for (idx, token) in content.split(',').enumerate() {
        let token = token.trim();

        // Try to parse as f32
        match token.parse::<f32>() {
            Ok(val) => values.push(val),
            Err(_) => {
                return Err(DataFusionError::Plan(format!(
                    "Invalid number at position {}: {}",
                    idx, token
                )));
            }
        }
    }

    // Create Float32Array from values
    let float_array = Float32Array::from(values.clone());
    let dimension = values.len() as i32;

    // Create FixedSizeListArray
    let field = Arc::new(Field::new("item", DataType::Float32, true));
    let fixed_size_list_array = arrow::array::FixedSizeListArray::try_new(
        field,
        dimension,
        Arc::new(float_array),
        None, // No null buffer
    )?;

    // Create FixedSizeList ScalarValue
    let scalar = ScalarValue::FixedSizeList(Arc::new(fixed_size_list_array));

    Ok(scalar)
}
