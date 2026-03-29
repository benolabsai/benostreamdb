// Copyright (c) 2026 Richard Albright. All rights reserved.

use datafusion::error::{DataFusionError, Result};
use datafusion::scalar::ScalarValue;
use arrow::array::Float32Array;
use arrow::datatypes::{DataType, Field};
use std::sync::Arc;

/// Validate that two vectors have compatible dimensions for binary operations
/// 
/// # Arguments
/// * `left_dim` - Dimension of the left operand
/// * `right_dim` - Dimension of the right operand
/// 
/// # Returns
/// * `Result<()>` - Ok if dimensions match, error otherwise
pub fn validate_vector_dimensions(left_dim: usize, right_dim: usize) -> Result<()> {
    if left_dim != right_dim {
        return Err(DataFusionError::Plan(
            format!("Vector dimension mismatch: expected {}, got {}", left_dim, right_dim)
        ));
    }
    Ok(())
}

/// Format a binary vector for display
/// 
/// Supports two output formats:
/// - Binary string: "10110101..." (default)
/// - Hex string: "0xB5..." (when use_hex is true)
/// 
/// # Arguments
/// * `data` - The byte array containing the binary vector
/// * `bits` - The number of bits in the vector (may be less than data.len() * 8)
/// * `use_hex` - If true, format as hex; otherwise format as binary string
/// 
/// # Returns
/// * `String` - The formatted binary vector
/// 
/// # Examples
/// ```
/// use hyperstreamdb::core::sql::vector_literal::format_binary_vector;
/// 
/// let bytes = vec![0b10110101];
/// let binary_str = format_binary_vector(&bytes, 8, false); // "10110101"
/// let hex_str = format_binary_vector(&bytes, 8, true);     // "0xB5"
/// ```
pub fn format_binary_vector(data: &[u8], bits: usize, use_hex: bool) -> String {
    if use_hex {
        // Format as hex string
        let mut result = String::from("0x");
        let byte_count = (bits + 7) / 8;
        for i in 0..byte_count.min(data.len()) {
            result.push_str(&format!("{:02X}", data[i]));
        }
        result
    } else {
        // Format as binary string
        let mut result = String::new();
        for bit_idx in 0..bits {
            let byte_idx = bit_idx / 8;
            if byte_idx >= data.len() {
                break;
            }
            let bit_pos = 7 - (bit_idx % 8); // MSB first
            let bit = (data[byte_idx] >> bit_pos) & 1;
            result.push(if bit == 1 { '1' } else { '0' });
        }
        result
    }
}

/// Parser for pgvector-compatible vector literals
/// 
/// Supports parsing dense vector literals in the format:
/// - `'[1,2,3]'::vector`
/// - `'[1.0, 2.0, 3.0]'::vector`
/// - `'[1,2,3]'::vector(3)`
/// 
/// Both integer and floating-point values are supported and converted to Float32.
#[derive(Debug)]
pub struct VectorLiteralParser;

impl VectorLiteralParser {
    /// Parse a binary vector literal string
    /// 
    /// Supports two formats:
    /// - Binary string: `B'10110101'` or `b'10110101'`
    /// - Hex string: `'\\xB5'` (with dimension specification)
    /// 
    /// # Arguments
    /// * `input` - The binary literal string
    /// * `expected_bits` - Optional expected number of bits for validation
    /// 
    /// # Returns
    /// * `Result<Vec<u8>>` - A byte array representation of the binary vector
    /// 
    /// # Examples
    /// ```
    /// use hyperstreamdb::core::sql::vector_literal::VectorLiteralParser;
    /// 
    /// let result = VectorLiteralParser::parse_binary("B'10110101'", Some(8)).unwrap();
    /// let result = VectorLiteralParser::parse_binary("'\\xB5'", Some(8)).unwrap();
    /// ```
    pub fn parse_binary(input: &str, expected_bits: Option<usize>) -> Result<Vec<u8>> {
        let trimmed = input.trim();
        
        // Check for binary string format: B'...' or b'...'
        if trimmed.starts_with("B'") || trimmed.starts_with("b'") {
            if !trimmed.ends_with('\'') {
                return Err(DataFusionError::Plan(
                    "Binary literal must end with single quote: B'...'".to_string()
                ));
            }
            
            // Extract binary string between quotes
            let binary_str = &trimmed[2..trimmed.len() - 1];
            
            // Validate that all characters are 0 or 1
            for (idx, ch) in binary_str.chars().enumerate() {
                if ch != '0' && ch != '1' {
                    return Err(DataFusionError::Plan(
                        format!("Invalid binary digit at position {}: expected '0' or '1', got '{}'", idx, ch)
                    ));
                }
            }
            
            let bit_count = binary_str.len();
            
            // Validate bit count if expected_bits is provided
            if let Some(expected) = expected_bits {
                if bit_count != expected {
                    return Err(DataFusionError::Plan(
                        format!("Binary literal bit count mismatch: expected {} bits, got {}", expected, bit_count)
                    ));
                }
            }
            
            // Convert binary string to bytes
            let byte_count = (bit_count + 7) / 8;
            let mut bytes = vec![0u8; byte_count];
            
            for (bit_idx, ch) in binary_str.chars().enumerate() {
                if ch == '1' {
                    let byte_idx = bit_idx / 8;
                    let bit_pos = 7 - (bit_idx % 8); // MSB first
                    bytes[byte_idx] |= 1 << bit_pos;
                }
            }
            
            Ok(bytes)
        }
        // Check for hex string format: '\x...'
        else if trimmed.starts_with("'\\x") || trimmed.starts_with("'\\X") {
            if !trimmed.ends_with('\'') {
                return Err(DataFusionError::Plan(
                    "Hex literal must end with single quote: '\\x...'".to_string()
                ));
            }
            
            // Extract hex string between \x and closing quote
            let hex_str = &trimmed[3..trimmed.len() - 1];
            
            // Parse hex string to bytes
            let mut bytes = Vec::new();
            let mut chars = hex_str.chars().peekable();
            
            while chars.peek().is_some() {
                let high = chars.next().ok_or_else(|| {
                    DataFusionError::Plan("Incomplete hex byte".to_string())
                })?;
                let low = chars.next().ok_or_else(|| {
                    DataFusionError::Plan("Incomplete hex byte: hex digits must come in pairs".to_string())
                })?;
                
                let hex_byte = format!("{}{}", high, low);
                let byte = u8::from_str_radix(&hex_byte, 16).map_err(|_| {
                    DataFusionError::Plan(
                        format!("Invalid hex digit in '{}': expected 0-9, a-f, A-F", hex_byte)
                    )
                })?;
                
                bytes.push(byte);
            }
            
            // Validate bit count if expected_bits is provided
            if let Some(expected) = expected_bits {
                let actual_bits = bytes.len() * 8;
                if actual_bits != expected {
                    return Err(DataFusionError::Plan(
                        format!("Binary literal bit count mismatch: expected {} bits, got {}", expected, actual_bits)
                    ));
                }
            }
            
            Ok(bytes)
        }
        else {
            Err(DataFusionError::Plan(
                "Binary literal must be in format B'...' or '\\x...'".to_string()
            ))
        }
    }

    /// Parse a vector literal string into a DataFusion ScalarValue
    /// 
    /// # Arguments
    /// * `input` - The vector literal string, e.g., "[1,2,3]" or "[1.0, 2.0, 3.0]"
    /// 
    /// # Returns
    /// * `Result<ScalarValue>` - A FixedSizeList ScalarValue containing Float32 elements
    /// 
    /// # Examples
    /// ```
    /// use hyperstreamdb::core::sql::vector_literal::VectorLiteralParser;
    /// 
    /// let result = VectorLiteralParser::parse("[1,2,3]").unwrap();
    /// let result = VectorLiteralParser::parse("[1.0, 2.0, 3.0]").unwrap();
    /// ```
    pub fn parse(input: &str) -> Result<ScalarValue> {
        let trimmed = input.trim();
        
        // Check for opening bracket
        if !trimmed.starts_with('[') {
            return Err(DataFusionError::Plan(
                "Vector literal must be enclosed in brackets: '[...]'".to_string()
            ));
        }
        
        // Check for closing bracket
        if !trimmed.ends_with(']') {
            return Err(DataFusionError::Plan(
                "Vector literal must be enclosed in brackets: '[...]'".to_string()
            ));
        }
        
        // Extract content between brackets
        let content = &trimmed[1..trimmed.len() - 1].trim();
        
        // Handle empty vector
        if content.is_empty() {
            return Err(DataFusionError::Plan(
                "Vector literal cannot be empty".to_string()
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
                    return Err(DataFusionError::Plan(
                        format!("Invalid number at position {}: {}", idx, token)
                    ));
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
    /// use hyperstreamdb::core::sql::vector_literal::VectorLiteralParser;
    /// 
    /// let result = VectorLiteralParser::parse_sparse("{1:0.5, 10:0.3}", 1000).unwrap();
    /// ```
    pub fn parse_sparse(input: &str, dim: usize) -> Result<crate::core::index::SparseVector> {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Array;
    
    #[test]
    fn test_parse_integer_vector() {
        let result = VectorLiteralParser::parse("[1,2,3]");
        assert!(result.is_ok());
        
        let scalar = result.unwrap();
        // Verify it's a FixedSizeList
        match scalar {
            ScalarValue::FixedSizeList(arr) => {
                assert_eq!(arr.len(), 1); // Single scalar value
                assert_eq!(arr.value_length(), 3); // 3 elements
            }
            _ => panic!("Expected FixedSizeList"),
        }
    }
    
    #[test]
    fn test_parse_float_vector() {
        let result = VectorLiteralParser::parse("[1.0, 2.5, 3.7]");
        assert!(result.is_ok());
        
        let scalar = result.unwrap();
        match scalar {
            ScalarValue::FixedSizeList(arr) => {
                assert_eq!(arr.value_length(), 3);
            }
            _ => panic!("Expected FixedSizeList"),
        }
    }
    
    #[test]
    fn test_parse_mixed_vector() {
        let result = VectorLiteralParser::parse("[1, 2.5, 3]");
        assert!(result.is_ok());
        
        let scalar = result.unwrap();
        match scalar {
            ScalarValue::FixedSizeList(arr) => {
                assert_eq!(arr.value_length(), 3);
            }
            _ => panic!("Expected FixedSizeList"),
        }
    }
    
    #[test]
    fn test_parse_with_spaces() {
        let result = VectorLiteralParser::parse("[ 1 , 2 , 3 ]");
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_parse_single_element() {
        let result = VectorLiteralParser::parse("[42]");
        assert!(result.is_ok());
        
        let scalar = result.unwrap();
        match scalar {
            ScalarValue::FixedSizeList(arr) => {
                assert_eq!(arr.value_length(), 1);
            }
            _ => panic!("Expected FixedSizeList"),
        }
    }
    
    #[test]
    fn test_parse_missing_opening_bracket() {
        let result = VectorLiteralParser::parse("1,2,3]");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("must be enclosed in brackets"));
    }
    
    #[test]
    fn test_parse_missing_closing_bracket() {
        let result = VectorLiteralParser::parse("[1,2,3");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("must be enclosed in brackets"));
    }
    
    #[test]
    fn test_parse_empty_vector() {
        let result = VectorLiteralParser::parse("[]");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot be empty"));
    }
    
    #[test]
    fn test_parse_invalid_number() {
        let result = VectorLiteralParser::parse("[1, abc, 3]");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid number at position"));
    }
    
    #[test]
    fn test_parse_negative_numbers() {
        let result = VectorLiteralParser::parse("[-1, -2.5, -3]");
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_parse_scientific_notation() {
        let result = VectorLiteralParser::parse("[1e-3, 2.5e2, 3.0]");
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_parse_special_float_values() {
        // NaN should parse successfully (f32::parse handles it)
        let result = VectorLiteralParser::parse("[NaN, 1.0, 2.0]");
        assert!(result.is_ok());
        
        // Infinity should parse successfully
        let result = VectorLiteralParser::parse("[inf, 1.0, 2.0]");
        assert!(result.is_ok());
        
        // Negative infinity should parse successfully
        let result = VectorLiteralParser::parse("[-inf, 1.0, 2.0]");
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_parse_multiple_errors() {
        // Missing both brackets
        let result = VectorLiteralParser::parse("1,2,3");
        assert!(result.is_err());
        
        // Invalid characters
        let result = VectorLiteralParser::parse("[1, 2, abc, def]");
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Invalid number at position"));
    }
    
    #[test]
    fn test_parse_whitespace_only() {
        let result = VectorLiteralParser::parse("[   ]");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot be empty"));
    }
    
    #[test]
    fn test_parse_trailing_comma() {
        // Trailing comma creates an empty token which should fail
        let result = VectorLiteralParser::parse("[1, 2, 3,]");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid number"));
    }
    
    #[test]
    fn test_parse_leading_comma() {
        // Leading comma creates an empty token which should fail
        let result = VectorLiteralParser::parse("[,1, 2, 3]");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid number"));
    }
    
    #[test]
    fn test_parse_double_comma() {
        // Double comma creates an empty token which should fail
        let result = VectorLiteralParser::parse("[1,, 2, 3]");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid number"));
    }
    
    #[test]
    fn test_parse_very_large_numbers() {
        // Very large numbers should parse as infinity
        let result = VectorLiteralParser::parse("[1e308, 1e309, 1.0]");
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_parse_very_small_numbers() {
        // Very small numbers should parse as zero or subnormal
        let result = VectorLiteralParser::parse("[1e-45, 1e-50, 1.0]");
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_validate_dimensions_match() {
        let result = validate_vector_dimensions(3, 3);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_validate_dimensions_mismatch() {
        let result = validate_vector_dimensions(3, 5);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("dimension mismatch"));
        assert!(err_msg.contains("expected 3"));
        assert!(err_msg.contains("got 5"));
    }
    
    #[test]
    fn test_validate_dimensions_zero() {
        // Zero dimensions should match
        let result = validate_vector_dimensions(0, 0);
        assert!(result.is_ok());
        
        // Zero vs non-zero should fail
        let result = validate_vector_dimensions(0, 5);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_validate_dimensions_large() {
        // Large dimensions should work
        let result = validate_vector_dimensions(2048, 2048);
        assert!(result.is_ok());
        
        let result = validate_vector_dimensions(2048, 2049);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_parse_sparse_vector_basic() {
        let result = VectorLiteralParser::parse_sparse("{1:0.5, 10:0.3, 100:0.8}", 1000);
        assert!(result.is_ok());
        
        let sparse = result.unwrap();
        assert_eq!(sparse.dim, 1000);
        assert_eq!(sparse.indices.len(), 3);
        assert_eq!(sparse.values.len(), 3);
        assert_eq!(sparse.indices, vec![1, 10, 100]);
        assert_eq!(sparse.values, vec![0.5, 0.3, 0.8]);
    }
    
    #[test]
    fn test_parse_sparse_vector_empty() {
        let result = VectorLiteralParser::parse_sparse("{}", 1000);
        assert!(result.is_ok());
        
        let sparse = result.unwrap();
        assert_eq!(sparse.dim, 1000);
        assert_eq!(sparse.indices.len(), 0);
        assert_eq!(sparse.values.len(), 0);
    }
    
    #[test]
    fn test_parse_sparse_vector_single_element() {
        let result = VectorLiteralParser::parse_sparse("{42:1.5}", 100);
        assert!(result.is_ok());
        
        let sparse = result.unwrap();
        assert_eq!(sparse.dim, 100);
        assert_eq!(sparse.indices, vec![42]);
        assert_eq!(sparse.values, vec![1.5]);
    }
    
    #[test]
    fn test_parse_sparse_vector_with_spaces() {
        let result = VectorLiteralParser::parse_sparse("{ 1 : 0.5 , 10 : 0.3 }", 1000);
        assert!(result.is_ok());
        
        let sparse = result.unwrap();
        assert_eq!(sparse.indices, vec![1, 10]);
        assert_eq!(sparse.values, vec![0.5, 0.3]);
    }
    
    #[test]
    fn test_parse_sparse_vector_missing_opening_brace() {
        let result = VectorLiteralParser::parse_sparse("1:0.5, 10:0.3}", 1000);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("must be enclosed in braces"));
    }
    
    #[test]
    fn test_parse_sparse_vector_missing_closing_brace() {
        let result = VectorLiteralParser::parse_sparse("{1:0.5, 10:0.3", 1000);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("must be enclosed in braces"));
    }
    
    #[test]
    fn test_parse_sparse_vector_invalid_format() {
        // Missing colon
        let result = VectorLiteralParser::parse_sparse("{1 0.5}", 1000);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("expected 'index:value'"));
        
        // Multiple colons
        let result = VectorLiteralParser::parse_sparse("{1:0.5:extra}", 1000);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("expected 'index:value'"));
    }
    
    #[test]
    fn test_parse_sparse_vector_invalid_index() {
        let result = VectorLiteralParser::parse_sparse("{abc:0.5}", 1000);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid index"));
    }
    
    #[test]
    fn test_parse_sparse_vector_invalid_value() {
        let result = VectorLiteralParser::parse_sparse("{1:abc}", 1000);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid value"));
    }
    
    #[test]
    fn test_parse_sparse_vector_index_out_of_bounds() {
        let result = VectorLiteralParser::parse_sparse("{1000:0.5}", 1000);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exceeds dimension"));
    }
    
    #[test]
    fn test_parse_sparse_vector_duplicate_indices() {
        let result = VectorLiteralParser::parse_sparse("{1:0.5, 10:0.3, 1:0.8}", 1000);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Duplicate index"));
    }
    
    #[test]
    fn test_parse_sparse_vector_negative_values() {
        let result = VectorLiteralParser::parse_sparse("{1:-0.5, 10:-0.3}", 1000);
        assert!(result.is_ok());
        
        let sparse = result.unwrap();
        assert_eq!(sparse.values, vec![-0.5, -0.3]);
    }
    
    #[test]
    fn test_parse_sparse_vector_zero_values() {
        // Zero values are allowed (though not efficient for sparse representation)
        let result = VectorLiteralParser::parse_sparse("{1:0.0, 10:0.5}", 1000);
        assert!(result.is_ok());
        
        let sparse = result.unwrap();
        assert_eq!(sparse.values, vec![0.0, 0.5]);
    }
    
    #[test]
    fn test_parse_binary_basic() {
        let result = VectorLiteralParser::parse_binary("B'10110101'", Some(8));
        assert!(result.is_ok());
        
        let bytes = result.unwrap();
        assert_eq!(bytes.len(), 1);
        assert_eq!(bytes[0], 0b10110101);
    }
    
    #[test]
    fn test_parse_binary_lowercase() {
        let result = VectorLiteralParser::parse_binary("b'10110101'", Some(8));
        assert!(result.is_ok());
        
        let bytes = result.unwrap();
        assert_eq!(bytes[0], 0b10110101);
    }
    
    #[test]
    fn test_parse_binary_multiple_bytes() {
        let result = VectorLiteralParser::parse_binary("B'1011010110101100'", Some(16));
        assert!(result.is_ok());
        
        let bytes = result.unwrap();
        assert_eq!(bytes.len(), 2);
        assert_eq!(bytes[0], 0b10110101);
        assert_eq!(bytes[1], 0b10101100);
    }
    
    #[test]
    fn test_parse_binary_partial_byte() {
        // 5 bits should still create 1 byte
        let result = VectorLiteralParser::parse_binary("B'10110'", Some(5));
        assert!(result.is_ok());
        
        let bytes = result.unwrap();
        assert_eq!(bytes.len(), 1);
        // 10110 padded with zeros: 10110000
        assert_eq!(bytes[0], 0b10110000);
    }
    
    #[test]
    fn test_parse_binary_hex_format() {
        let result = VectorLiteralParser::parse_binary("'\\xB5'", Some(8));
        assert!(result.is_ok());
        
        let bytes = result.unwrap();
        assert_eq!(bytes.len(), 1);
        assert_eq!(bytes[0], 0xB5);
    }
    
    #[test]
    fn test_parse_binary_hex_uppercase() {
        let result = VectorLiteralParser::parse_binary("'\\XFF'", Some(8));
        assert!(result.is_ok());
        
        let bytes = result.unwrap();
        assert_eq!(bytes[0], 0xFF);
    }
    
    #[test]
    fn test_parse_binary_hex_multiple_bytes() {
        let result = VectorLiteralParser::parse_binary("'\\xB5AC'", Some(16));
        assert!(result.is_ok());
        
        let bytes = result.unwrap();
        assert_eq!(bytes.len(), 2);
        assert_eq!(bytes[0], 0xB5);
        assert_eq!(bytes[1], 0xAC);
    }
    
    #[test]
    fn test_parse_binary_hex_lowercase() {
        let result = VectorLiteralParser::parse_binary("'\\xb5ac'", Some(16));
        assert!(result.is_ok());
        
        let bytes = result.unwrap();
        assert_eq!(bytes[0], 0xB5);
        assert_eq!(bytes[1], 0xAC);
    }
    
    #[test]
    fn test_parse_binary_invalid_digit() {
        let result = VectorLiteralParser::parse_binary("B'10210'", Some(5));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid binary digit"));
    }
    
    #[test]
    fn test_parse_binary_missing_closing_quote() {
        let result = VectorLiteralParser::parse_binary("B'10110101", Some(8));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("must end with single quote"));
    }
    
    #[test]
    fn test_parse_binary_bit_count_mismatch() {
        let result = VectorLiteralParser::parse_binary("B'10110101'", Some(16));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("bit count mismatch"));
    }
    
    #[test]
    fn test_parse_binary_hex_invalid_digit() {
        let result = VectorLiteralParser::parse_binary("'\\xGH'", Some(8));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid hex digit"));
    }
    
    #[test]
    fn test_parse_binary_hex_incomplete_byte() {
        let result = VectorLiteralParser::parse_binary("'\\xB'", Some(8));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Incomplete hex byte"));
    }
    
    #[test]
    fn test_parse_binary_invalid_format() {
        let result = VectorLiteralParser::parse_binary("10110101", Some(8));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("must be in format"));
    }
    
    #[test]
    fn test_parse_binary_no_expected_bits() {
        // Should work without expected_bits validation
        let result = VectorLiteralParser::parse_binary("B'10110101'", None);
        assert!(result.is_ok());
        
        let bytes = result.unwrap();
        assert_eq!(bytes[0], 0b10110101);
    }
    
    #[test]
    fn test_format_binary_vector_binary_string() {
        let bytes = vec![0b10110101];
        let result = format_binary_vector(&bytes, 8, false);
        assert_eq!(result, "10110101");
    }
    
    #[test]
    fn test_format_binary_vector_hex_string() {
        let bytes = vec![0xB5];
        let result = format_binary_vector(&bytes, 8, true);
        assert_eq!(result, "0xB5");
    }
    
    #[test]
    fn test_format_binary_vector_multiple_bytes_binary() {
        let bytes = vec![0b10110101, 0b10101100];
        let result = format_binary_vector(&bytes, 16, false);
        assert_eq!(result, "1011010110101100");
    }
    
    #[test]
    fn test_format_binary_vector_multiple_bytes_hex() {
        let bytes = vec![0xB5, 0xAC];
        let result = format_binary_vector(&bytes, 16, true);
        assert_eq!(result, "0xB5AC");
    }
    
    #[test]
    fn test_format_binary_vector_partial_byte() {
        let bytes = vec![0b10110000];
        let result = format_binary_vector(&bytes, 5, false);
        assert_eq!(result, "10110");
    }
    
    #[test]
    fn test_format_binary_vector_all_zeros() {
        let bytes = vec![0x00];
        let result = format_binary_vector(&bytes, 8, false);
        assert_eq!(result, "00000000");
        
        let result = format_binary_vector(&bytes, 8, true);
        assert_eq!(result, "0x00");
    }
    
    #[test]
    fn test_format_binary_vector_all_ones() {
        let bytes = vec![0xFF];
        let result = format_binary_vector(&bytes, 8, false);
        assert_eq!(result, "11111111");
        
        let result = format_binary_vector(&bytes, 8, true);
        assert_eq!(result, "0xFF");
    }
    
    #[test]
    fn test_binary_round_trip() {
        // Parse and format should be inverse operations
        let original = "B'10110101'";
        let bytes = VectorLiteralParser::parse_binary(original, Some(8)).unwrap();
        let formatted = format_binary_vector(&bytes, 8, false);
        assert_eq!(formatted, "10110101");
        
        // Hex round trip
        let original_hex = "'\\xB5'";
        let bytes_hex = VectorLiteralParser::parse_binary(original_hex, Some(8)).unwrap();
        let formatted_hex = format_binary_vector(&bytes_hex, 8, true);
        assert_eq!(formatted_hex, "0xB5");
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;
    use arrow::array::Float32Array;

    // Feature: pgvector-sql-support, Property 8: Vector Literal Parsing
    // **Validates: Requirements 4.1, 4.2, 4.3**
    //
    // Property: For any valid vector literal string in the format '[n1, n2, ..., nk]'::vector,
    // the parser should produce a FixedSizeList scalar value with Float32 elements matching
    // the input values.
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_vector_literal_parsing_correctness(
            // Generate vectors with dimensions from 1 to 128
            values in prop::collection::vec(
                // Generate f32 values in a reasonable range, avoiding special values
                prop::num::f32::NORMAL,
                1..=128
            )
        ) {
            // Filter out NaN and infinite values to ensure valid input
            let valid_values: Vec<f32> = values.iter()
                .filter(|v| v.is_finite())
                .copied()
                .collect();
            
            // Skip if we filtered out all values
            if valid_values.is_empty() {
                return Ok(());
            }

            // Construct the vector literal string
            let literal = format!("[{}]", 
                valid_values.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );

            // Parse the literal
            let result = VectorLiteralParser::parse(&literal);
            prop_assert!(result.is_ok(), "Failed to parse valid literal: {}", literal);

            let scalar = result.unwrap();

            // Verify it's a FixedSizeList
            match scalar {
                ScalarValue::FixedSizeList(arr) => {
                    // Verify the dimension matches
                    prop_assert_eq!(arr.value_length() as usize, valid_values.len(),
                        "Dimension mismatch: expected {}, got {}", 
                        valid_values.len(), arr.value_length());

                    // Extract the Float32Array from the FixedSizeList
                    let float_array = arr.value(0);
                    let float_array = float_array.as_any()
                        .downcast_ref::<Float32Array>()
                        .expect("Expected Float32Array");

                    // Verify each value matches (within floating point precision)
                    prop_assert_eq!(float_array.len(), valid_values.len(),
                        "Array length mismatch");

                    for (i, expected) in valid_values.iter().enumerate() {
                        let actual = float_array.value(i);
                        // Use relative comparison for floating point
                        let diff = (actual - expected).abs();
                        let tolerance = expected.abs() * 1e-6 + 1e-9;
                        prop_assert!(diff <= tolerance,
                            "Value mismatch at index {}: expected {}, got {}, diff {}",
                            i, expected, actual, diff);
                    }
                }
                _ => {
                    return Err(proptest::test_runner::TestCaseError::fail(
                        "Expected FixedSizeList scalar value"
                    ));
                }
            }
        }

        #[test]
        fn test_vector_literal_with_various_formats(
            // Test different formatting styles
            values in prop::collection::vec(
                prop::num::f32::NORMAL,
                1..=32
            ),
            // Random spacing patterns
            space_after_bracket in prop::bool::ANY,
            space_before_bracket in prop::bool::ANY,
            space_after_comma in prop::bool::ANY,
            space_before_comma in prop::bool::ANY,
        ) {
            // Filter out special values
            let valid_values: Vec<f32> = values.iter()
                .filter(|v| v.is_finite())
                .copied()
                .collect();
            
            if valid_values.is_empty() {
                return Ok(());
            }

            // Build literal with various spacing
            let mut literal = String::from("[");
            if space_after_bracket {
                literal.push(' ');
            }

            for (i, v) in valid_values.iter().enumerate() {
                if i > 0 {
                    if space_before_comma {
                        literal.push(' ');
                    }
                    literal.push(',');
                    if space_after_comma {
                        literal.push(' ');
                    }
                }
                literal.push_str(&v.to_string());
            }

            if space_before_bracket {
                literal.push(' ');
            }
            literal.push(']');

            // Parse should succeed regardless of spacing
            let result = VectorLiteralParser::parse(&literal);
            prop_assert!(result.is_ok(), 
                "Failed to parse literal with spacing: {}", literal);

            // Verify dimension is correct
            if let ScalarValue::FixedSizeList(arr) = result.unwrap() {
                prop_assert_eq!(arr.value_length() as usize, valid_values.len());
            }
        }

        #[test]
        fn test_vector_literal_with_integer_values(
            // Test that integers are properly converted to Float32
            values in prop::collection::vec(
                -1000i32..=1000i32,
                1..=64
            )
        ) {
            // Construct literal with integer values
            let literal = format!("[{}]", 
                values.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );

            let result = VectorLiteralParser::parse(&literal);
            prop_assert!(result.is_ok(), "Failed to parse integer literal: {}", literal);

            // Verify values are stored as Float32
            if let ScalarValue::FixedSizeList(arr) = result.unwrap() {
                let float_array = arr.value(0);
                let float_array = float_array.as_any()
                    .downcast_ref::<Float32Array>()
                    .expect("Expected Float32Array");

                for (i, expected) in values.iter().enumerate() {
                    let actual = float_array.value(i);
                    prop_assert_eq!(actual, *expected as f32,
                        "Integer conversion mismatch at index {}", i);
                }
            }
        }

        #[test]
        fn test_vector_literal_with_mixed_types(
            // Test mixing integers and floats
            int_values in prop::collection::vec(-100i32..=100i32, 1..=16),
            float_values in prop::collection::vec(
                prop::num::f32::NORMAL,
                1..=16
            )
        ) {
            // Filter floats
            let valid_floats: Vec<f32> = float_values.iter()
                .filter(|v| v.is_finite())
                .copied()
                .collect();
            
            if valid_floats.is_empty() {
                return Ok(());
            }

            // Interleave integers and floats
            let mut mixed_values = Vec::new();
            let mut literal_parts = Vec::new();
            
            for (i, int_val) in int_values.iter().enumerate() {
                mixed_values.push(*int_val as f32);
                literal_parts.push(int_val.to_string());
                
                if i < valid_floats.len() {
                    mixed_values.push(valid_floats[i]);
                    literal_parts.push(valid_floats[i].to_string());
                }
            }

            let literal = format!("[{}]", literal_parts.join(","));

            let result = VectorLiteralParser::parse(&literal);
            prop_assert!(result.is_ok(), "Failed to parse mixed literal: {}", literal);

            // Verify all values are correctly parsed
            if let ScalarValue::FixedSizeList(arr) = result.unwrap() {
                prop_assert_eq!(arr.value_length() as usize, mixed_values.len());
                
                let float_array = arr.value(0);
                let float_array = float_array.as_any()
                    .downcast_ref::<Float32Array>()
                    .expect("Expected Float32Array");

                for (i, expected) in mixed_values.iter().enumerate() {
                    let actual = float_array.value(i);
                    let diff = (actual - expected).abs();
                    let tolerance = expected.abs() * 1e-6 + 1e-9;
                    prop_assert!(diff <= tolerance,
                        "Mixed value mismatch at index {}: expected {}, got {}",
                        i, expected, actual);
                }
            }
        }
    }

    // Feature: pgvector-sql-support, Property 9: Vector Literal Error Handling
    // **Validates: Requirements 4.4**
    //
    // Property: For any malformed vector literal (missing brackets, invalid numbers, etc.),
    // the parser should return a descriptive error message indicating the specific parsing failure.
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_vector_literal_missing_opening_bracket(
            values in prop::collection::vec(
                prop::num::f32::NORMAL,
                1..=32
            )
        ) {
            let valid_values: Vec<f32> = values.iter()
                .filter(|v| v.is_finite())
                .copied()
                .collect();
            
            if valid_values.is_empty() {
                return Ok(());
            }

            // Create literal without opening bracket
            let literal = format!("{}]", 
                valid_values.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );

            let result = VectorLiteralParser::parse(&literal);
            prop_assert!(result.is_err(), "Should fail for missing opening bracket");
            
            let err_msg = result.unwrap_err().to_string();
            prop_assert!(err_msg.contains("must be enclosed in brackets"),
                "Error message should mention brackets, got: {}", err_msg);
        }

        #[test]
        fn test_vector_literal_missing_closing_bracket(
            values in prop::collection::vec(
                prop::num::f32::NORMAL,
                1..=32
            )
        ) {
            let valid_values: Vec<f32> = values.iter()
                .filter(|v| v.is_finite())
                .copied()
                .collect();
            
            if valid_values.is_empty() {
                return Ok(());
            }

            // Create literal without closing bracket
            let literal = format!("[{}", 
                valid_values.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );

            let result = VectorLiteralParser::parse(&literal);
            prop_assert!(result.is_err(), "Should fail for missing closing bracket");
            
            let err_msg = result.unwrap_err().to_string();
            prop_assert!(err_msg.contains("must be enclosed in brackets"),
                "Error message should mention brackets, got: {}", err_msg);
        }

        #[test]
        fn test_vector_literal_invalid_token(
            valid_values in prop::collection::vec(
                prop::num::f32::NORMAL,
                1..=16
            ),
            invalid_token in "[a-zA-Z]{1,10}",
            insert_position in 0..=16usize,
        ) {
            let valid_values: Vec<f32> = valid_values.iter()
                .filter(|v| v.is_finite())
                .copied()
                .collect();
            
            if valid_values.is_empty() {
                return Ok(());
            }

            // Insert invalid token at random position
            let mut parts: Vec<String> = valid_values.iter()
                .map(|v| v.to_string())
                .collect();
            
            let pos = insert_position.min(parts.len());
            parts.insert(pos, invalid_token.clone());

            let literal = format!("[{}]", parts.join(","));

            let result = VectorLiteralParser::parse(&literal);
            prop_assert!(result.is_err(), 
                "Should fail for invalid token: {}", literal);
            
            let err_msg = result.unwrap_err().to_string();
            prop_assert!(err_msg.contains("Invalid number"),
                "Error message should mention invalid number, got: {}", err_msg);
            prop_assert!(err_msg.contains(&format!("position {}", pos)),
                "Error message should mention position, got: {}", err_msg);
        }

        #[test]
        fn test_vector_literal_empty_brackets(
            // Test various whitespace patterns in empty brackets
            whitespace in prop::string::string_regex("[ \t\n\r]*").unwrap()
        ) {
            let literal = format!("[{}]", whitespace);

            let result = VectorLiteralParser::parse(&literal);
            prop_assert!(result.is_err(), "Should fail for empty vector");
            
            let err_msg = result.unwrap_err().to_string();
            prop_assert!(err_msg.contains("cannot be empty"),
                "Error message should mention empty vector, got: {}", err_msg);
        }

        #[test]
        fn test_vector_literal_trailing_comma(
            values in prop::collection::vec(
                prop::num::f32::NORMAL,
                1..=32
            )
        ) {
            let valid_values: Vec<f32> = values.iter()
                .filter(|v| v.is_finite())
                .copied()
                .collect();
            
            if valid_values.is_empty() {
                return Ok(());
            }

            // Create literal with trailing comma
            let literal = format!("[{},]", 
                valid_values.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );

            let result = VectorLiteralParser::parse(&literal);
            prop_assert!(result.is_err(), 
                "Should fail for trailing comma: {}", literal);
            
            let err_msg = result.unwrap_err().to_string();
            prop_assert!(err_msg.contains("Invalid number"),
                "Error message should mention invalid number, got: {}", err_msg);
        }

        #[test]
        fn test_vector_literal_leading_comma(
            values in prop::collection::vec(
                prop::num::f32::NORMAL,
                1..=32
            )
        ) {
            let valid_values: Vec<f32> = values.iter()
                .filter(|v| v.is_finite())
                .copied()
                .collect();
            
            if valid_values.is_empty() {
                return Ok(());
            }

            // Create literal with leading comma
            let literal = format!("[,{}]", 
                valid_values.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );

            let result = VectorLiteralParser::parse(&literal);
            prop_assert!(result.is_err(), 
                "Should fail for leading comma: {}", literal);
            
            let err_msg = result.unwrap_err().to_string();
            prop_assert!(err_msg.contains("Invalid number"),
                "Error message should mention invalid number, got: {}", err_msg);
        }

        #[test]
        fn test_vector_literal_double_comma(
            values in prop::collection::vec(
                prop::num::f32::NORMAL,
                2..=32
            ),
            insert_position in 1..=31usize,
        ) {
            let valid_values: Vec<f32> = values.iter()
                .filter(|v| v.is_finite())
                .copied()
                .collect();
            
            if valid_values.len() < 2 {
                return Ok(());
            }

            // Insert double comma at random position
            let mut parts: Vec<String> = valid_values.iter()
                .map(|v| v.to_string())
                .collect();
            
            let pos = insert_position.min(parts.len() - 1);
            parts.insert(pos, "".to_string()); // Empty string between commas

            let literal = format!("[{}]", parts.join(","));

            let result = VectorLiteralParser::parse(&literal);
            prop_assert!(result.is_err(), 
                "Should fail for double comma: {}", literal);
            
            let err_msg = result.unwrap_err().to_string();
            prop_assert!(err_msg.contains("Invalid number"),
                "Error message should mention invalid number, got: {}", err_msg);
        }

        #[test]
        fn test_vector_literal_no_brackets(
            values in prop::collection::vec(
                prop::num::f32::NORMAL,
                1..=32
            )
        ) {
            let valid_values: Vec<f32> = values.iter()
                .filter(|v| v.is_finite())
                .copied()
                .collect();
            
            if valid_values.is_empty() {
                return Ok(());
            }

            // Create literal without any brackets
            let literal = valid_values.iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",");

            let result = VectorLiteralParser::parse(&literal);
            prop_assert!(result.is_err(), "Should fail for missing brackets");
            
            let err_msg = result.unwrap_err().to_string();
            prop_assert!(err_msg.contains("must be enclosed in brackets"),
                "Error message should mention brackets, got: {}", err_msg);
        }
    }
}

#[cfg(test)]
mod dimension_validation_property_tests {
    use super::*;
    use proptest::prelude::*;

    // Feature: pgvector-sql-support, Property 10: Dimension Compatibility Validation
    // **Validates: Requirements 4.5**
    //
    // Property: For any expression involving vector operations, if the operand dimensions
    // are incompatible, the system should return an error before execution.
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_dimension_validation_matching_dimensions(
            dim in 1..=2048usize,
        ) {
            // Same dimensions should always validate successfully
            let result = validate_vector_dimensions(dim, dim);
            prop_assert!(result.is_ok(),
                "Validation should succeed for matching dimensions: {}", dim);
        }

        #[test]
        fn test_dimension_validation_mismatched_dimensions(
            dim1 in 1..=1024usize,
            dim2 in 1..=1024usize,
        ) {
            // Skip if dimensions happen to match
            if dim1 == dim2 {
                return Ok(());
            }

            // Different dimensions should always fail validation
            let result = validate_vector_dimensions(dim1, dim2);
            prop_assert!(result.is_err(),
                "Validation should fail for mismatched dimensions: {} vs {}", dim1, dim2);

            let err_msg = result.unwrap_err().to_string();
            prop_assert!(err_msg.contains("dimension mismatch"),
                "Error should mention dimension mismatch, got: {}", err_msg);
            prop_assert!(err_msg.contains(&dim1.to_string()),
                "Error should mention first dimension {}, got: {}", dim1, err_msg);
            prop_assert!(err_msg.contains(&dim2.to_string()),
                "Error should mention second dimension {}, got: {}", dim2, err_msg);
        }

        #[test]
        fn test_dimension_validation_with_parsed_vectors(
            values1 in prop::collection::vec(
                prop::num::f32::NORMAL,
                1..=64
            ),
            values2 in prop::collection::vec(
                prop::num::f32::NORMAL,
                1..=64
            ),
        ) {
            // Filter out special values
            let valid_values1: Vec<f32> = values1.iter()
                .filter(|v| v.is_finite())
                .copied()
                .collect();
            let valid_values2: Vec<f32> = values2.iter()
                .filter(|v| v.is_finite())
                .copied()
                .collect();
            
            if valid_values1.is_empty() || valid_values2.is_empty() {
                return Ok(());
            }

            // Parse both vectors
            let literal1 = format!("[{}]", 
                valid_values1.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );
            let literal2 = format!("[{}]", 
                valid_values2.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );

            let result1 = VectorLiteralParser::parse(&literal1);
            let result2 = VectorLiteralParser::parse(&literal2);

            prop_assert!(result1.is_ok() && result2.is_ok(),
                "Both vectors should parse successfully");

            // Validate dimensions
            let dim1 = valid_values1.len();
            let dim2 = valid_values2.len();
            let validation_result = validate_vector_dimensions(dim1, dim2);

            if dim1 == dim2 {
                prop_assert!(validation_result.is_ok(),
                    "Validation should succeed for matching dimensions: {}", dim1);
            } else {
                prop_assert!(validation_result.is_err(),
                    "Validation should fail for mismatched dimensions: {} vs {}", dim1, dim2);
            }
        }

        #[test]
        fn test_dimension_validation_edge_cases(
            dim in prop::sample::select(vec![1usize, 2, 3, 127, 128, 129, 255, 256, 512, 1024, 2048]),
            offset in 1..=10usize,
        ) {
            // Test validation around common dimension boundaries
            let other_dim = if dim > offset { dim - offset } else { dim + offset };

            let result = validate_vector_dimensions(dim, other_dim);
            
            if dim == other_dim {
                prop_assert!(result.is_ok(),
                    "Validation should succeed for equal dimensions: {}", dim);
            } else {
                prop_assert!(result.is_err(),
                    "Validation should fail for different dimensions: {} vs {}", dim, other_dim);
                
                let err_msg = result.unwrap_err().to_string();
                prop_assert!(err_msg.contains("dimension mismatch"),
                    "Error should mention dimension mismatch");
            }
        }

        #[test]
        fn test_dimension_validation_zero_dimension(
            other_dim in 0..=128usize,
        ) {
            let result = validate_vector_dimensions(0, other_dim);
            
            if other_dim == 0 {
                prop_assert!(result.is_ok(),
                    "Validation should succeed for both zero dimensions");
            } else {
                prop_assert!(result.is_err(),
                    "Validation should fail when one dimension is zero and other is {}", other_dim);
            }
        }
    }
}

#[cfg(test)]
mod sparse_vector_property_tests {
    use super::*;
    use proptest::prelude::*;

    // Feature: pgvector-sql-support, Property 11: Sparse Vector Parsing
    // **Validates: Requirements 5.2**
    //
    // Property: For any valid sparse vector literal in the format '{idx1:val1, idx2:val2, ...}'::sparsevec(dim),
    // the parser should produce a SparseVector with the correct indices, values, and dimensionality.
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_sparse_vector_parsing_correctness(
            // Generate sparse vectors with various sparsity levels
            dim in 10usize..1000,
            num_nonzero in 1usize..50,
        ) {
            // Ensure num_nonzero doesn't exceed dim
            let actual_nonzero = num_nonzero.min(dim);
            
            // Generate unique random indices within bounds
            let indices: Vec<u32> = (0..actual_nonzero)
                .map(|i| {
                    // Distribute indices evenly to ensure uniqueness
                    let step = dim / actual_nonzero;
                    (i * step) as u32
                })
                .collect();
            
            if indices.is_empty() {
                return Ok(());
            }
            
            // Generate random values
            let values: Vec<f32> = (0..indices.len())
                .map(|i| (i as f32 + 1.0) * 0.1)
                .collect();
            
            // Construct sparse vector literal
            let pairs: Vec<String> = indices.iter()
                .zip(values.iter())
                .map(|(idx, val)| format!("{}:{}", idx, val))
                .collect();
            let literal = format!("{{{}}}", pairs.join(", "));
            
            // Parse the literal
            let result = VectorLiteralParser::parse_sparse(&literal, dim);
            prop_assert!(result.is_ok(), "Failed to parse sparse literal: {}", literal);
            
            let sparse = result.unwrap();
            
            // Verify dimension
            prop_assert_eq!(sparse.dim, dim,
                "Dimension mismatch: expected {}, got {}", dim, sparse.dim);
            
            // Verify number of non-zero elements
            prop_assert_eq!(sparse.indices.len(), indices.len(),
                "Number of indices mismatch");
            prop_assert_eq!(sparse.values.len(), values.len(),
                "Number of values mismatch");
            
            // Verify indices and values
            for (i, (expected_idx, expected_val)) in indices.iter().zip(values.iter()).enumerate() {
                prop_assert_eq!(sparse.indices[i], *expected_idx,
                    "Index mismatch at position {}", i);
                
                let diff = (sparse.values[i] - expected_val).abs();
                prop_assert!(diff < 1e-6,
                    "Value mismatch at position {}: expected {}, got {}", 
                    i, expected_val, sparse.values[i]);
            }
        }

        #[test]
        fn test_sparse_vector_empty_parsing(
            dim in 1usize..1000,
        ) {
            // Empty sparse vector (all zeros)
            let literal = "{}";
            
            let result = VectorLiteralParser::parse_sparse(literal, dim);
            prop_assert!(result.is_ok(), "Failed to parse empty sparse vector");
            
            let sparse = result.unwrap();
            prop_assert_eq!(sparse.dim, dim);
            prop_assert_eq!(sparse.indices.len(), 0);
            prop_assert_eq!(sparse.values.len(), 0);
        }

        #[test]
        fn test_sparse_vector_with_various_formats(
            dim in 10usize..100,
            num_nonzero in 1usize..10,
            space_after_brace in prop::bool::ANY,
            space_before_brace in prop::bool::ANY,
            space_after_comma in prop::bool::ANY,
            space_before_colon in prop::bool::ANY,
            space_after_colon in prop::bool::ANY,
        ) {
            // Ensure num_nonzero doesn't exceed dim
            let actual_nonzero = num_nonzero.min(dim);
            
            // Generate unique indices within bounds
            let indices: Vec<u32> = (0..actual_nonzero)
                .map(|i| {
                    // Ensure index is strictly less than dim
                    let step = (dim - 1) / actual_nonzero.max(1);
                    (i * step) as u32
                })
                .collect();
            let values: Vec<f32> = (0..indices.len())
                .map(|i| (i as f32 + 1.0) * 0.5)
                .collect();
            
            if indices.is_empty() {
                return Ok(());
            }
            
            // Build literal with various spacing
            let mut literal = String::from("{");
            if space_after_brace {
                literal.push(' ');
            }
            
            for (i, (idx, val)) in indices.iter().zip(values.iter()).enumerate() {
                if i > 0 {
                    literal.push(',');
                    if space_after_comma {
                        literal.push(' ');
                    }
                }
                literal.push_str(&idx.to_string());
                if space_before_colon {
                    literal.push(' ');
                }
                literal.push(':');
                if space_after_colon {
                    literal.push(' ');
                }
                literal.push_str(&val.to_string());
            }
            
            if space_before_brace {
                literal.push(' ');
            }
            literal.push('}');
            
            // Parse should succeed regardless of spacing
            let result = VectorLiteralParser::parse_sparse(&literal, dim);
            prop_assert!(result.is_ok(), 
                "Failed to parse sparse literal with spacing: {}", literal);
            
            let sparse = result.unwrap();
            prop_assert_eq!(sparse.dim, dim);
            prop_assert_eq!(sparse.indices.len(), indices.len());
        }

        #[test]
        fn test_sparse_vector_index_bounds_validation(
            dim in 10usize..100,
            num_valid in 1usize..10,
        ) {
            // Generate unique indices within bounds
            let actual_num = num_valid.min(dim);
            let valid_indices: Vec<u32> = (0..actual_num)
                .map(|i| {
                    // Ensure unique indices by spacing them out
                    let step = (dim - 1) / actual_num.max(1);
                    (i * step) as u32
                })
                .collect();
            
            let values: Vec<f32> = (0..valid_indices.len())
                .map(|i| (i as f32 + 1.0) * 0.1)
                .collect();
            
            let pairs: Vec<String> = valid_indices.iter()
                .zip(values.iter())
                .map(|(idx, val)| format!("{}:{}", idx, val))
                .collect();
            let literal = format!("{{{}}}", pairs.join(", "));
            
            let result = VectorLiteralParser::parse_sparse(&literal, dim);
            prop_assert!(result.is_ok(),
                "Should parse successfully when all indices are within bounds");
            
            // Now test with an out-of-bounds index
            let out_of_bounds_literal = format!("{{{}:{}}}", dim, 1.0);
            let result = VectorLiteralParser::parse_sparse(&out_of_bounds_literal, dim);
            prop_assert!(result.is_err(),
                "Should fail when index equals or exceeds dimension");
            
            let err_msg = result.unwrap_err().to_string();
            prop_assert!(err_msg.contains("exceeds dimension"),
                "Error should mention dimension bounds");
        }

        #[test]
        fn test_sparse_vector_duplicate_detection(
            dim in 10usize..100,
            duplicate_index in 0u32..10,
            val1 in -10.0f32..10.0,
            val2 in -10.0f32..10.0,
        ) {
            // Create a sparse vector with duplicate indices
            let literal = format!("{{{}:{}, {}:{}}}", 
                duplicate_index, val1, duplicate_index, val2);
            
            let result = VectorLiteralParser::parse_sparse(&literal, dim);
            prop_assert!(result.is_err(),
                "Should fail for duplicate indices");
            
            let err_msg = result.unwrap_err().to_string();
            prop_assert!(err_msg.contains("Duplicate index"),
                "Error should mention duplicate index, got: {}", err_msg);
        }

        #[test]
        fn test_sparse_vector_with_negative_values(
            dim in 10usize..100,
            num_nonzero in 1usize..10,
        ) {
            // Ensure num_nonzero doesn't exceed dim
            let actual_nonzero = num_nonzero.min(dim);
            
            // Generate unique indices within bounds
            let indices: Vec<u32> = (0..actual_nonzero)
                .map(|i| {
                    // Ensure index is strictly less than dim
                    let step = (dim - 1) / actual_nonzero.max(1);
                    (i * step) as u32
                })
                .collect();
            let values: Vec<f32> = (0..indices.len())
                .map(|i| -((i as f32 + 1.0) * 0.5))
                .collect();
            
            if indices.is_empty() {
                return Ok(());
            }
            
            let pairs: Vec<String> = indices.iter()
                .zip(values.iter())
                .map(|(idx, val)| format!("{}:{}", idx, val))
                .collect();
            let literal = format!("{{{}}}", pairs.join(", "));
            
            let result = VectorLiteralParser::parse_sparse(&literal, dim);
            prop_assert!(result.is_ok(),
                "Should parse negative values successfully");
            
            let sparse = result.unwrap();
            for (i, expected_val) in values.iter().enumerate() {
                let diff = (sparse.values[i] - expected_val).abs();
                prop_assert!(diff < 1e-6,
                    "Negative value mismatch at position {}", i);
            }
        }

        #[test]
        fn test_sparse_vector_with_zero_values(
            dim in 10usize..100,
            zero_index in 0u32..5,
            nonzero_offset in 6u32..10,
        ) {
            // Ensure both indices are within bounds
            let zero_idx = zero_index.min((dim - 1) as u32);
            let nonzero_idx = (zero_index + nonzero_offset).min((dim - 1) as u32);
            
            // Skip if indices would be the same
            if zero_idx == nonzero_idx {
                return Ok(());
            }
            
            // Sparse vectors can contain explicit zeros (though not efficient)
            let literal = format!("{{{}:0.0, {}:1.5}}", zero_idx, nonzero_idx);
            
            let result = VectorLiteralParser::parse_sparse(&literal, dim);
            prop_assert!(result.is_ok(),
                "Should parse zero values successfully");
            
            let sparse = result.unwrap();
            prop_assert_eq!(sparse.indices.len(), 2);
            prop_assert_eq!(sparse.values[0], 0.0);
            prop_assert_eq!(sparse.values[1], 1.5);
        }

        #[test]
        fn test_sparse_vector_error_handling_invalid_format(
            dim in 10usize..100,
            invalid_token in "[a-zA-Z]{3,10}",
        ) {
            // Test various invalid formats
            
            // Missing colon
            let literal = format!("{{0 {}}}", invalid_token);
            let result = VectorLiteralParser::parse_sparse(&literal, dim);
            prop_assert!(result.is_err(),
                "Should fail for missing colon");
            
            // Invalid index
            let literal = format!("{{{}:1.0}}", invalid_token);
            let result = VectorLiteralParser::parse_sparse(&literal, dim);
            prop_assert!(result.is_err(),
                "Should fail for invalid index");
            
            // Invalid value
            let literal = format!("{{0:{}}}", invalid_token);
            let result = VectorLiteralParser::parse_sparse(&literal, dim);
            prop_assert!(result.is_err(),
                "Should fail for invalid value");
        }

        #[test]
        fn test_sparse_vector_missing_braces(
            dim in 10usize..100,
        ) {
            // Missing opening brace
            let literal = "0:1.0, 1:2.0}";
            let result = VectorLiteralParser::parse_sparse(literal, dim);
            prop_assert!(result.is_err(),
                "Should fail for missing opening brace");
            prop_assert!(result.unwrap_err().to_string().contains("must be enclosed in braces"));
            
            // Missing closing brace
            let literal = "{0:1.0, 1:2.0";
            let result = VectorLiteralParser::parse_sparse(literal, dim);
            prop_assert!(result.is_err(),
                "Should fail for missing closing brace");
            prop_assert!(result.unwrap_err().to_string().contains("must be enclosed in braces"));
        }

        #[test]
        fn test_sparse_vector_high_sparsity(
            dim in 1000usize..10000,
            num_nonzero in 1usize..10,
        ) {
            // Test very sparse vectors (high dimensionality, few non-zeros)
            let indices: Vec<u32> = (0..num_nonzero)
                .map(|i| (i * (dim / num_nonzero)) as u32)
                .collect();
            let values: Vec<f32> = (0..indices.len())
                .map(|i| (i as f32 + 1.0) * 0.1)
                .collect();
            
            let pairs: Vec<String> = indices.iter()
                .zip(values.iter())
                .map(|(idx, val)| format!("{}:{}", idx, val))
                .collect();
            let literal = format!("{{{}}}", pairs.join(", "));
            
            let result = VectorLiteralParser::parse_sparse(&literal, dim);
            prop_assert!(result.is_ok(),
                "Should parse high-dimensional sparse vectors");
            
            let sparse = result.unwrap();
            prop_assert_eq!(sparse.dim, dim);
            
            // Verify sparsity
            let sparsity = 1.0 - (sparse.indices.len() as f64 / dim as f64);
            prop_assert!(sparsity > 0.99,
                "Should maintain high sparsity: {}", sparsity);
        }
    }
}


#[cfg(test)]
mod binary_vector_property_tests {
    use super::*;
    use proptest::prelude::*;

    // Feature: pgvector-sql-support, Property 17: Binary Quantization Correctness
    // **Validates: Requirements 7.1**
    //
    // Property: For any vector, binary_quantize should produce a bit-packed representation
    // where each bit is 1 if the corresponding element is >= 0, and 0 otherwise.
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_binary_quantization_correctness(
            // Generate vectors with various dimensions
            values in prop::collection::vec(
                // Generate f32 values in a reasonable range
                -10.0f32..10.0f32,
                1..=256
            )
        ) {
            // For each value, determine expected bit
            let expected_bits: Vec<bool> = values.iter()
                .map(|v| *v >= 0.0)
                .collect();
            
            // Construct binary literal from expected bits
            let binary_str: String = expected_bits.iter()
                .map(|b| if *b { '1' } else { '0' })
                .collect();
            let literal = format!("B'{}'", binary_str);
            
            // Parse the binary literal
            let result = VectorLiteralParser::parse_binary(&literal, Some(values.len()));
            prop_assert!(result.is_ok(), "Failed to parse binary literal: {}", literal);
            
            let bytes = result.unwrap();
            
            // Verify each bit matches the expected quantization
            for (bit_idx, expected_bit) in expected_bits.iter().enumerate() {
                let byte_idx = bit_idx / 8;
                let bit_pos = 7 - (bit_idx % 8); // MSB first
                let actual_bit = (bytes[byte_idx] >> bit_pos) & 1;
                
                prop_assert_eq!(actual_bit == 1, *expected_bit,
                    "Bit mismatch at index {}: expected {}, got {} (value was {})",
                    bit_idx, expected_bit, actual_bit == 1, values[bit_idx]);
            }
        }

        #[test]
        fn test_binary_literal_parsing_binary_format(
            // Generate random bit patterns
            bits in prop::collection::vec(
                prop::bool::ANY,
                1..=256
            )
        ) {
            // Construct binary literal
            let binary_str: String = bits.iter()
                .map(|b| if *b { '1' } else { '0' })
                .collect();
            let literal = format!("B'{}'", binary_str);
            
            // Parse the literal
            let result = VectorLiteralParser::parse_binary(&literal, Some(bits.len()));
            prop_assert!(result.is_ok(), "Failed to parse binary literal: {}", literal);
            
            let bytes = result.unwrap();
            
            // Verify byte count
            let expected_byte_count = (bits.len() + 7) / 8;
            prop_assert_eq!(bytes.len(), expected_byte_count,
                "Byte count mismatch: expected {}, got {}", expected_byte_count, bytes.len());
            
            // Verify each bit
            for (bit_idx, expected_bit) in bits.iter().enumerate() {
                let byte_idx = bit_idx / 8;
                let bit_pos = 7 - (bit_idx % 8);
                let actual_bit = (bytes[byte_idx] >> bit_pos) & 1;
                
                prop_assert_eq!(actual_bit == 1, *expected_bit,
                    "Bit mismatch at index {}", bit_idx);
            }
        }

        #[test]
        fn test_binary_literal_parsing_hex_format(
            // Generate random bytes
            bytes_input in prop::collection::vec(
                any::<u8>(),
                1..=32
            )
        ) {
            // Construct hex literal
            let hex_str: String = bytes_input.iter()
                .map(|b| format!("{:02X}", b))
                .collect();
            let literal = format!("'\\x{}'", hex_str);
            
            // Parse the literal
            let result = VectorLiteralParser::parse_binary(&literal, Some(bytes_input.len() * 8));
            prop_assert!(result.is_ok(), "Failed to parse hex literal: {}", literal);
            
            let bytes = result.unwrap();
            
            // Verify bytes match
            prop_assert_eq!(bytes.len(), bytes_input.len(),
                "Byte count mismatch");
            
            for (i, (expected, actual)) in bytes_input.iter().zip(bytes.iter()).enumerate() {
                prop_assert_eq!(actual, expected,
                    "Byte mismatch at index {}: expected 0x{:02X}, got 0x{:02X}",
                    i, expected, actual);
            }
        }

        #[test]
        fn test_binary_format_round_trip(
            // Generate random bit patterns
            bits in prop::collection::vec(
                prop::bool::ANY,
                8..=128
            )
        ) {
            // Construct binary literal
            let binary_str: String = bits.iter()
                .map(|b| if *b { '1' } else { '0' })
                .collect();
            let literal = format!("B'{}'", binary_str);
            
            // Parse and format
            let bytes = VectorLiteralParser::parse_binary(&literal, Some(bits.len())).unwrap();
            let formatted = format_binary_vector(&bytes, bits.len(), false);
            
            // Should match original
            prop_assert_eq!(formatted, binary_str,
                "Binary format round-trip failed");
        }

        #[test]
        fn test_hex_format_round_trip(
            // Generate random bytes
            bytes_input in prop::collection::vec(
                any::<u8>(),
                1..=32
            )
        ) {
            // Construct hex literal
            let hex_str: String = bytes_input.iter()
                .map(|b| format!("{:02X}", b))
                .collect();
            let literal = format!("'\\x{}'", hex_str);
            
            // Parse and format
            let bytes = VectorLiteralParser::parse_binary(&literal, Some(bytes_input.len() * 8)).unwrap();
            let formatted = format_binary_vector(&bytes, bytes_input.len() * 8, true);
            
            // Should match original (with 0x prefix)
            let expected = format!("0x{}", hex_str);
            prop_assert_eq!(formatted, expected,
                "Hex format round-trip failed");
        }

        #[test]
        fn test_binary_literal_bit_count_validation(
            // Generate random bit patterns
            bits in prop::collection::vec(
                prop::bool::ANY,
                1..=128
            ),
            wrong_count in 1..=256usize,
        ) {
            // Skip if wrong_count happens to match
            if wrong_count == bits.len() {
                return Ok(());
            }
            
            // Construct binary literal
            let binary_str: String = bits.iter()
                .map(|b| if *b { '1' } else { '0' })
                .collect();
            let literal = format!("B'{}'", binary_str);
            
            // Parse with wrong expected bit count
            let result = VectorLiteralParser::parse_binary(&literal, Some(wrong_count));
            prop_assert!(result.is_err(),
                "Should fail when expected bit count ({}) doesn't match actual ({})",
                wrong_count, bits.len());
            
            let err_msg = result.unwrap_err().to_string();
            prop_assert!(err_msg.contains("bit count mismatch"),
                "Error should mention bit count mismatch, got: {}", err_msg);
        }

        #[test]
        fn test_binary_literal_invalid_digit(
            // Generate valid binary prefix
            valid_bits in prop::collection::vec(
                prop::bool::ANY,
                1..=16
            ),
            // Generate invalid character
            invalid_char in "[2-9a-zA-Z]",
            insert_position in 0..=16usize,
        ) {
            // Construct binary string with invalid digit
            let mut binary_str: String = valid_bits.iter()
                .map(|b| if *b { '1' } else { '0' })
                .collect();
            
            let pos = insert_position.min(binary_str.len());
            binary_str.insert_str(pos, &invalid_char);
            
            let literal = format!("B'{}'", binary_str);
            
            // Parse should fail
            let result = VectorLiteralParser::parse_binary(&literal, None);
            prop_assert!(result.is_err(),
                "Should fail for invalid binary digit: {}", literal);
            
            let err_msg = result.unwrap_err().to_string();
            prop_assert!(err_msg.contains("Invalid binary digit"),
                "Error should mention invalid binary digit, got: {}", err_msg);
        }

        #[test]
        fn test_hex_literal_invalid_digit(
            // Generate valid hex prefix
            valid_bytes in prop::collection::vec(
                any::<u8>(),
                1..=8
            ),
            // Generate invalid character
            invalid_char in "[G-Zg-z]",
        ) {
            // Construct hex string with invalid digit
            let mut hex_str: String = valid_bytes.iter()
                .map(|b| format!("{:02X}", b))
                .collect();
            
            // Insert invalid character
            hex_str.push_str(&invalid_char);
            hex_str.push('0'); // Make it even length
            
            let literal = format!("'\\x{}'", hex_str);
            
            // Parse should fail
            let result = VectorLiteralParser::parse_binary(&literal, None);
            prop_assert!(result.is_err(),
                "Should fail for invalid hex digit: {}", literal);
            
            let err_msg = result.unwrap_err().to_string();
            prop_assert!(err_msg.contains("Invalid hex digit"),
                "Error should mention invalid hex digit, got: {}", err_msg);
        }

        #[test]
        fn test_binary_literal_partial_byte_handling(
            // Generate bit patterns that don't align to byte boundaries
            bit_count in 1..=15usize,
        ) {
            // Generate random bits
            let bits: Vec<bool> = (0..bit_count)
                .map(|i| i % 2 == 0)
                .collect();
            
            let binary_str: String = bits.iter()
                .map(|b| if *b { '1' } else { '0' })
                .collect();
            let literal = format!("B'{}'", binary_str);
            
            // Parse should succeed
            let result = VectorLiteralParser::parse_binary(&literal, Some(bit_count));
            prop_assert!(result.is_ok(),
                "Should parse partial byte successfully: {}", literal);
            
            let bytes = result.unwrap();
            
            // Verify byte count (should round up)
            let expected_byte_count = (bit_count + 7) / 8;
            prop_assert_eq!(bytes.len(), expected_byte_count,
                "Byte count should round up for partial bytes");
            
            // Verify the bits we care about
            for (bit_idx, expected_bit) in bits.iter().enumerate() {
                let byte_idx = bit_idx / 8;
                let bit_pos = 7 - (bit_idx % 8);
                let actual_bit = (bytes[byte_idx] >> bit_pos) & 1;
                
                prop_assert_eq!(actual_bit == 1, *expected_bit,
                    "Bit mismatch at index {}", bit_idx);
            }
        }

        #[test]
        fn test_binary_format_consistency(
            // Generate random bytes
            bytes_input in prop::collection::vec(
                any::<u8>(),
                1..=32
            ),
            bit_count in 8..=256usize,
        ) {
            // Ensure bit_count is valid for the byte count
            let actual_bit_count = bit_count.min(bytes_input.len() * 8);
            
            // Format as binary
            let binary_str = format_binary_vector(&bytes_input, actual_bit_count, false);
            
            // Format as hex
            let hex_str = format_binary_vector(&bytes_input, actual_bit_count, true);
            
            // Verify binary string length
            prop_assert_eq!(binary_str.len(), actual_bit_count,
                "Binary string length should match bit count");
            
            // Verify hex string format
            prop_assert!(hex_str.starts_with("0x"),
                "Hex string should start with 0x");
            
            // Verify all characters are valid
            for ch in binary_str.chars() {
                prop_assert!(ch == '0' || ch == '1',
                    "Binary string should only contain 0 and 1");
            }
            
            for ch in hex_str.chars().skip(2) {
                prop_assert!(ch.is_ascii_hexdigit(),
                    "Hex string should only contain hex digits");
            }
        }
    }
}
