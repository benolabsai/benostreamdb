// Copyright (c) 2026 Richard Albright. All rights reserved.

use datafusion::error::{DataFusionError, Result};

pub mod binary;
pub mod dense;
pub mod sparse;

// Re-exports from submodules for backward compatibility
pub use binary::{format_binary_vector, parse_binary_vector};
pub use dense::parse_vector_literal;
pub use sparse::parse_sparse_vector;

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
    /// Delegate to dense vector parsing
    pub fn parse(input: &str) -> datafusion::error::Result<datafusion::scalar::ScalarValue> {
        dense::parse_vector_literal(input)
    }

    /// Delegate to binary vector parsing
    pub fn parse_binary(
        input: &str,
        expected_bits: Option<usize>,
    ) -> datafusion::error::Result<Vec<u8>> {
        binary::parse_binary_vector(input, expected_bits)
    }

    /// Delegate to sparse vector parsing
    pub fn parse_sparse(
        input: &str,
        dim: usize,
    ) -> datafusion::error::Result<crate::core::index::SparseVector> {
        sparse::parse_sparse_vector(input, dim)
    }
}

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
        return Err(DataFusionError::Plan(format!(
            "Vector dimension mismatch: expected {}, got {}",
            left_dim, right_dim
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Array;

    // -- Dense vector tests --

    #[test]
    fn test_parse_integer_vector() {
        let result = VectorLiteralParser::parse("[1,2,3]");
        assert!(result.is_ok());

        let scalar = result.unwrap();
        match scalar {
            datafusion::scalar::ScalarValue::FixedSizeList(arr) => {
                assert_eq!(arr.len(), 1);
                assert_eq!(arr.value_length(), 3);
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
            datafusion::scalar::ScalarValue::FixedSizeList(arr) => {
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
            datafusion::scalar::ScalarValue::FixedSizeList(arr) => {
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
            datafusion::scalar::ScalarValue::FixedSizeList(arr) => {
                assert_eq!(arr.value_length(), 1);
            }
            _ => panic!("Expected FixedSizeList"),
        }
    }

    #[test]
    fn test_parse_missing_opening_bracket() {
        let result = VectorLiteralParser::parse("1,2,3]");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("must be enclosed in brackets"));
    }

    #[test]
    fn test_parse_missing_closing_bracket() {
        let result = VectorLiteralParser::parse("[1,2,3");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("must be enclosed in brackets"));
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
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid number at position"));
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
        let result = VectorLiteralParser::parse("[NaN, 1.0, 2.0]");
        assert!(result.is_ok());

        let result = VectorLiteralParser::parse("[inf, 1.0, 2.0]");
        assert!(result.is_ok());

        let result = VectorLiteralParser::parse("[-inf, 1.0, 2.0]");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_multiple_errors() {
        let result = VectorLiteralParser::parse("1,2,3");
        assert!(result.is_err());

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
        let result = VectorLiteralParser::parse("[1, 2, 3,]");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid number"));
    }

    #[test]
    fn test_parse_leading_comma() {
        let result = VectorLiteralParser::parse("[,1, 2, 3]");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid number"));
    }

    #[test]
    fn test_parse_double_comma() {
        let result = VectorLiteralParser::parse("[1,, 2, 3]");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid number"));
    }

    #[test]
    fn test_parse_very_large_numbers() {
        let result = VectorLiteralParser::parse("[1e308, 1e309, 1.0]");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_very_small_numbers() {
        let result = VectorLiteralParser::parse("[1e-45, 1e-50, 1.0]");
        assert!(result.is_ok());
    }

    // -- Dimension validation tests --

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
        let result = validate_vector_dimensions(0, 0);
        assert!(result.is_ok());

        let result = validate_vector_dimensions(0, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_dimensions_large() {
        let result = validate_vector_dimensions(2048, 2048);
        assert!(result.is_ok());

        let result = validate_vector_dimensions(2048, 2049);
        assert!(result.is_err());
    }

    // -- Sparse vector tests --

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
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("must be enclosed in braces"));
    }

    #[test]
    fn test_parse_sparse_vector_missing_closing_brace() {
        let result = VectorLiteralParser::parse_sparse("{1:0.5, 10:0.3", 1000);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("must be enclosed in braces"));
    }

    #[test]
    fn test_parse_sparse_vector_invalid_format() {
        let result = VectorLiteralParser::parse_sparse("{1 0.5}", 1000);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("expected 'index:value'"));

        let result = VectorLiteralParser::parse_sparse("{1:0.5:extra}", 1000);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("expected 'index:value'"));
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
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("exceeds dimension"));
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
        let result = VectorLiteralParser::parse_sparse("{1:0.0, 10:0.5}", 1000);
        assert!(result.is_ok());

        let sparse = result.unwrap();
        assert_eq!(sparse.values, vec![0.0, 0.5]);
    }

    // -- Binary vector tests --

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
        let result = VectorLiteralParser::parse_binary("B'10110'", Some(5));
        assert!(result.is_ok());

        let bytes = result.unwrap();
        assert_eq!(bytes.len(), 1);
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
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid binary digit"));
    }

    #[test]
    fn test_parse_binary_missing_closing_quote() {
        let result = VectorLiteralParser::parse_binary("B'10110101", Some(8));
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("must end with single quote"));
    }

    #[test]
    fn test_parse_binary_bit_count_mismatch() {
        let result = VectorLiteralParser::parse_binary("B'10110101'", Some(16));
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("bit count mismatch"));
    }

    #[test]
    fn test_parse_binary_hex_invalid_digit() {
        let result = VectorLiteralParser::parse_binary("'\\xGH'", Some(8));
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid hex digit"));
    }

    #[test]
    fn test_parse_binary_hex_incomplete_byte() {
        let result = VectorLiteralParser::parse_binary("'\\xB'", Some(8));
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Incomplete hex byte"));
    }

    #[test]
    fn test_parse_binary_invalid_format() {
        let result = VectorLiteralParser::parse_binary("10110101", Some(8));
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("must be in format"));
    }

    #[test]
    fn test_parse_binary_no_expected_bits() {
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
        let original = "B'10110101'";
        let bytes = VectorLiteralParser::parse_binary(original, Some(8)).unwrap();
        let formatted = format_binary_vector(&bytes, 8, false);
        assert_eq!(formatted, "10110101");

        let original_hex = "'\\xB5'";
        let bytes_hex = VectorLiteralParser::parse_binary(original_hex, Some(8)).unwrap();
        let formatted_hex = format_binary_vector(&bytes_hex, 8, true);
        assert_eq!(formatted_hex, "0xB5");
    }

    // -- Property tests for dense vectors --

    #[cfg(feature = "proptest")]
    mod property_tests {
        use super::*;
        use arrow::array::Float32Array;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(100))]

            #[test]
            fn test_vector_literal_parsing_correctness(
                values in prop::collection::vec(
                    prop::num::f32::NORMAL,
                    1..=128
                )
            ) {
                let valid_values: Vec<f32> = values.iter()
                    .filter(|v| v.is_finite())
                    .copied()
                    .collect();

                if valid_values.is_empty() {
                    return Ok(());
                }

                let literal = format!("[{}]",
                    valid_values.iter()
                        .map(|v| v.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                );

                let result = VectorLiteralParser::parse(&literal);
                prop_assert!(result.is_ok(), "Failed to parse valid literal: {}", literal);

                let scalar = result.unwrap();

                match scalar {
                    datafusion::scalar::ScalarValue::FixedSizeList(arr) => {
                        prop_assert_eq!(arr.value_length() as usize, valid_values.len(),
                            "Dimension mismatch: expected {}, got {}",
                            valid_values.len(), arr.value_length());

                        let float_array = arr.value(0);
                        let float_array = float_array.as_any()
                            .downcast_ref::<Float32Array>()
                            .expect("Expected Float32Array");

                        prop_assert_eq!(float_array.len(), valid_values.len(),
                            "Array length mismatch");

                        for (i, expected) in valid_values.iter().enumerate() {
                            let actual = float_array.value(i);
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
                values in prop::collection::vec(
                    prop::num::f32::NORMAL,
                    1..=32
                ),
                space_after_bracket in prop::bool::ANY,
                space_before_bracket in prop::bool::ANY,
                space_after_comma in prop::bool::ANY,
                space_before_comma in prop::bool::ANY,
            ) {
                let valid_values: Vec<f32> = values.iter()
                    .filter(|v| v.is_finite())
                    .copied()
                    .collect();

                if valid_values.is_empty() {
                    return Ok(());
                }

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

                let result = VectorLiteralParser::parse(&literal);
                prop_assert!(result.is_ok(),
                    "Failed to parse literal with spacing: {}", literal);

                if let datafusion::scalar::ScalarValue::FixedSizeList(arr) = result.unwrap() {
                    prop_assert_eq!(arr.value_length() as usize, valid_values.len());
                }
            }

            #[test]
            fn test_vector_literal_with_integer_values(
                values in prop::collection::vec(
                    -1000i32..=1000i32,
                    1..=64
                )
            ) {
                let literal = format!("[{}]",
                    values.iter()
                        .map(|v| v.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                );

                let result = VectorLiteralParser::parse(&literal);
                prop_assert!(result.is_ok(), "Failed to parse integer literal: {}", literal);

                if let datafusion::scalar::ScalarValue::FixedSizeList(arr) = result.unwrap() {
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
                int_values in prop::collection::vec(-100i32..=100i32, 1..=16),
                float_values in prop::collection::vec(
                    prop::num::f32::NORMAL,
                    1..=16
                )
            ) {
                let valid_floats: Vec<f32> = float_values.iter()
                    .filter(|v| v.is_finite())
                    .copied()
                    .collect();

                if valid_floats.is_empty() {
                    return Ok(());
                }

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

                if let datafusion::scalar::ScalarValue::FixedSizeList(arr) = result.unwrap() {
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
    }

    // -- Property tests for dimension validation --

    #[cfg(feature = "proptest")]
    mod dimension_validation_property_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(100))]

            #[test]
            fn test_dimension_validation_matching_dimensions(
                dim in 1..=2048usize,
            ) {
                let result = validate_vector_dimensions(dim, dim);
                prop_assert!(result.is_ok(),
                    "Validation should succeed for matching dimensions: {}", dim);
            }

            #[test]
            fn test_dimension_validation_mismatched_dimensions(
                dim1 in 1..=1024usize,
                dim2 in 1..=1024usize,
            ) {
                if dim1 == dim2 {
                    return Ok(());
                }

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

    // -- Property tests for sparse vectors --

    #[cfg(feature = "proptest")]
    mod sparse_vector_property_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(100))]

            #[test]
            fn test_sparse_vector_parsing_correctness(
                dim in 10usize..1000,
                num_nonzero in 1usize..50,
            ) {
                let actual_nonzero = num_nonzero.min(dim);

                let indices: Vec<u32> = (0..actual_nonzero)
                    .map(|i| {
                        let step = dim / actual_nonzero;
                        (i * step) as u32
                    })
                    .collect();

                if indices.is_empty() {
                    return Ok(());
                }

                let values: Vec<f32> = (0..indices.len())
                    .map(|i| (i as f32 + 1.0) * 0.1)
                    .collect();

                let pairs: Vec<String> = indices.iter()
                    .zip(values.iter())
                    .map(|(idx, val)| format!("{}:{}", idx, val))
                    .collect();
                let literal = format!("{{{}}}", pairs.join(", "));

                let result = VectorLiteralParser::parse_sparse(&literal, dim);
                prop_assert!(result.is_ok(), "Failed to parse sparse literal: {}", literal);

                let sparse = result.unwrap();

                prop_assert_eq!(sparse.dim, dim,
                    "Dimension mismatch: expected {}, got {}", dim, sparse.dim);

                prop_assert_eq!(sparse.indices.len(), indices.len(),
                    "Number of indices mismatch");
                prop_assert_eq!(sparse.values.len(), values.len(),
                    "Number of values mismatch");

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
                let actual_nonzero = num_nonzero.min(dim);

                let indices: Vec<u32> = (0..actual_nonzero)
                    .map(|i| {
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
                let actual_num = num_valid.min(dim);
                let valid_indices: Vec<u32> = (0..actual_num)
                    .map(|i| {
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
                let actual_nonzero = num_nonzero.min(dim);

                let indices: Vec<u32> = (0..actual_nonzero)
                    .map(|i| {
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
                let zero_idx = zero_index.min((dim - 1) as u32);
                let nonzero_idx = (zero_index + nonzero_offset).min((dim - 1) as u32);

                if zero_idx == nonzero_idx {
                    return Ok(());
                }

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
                let literal = format!("{{0 {}}}", invalid_token);
                let result = VectorLiteralParser::parse_sparse(&literal, dim);
                prop_assert!(result.is_err(),
                    "Should fail for missing colon");

                let literal = format!("{{{}:1.0}}", invalid_token);
                let result = VectorLiteralParser::parse_sparse(&literal, dim);
                prop_assert!(result.is_err(),
                    "Should fail for invalid index");

                let literal = format!("{{0:{}}}", invalid_token);
                let result = VectorLiteralParser::parse_sparse(&literal, dim);
                prop_assert!(result.is_err(),
                    "Should fail for invalid value");
            }

            #[test]
            fn test_sparse_vector_missing_braces(
                dim in 10usize..100,
            ) {
                let literal = "0:1.0, 1:2.0}";
                let result = VectorLiteralParser::parse_sparse(literal, dim);
                prop_assert!(result.is_err(),
                    "Should fail for missing opening brace");
                prop_assert!(result.unwrap_err().to_string().contains("must be enclosed in braces"));

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

                let sparsity = 1.0 - (sparse.indices.len() as f64 / dim as f64);
                prop_assert!(sparsity > 0.99,
                    "Should maintain high sparsity: {}", sparsity);
            }
        }
    }

    // -- Property tests for binary vectors --

    #[cfg(feature = "proptest")]
    mod binary_vector_property_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(100))]

            #[test]
            fn test_binary_quantization_correctness(
                values in prop::collection::vec(
                    -10.0f32..10.0f32,
                    1..=256
                )
            ) {
                let expected_bits: Vec<bool> = values.iter()
                    .map(|v| *v >= 0.0)
                    .collect();

                let binary_str: String = expected_bits.iter()
                    .map(|b| if *b { '1' } else { '0' })
                    .collect();
                let literal = format!("B'{}'", binary_str);

                let result = VectorLiteralParser::parse_binary(&literal, Some(values.len()));
                prop_assert!(result.is_ok(), "Failed to parse binary literal: {}", literal);

                let bytes = result.unwrap();

                for (bit_idx, expected_bit) in expected_bits.iter().enumerate() {
                    let byte_idx = bit_idx / 8;
                    let bit_pos = 7 - (bit_idx % 8);
                    let actual_bit = (bytes[byte_idx] >> bit_pos) & 1;

                    prop_assert_eq!(actual_bit == 1, *expected_bit,
                        "Bit mismatch at index {}: expected {}, got {} (value was {})",
                        bit_idx, expected_bit, actual_bit == 1, values[bit_idx]);
                }
            }

            #[test]
            fn test_binary_literal_parsing_binary_format(
                bits in prop::collection::vec(
                    prop::bool::ANY,
                    1..=256
                )
            ) {
                let binary_str: String = bits.iter()
                    .map(|b| if *b { '1' } else { '0' })
                    .collect();
                let literal = format!("B'{}'", binary_str);

                let result = VectorLiteralParser::parse_binary(&literal, Some(bits.len()));
                prop_assert!(result.is_ok(), "Failed to parse binary literal: {}", literal);

                let bytes = result.unwrap();

                let expected_byte_count = bits.len().div_ceil(8);
                prop_assert_eq!(bytes.len(), expected_byte_count,
                    "Byte count mismatch: expected {}, got {}", expected_byte_count, bytes.len());

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
                bytes_input in prop::collection::vec(
                    any::<u8>(),
                    1..=32
                )
            ) {
                let hex_str: String = bytes_input.iter()
                    .map(|b| format!("{:02X}", b))
                    .collect();
                let literal = format!("'\\x{}'", hex_str);

                let result = VectorLiteralParser::parse_binary(&literal, Some(bytes_input.len() * 8));
                prop_assert!(result.is_ok(), "Failed to parse hex literal: {}", literal);

                let bytes = result.unwrap();

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
                bits in prop::collection::vec(
                    prop::bool::ANY,
                    8..=128
                )
            ) {
                let binary_str: String = bits.iter()
                    .map(|b| if *b { '1' } else { '0' })
                    .collect();
                let literal = format!("B'{}'", binary_str);

                let bytes = VectorLiteralParser::parse_binary(&literal, Some(bits.len())).unwrap();
                let formatted = format_binary_vector(&bytes, bits.len(), false);

                prop_assert_eq!(formatted, binary_str,
                    "Binary format round-trip failed");
            }

            #[test]
            fn test_hex_format_round_trip(
                bytes_input in prop::collection::vec(
                    any::<u8>(),
                    1..=32
                )
            ) {
                let hex_str: String = bytes_input.iter()
                    .map(|b| format!("{:02X}", b))
                    .collect();
                let literal = format!("'\\x{}'", hex_str);

                let bytes = VectorLiteralParser::parse_binary(&literal, Some(bytes_input.len() * 8)).unwrap();
                let formatted = format_binary_vector(&bytes, bytes_input.len() * 8, true);

                let expected = format!("0x{}", hex_str);
                prop_assert_eq!(formatted, expected,
                    "Hex format round-trip failed");
            }

            #[test]
            fn test_binary_literal_bit_count_validation(
                bits in prop::collection::vec(
                    prop::bool::ANY,
                    1..=128
                ),
                wrong_count in 1..=256usize,
            ) {
                if wrong_count == bits.len() {
                    return Ok(());
                }

                let binary_str: String = bits.iter()
                    .map(|b| if *b { '1' } else { '0' })
                    .collect();
                let literal = format!("B'{}'", binary_str);

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
                valid_bits in prop::collection::vec(
                    prop::bool::ANY,
                    1..=16
                ),
                invalid_char in "[2-9a-zA-Z]",
                insert_position in 0..=16usize,
            ) {
                let mut binary_str: String = valid_bits.iter()
                    .map(|b| if *b { '1' } else { '0' })
                    .collect();

                let pos = insert_position.min(binary_str.len());
                binary_str.insert_str(pos, &invalid_char);

                let literal = format!("B'{}'", binary_str);

                let result = VectorLiteralParser::parse_binary(&literal, None);
                prop_assert!(result.is_err(),
                    "Should fail for invalid binary digit: {}", literal);

                let err_msg = result.unwrap_err().to_string();
                prop_assert!(err_msg.contains("Invalid binary digit"),
                    "Error should mention invalid binary digit, got: {}", err_msg);
            }

            #[test]
            fn test_hex_literal_invalid_digit(
                valid_bytes in prop::collection::vec(
                    any::<u8>(),
                    1..=8
                ),
                invalid_char in "[G-Zg-z]",
            ) {
                let mut hex_str: String = valid_bytes.iter()
                    .map(|b| format!("{:02X}", b))
                    .collect();

                hex_str.push_str(&invalid_char);
                hex_str.push('0');

                let literal = format!("'\\x{}'", hex_str);

                let result = VectorLiteralParser::parse_binary(&literal, None);
                prop_assert!(result.is_err(),
                    "Should fail for invalid hex digit: {}", literal);

                let err_msg = result.unwrap_err().to_string();
                prop_assert!(err_msg.contains("Invalid hex digit"),
                    "Error should mention invalid hex digit, got: {}", err_msg);
            }

            #[test]
            fn test_binary_literal_partial_byte_handling(
                bit_count in 1..=15usize,
            ) {
                let bits: Vec<bool> = (0..bit_count)
                    .map(|i| i % 2 == 0)
                    .collect();

                let binary_str: String = bits.iter()
                    .map(|b| if *b { '1' } else { '0' })
                    .collect();
                let literal = format!("B'{}'", binary_str);

                let result = VectorLiteralParser::parse_binary(&literal, Some(bit_count));
                prop_assert!(result.is_ok(),
                    "Should parse partial byte successfully: {}", literal);

                let bytes = result.unwrap();

                let expected_byte_count = bit_count.div_ceil(8);
                prop_assert_eq!(bytes.len(), expected_byte_count,
                    "Byte count should round up for partial bytes");

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
                bytes_input in prop::collection::vec(
                    any::<u8>(),
                    1..=32
                ),
                bit_count in 8..=256usize,
            ) {
                let actual_bit_count = bit_count.min(bytes_input.len() * 8);

                let binary_str = format_binary_vector(&bytes_input, actual_bit_count, false);

                let hex_str = format_binary_vector(&bytes_input, actual_bit_count, true);

                prop_assert_eq!(binary_str.len(), actual_bit_count,
                    "Binary string length should match bit count");

                prop_assert!(hex_str.starts_with("0x"),
                    "Hex string should start with 0x");

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
}
