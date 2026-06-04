// Copyright (c) 2026 Richard Albright. All rights reserved.

use datafusion::error::{DataFusionError, Result};

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
/// use hyperstreamdb::core::sql::literal::binary::parse_binary_vector;
///
/// let result = parse_binary_vector("B'10110101'", Some(8)).unwrap();
/// let result = parse_binary_vector("'\\xB5'", Some(8)).unwrap();
/// ```
pub fn parse_binary_vector(input: &str, expected_bits: Option<usize>) -> Result<Vec<u8>> {
    let trimmed = input.trim();

    // Check for binary string format: B'...' or b'...'
    if trimmed.starts_with("B'") || trimmed.starts_with("b'") {
        if !trimmed.ends_with('\'') {
            return Err(DataFusionError::Plan(
                "Binary literal must end with single quote: B'...'".to_string(),
            ));
        }

        // Extract binary string between quotes
        let binary_str = &trimmed[2..trimmed.len() - 1];

        // Validate that all characters are 0 or 1
        for (idx, ch) in binary_str.chars().enumerate() {
            if ch != '0' && ch != '1' {
                return Err(DataFusionError::Plan(format!(
                    "Invalid binary digit at position {}: expected '0' or '1', got '{}'",
                    idx, ch
                )));
            }
        }

        let bit_count = binary_str.len();

        // Validate bit count if expected_bits is provided
        if let Some(expected) = expected_bits {
            if bit_count != expected {
                return Err(DataFusionError::Plan(format!(
                    "Binary literal bit count mismatch: expected {} bits, got {}",
                    expected, bit_count
                )));
            }
        }

        // Convert binary string to bytes
        let byte_count = bit_count.div_ceil(8);
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
                "Hex literal must end with single quote: '\\x...'".to_string(),
            ));
        }

        // Extract hex string between \x and closing quote
        let hex_str = &trimmed[3..trimmed.len() - 1];

        // Parse hex string to bytes
        let mut bytes = Vec::new();
        let mut chars = hex_str.chars().peekable();

        while chars.peek().is_some() {
            let high = chars
                .next()
                .ok_or_else(|| DataFusionError::Plan("Incomplete hex byte".to_string()))?;
            let low = chars.next().ok_or_else(|| {
                DataFusionError::Plan(
                    "Incomplete hex byte: hex digits must come in pairs".to_string(),
                )
            })?;

            let hex_byte = format!("{}{}", high, low);
            let byte = u8::from_str_radix(&hex_byte, 16).map_err(|_| {
                DataFusionError::Plan(format!(
                    "Invalid hex digit in '{}': expected 0-9, a-f, A-F",
                    hex_byte
                ))
            })?;

            bytes.push(byte);
        }

        // Validate bit count if expected_bits is provided
        if let Some(expected) = expected_bits {
            let actual_bits = bytes.len() * 8;
            if actual_bits != expected {
                return Err(DataFusionError::Plan(format!(
                    "Binary literal bit count mismatch: expected {} bits, got {}",
                    expected, actual_bits
                )));
            }
        }

        Ok(bytes)
    } else {
        Err(DataFusionError::Plan(
            "Binary literal must be in format B'...' or '\\x...'".to_string(),
        ))
    }
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
/// use hyperstreamdb::core::sql::literal::binary::format_binary_vector;
///
/// let bytes = vec![0b10110101];
/// let binary_str = format_binary_vector(&bytes, 8, false); // "10110101"
/// let hex_str = format_binary_vector(&bytes, 8, true);     // "0xB5"
/// ```
pub fn format_binary_vector(data: &[u8], bits: usize, use_hex: bool) -> String {
    if use_hex {
        // Format as hex string
        let mut result = String::from("0x");
        let byte_count = bits.div_ceil(8);
        for byte in &data[0..byte_count.min(data.len())] {
            result.push_str(&format!("{:02X}", byte));
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
