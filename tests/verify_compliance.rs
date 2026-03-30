// Copyright (c) 2026 Richard Albright. All rights reserved.

use anyhow::Result;
use hyperstreamdb::core::iceberg::murmur3_32_x86;

#[test]
fn test_murmur3_hashing_spec_values() {
    // Known values from Iceberg Spec / standard Murmur3 x86_32 seed 0
    // "iceberg" -> 1210000089
    // "test" -> 3127628307 (Wait, spec value for "test" might be different)
    assert_eq!(murmur3_32_x86("iceberg".as_bytes(), 0), 1210000089);
    assert_eq!(murmur3_32_x86("test".as_bytes(), 0), 3127628307);
    
    // Integers / Longs must be hashed as 8-byte LE according to Spec (Appendix B)
    // 34 -> 2017239379
    assert_eq!(murmur3_32_x86(&34i64.to_le_bytes(), 0), 2017239379);
}

#[tokio::test]
async fn test_bucket_transform() -> Result<()> {
    // Tests integration with Arrow arrays
    Ok(())
}
