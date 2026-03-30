// Copyright (c) 2026 Richard Albright. All rights reserved.

use hyperstreamdb::core::iceberg::{murmur3_32_x86, IcebergTransform};
use arrow::array::{StringArray, Date32Array, TimestampMicrosecondArray};
use serde_json::json;

#[test]
fn test_murmur3_standard_vectors() {
    // Known Murmur3 x86_32 vectors (seed 0)
    // "iceberg" -> 1216061395 (Java Spec)
    // "iceberg" -> 1210000089 (Official Rust Crate)
    // Ref: https://github.com/apache/iceberg-rust/blob/main/crates/iceberg/src/transform/bucket.rs#L789
    
    // We align with the official Rust implementation.
    let hash1 = murmur3_32_x86(b"iceberg", 0);
    assert_eq!(hash1, 1210000089, "Hash of 'iceberg' matches Official Rust Crate");

    // "test" -> 2808560321 (Java Spec)
    // "test" -> 3127628307 (Official Rust Crate / murmur3 crate)
    let hash2 = murmur3_32_x86(b"test", 0);
    assert_eq!(hash2, 3127628307, "Hash of 'test' matches Official Rust Crate");
}

#[test]
fn test_bucket_transform_logic() {
    // Spec: (hash & Integer.MAX_VALUE) % N
    // For "iceberg" (1210000089): 
    // 1210000089 & 0x7FFFFFFF = 1210000089
    // 1210000089 % 16 = 9
    
    let array = StringArray::from(vec!["iceberg"]);
    let transform = IcebergTransform::Bucket(16);
    
    let res = transform.apply(&array, 0);
    assert_eq!(res, json!(9));
    
    // For "test" (3127628307 => 0xBA6BD213)
    // 0xBA6BD213 & 0x7FFFFFFF = 0x3A6BD213 = 980111891
    // 980111891 % 16 = 3
    
    let array = StringArray::from(vec!["test"]);
    let res = transform.apply(&array, 0);
    assert_eq!(res, json!(3));
}

#[test]
fn test_temporal_transforms() {
    // 1. Year
    // 2023-11-23
    let date_array = Date32Array::from(vec![19684]); // days since epoch for 2023-11-23
    let t = IcebergTransform::Year;
    assert_eq!(t.apply(&date_array, 0), json!(53)); // 2023 - 1970 = 53

    // 2. Month
    // 2023-11-23 -> 53 years + 11th month (index 10?)
    // Spec: (Year-1970)*12 + Month-1
    // (2023-1970)*12 + 11 - 1 = 53*12 + 10 = 636 + 10 = 646.
    let t = IcebergTransform::Month;
    assert_eq!(t.apply(&date_array, 0), json!(646));
    
    // 3. Day
    // Should be absolute days from epoch
    let t = IcebergTransform::Day;
    assert_eq!(t.apply(&date_array, 0), json!(19684));
    
    // 4. Hour
    // 2023-11-23 10:00:00 UTC
    // Timestamp micros.
    // 2023-11-23 10:00:00 => 1700733600 seconds
    let ts_array = TimestampMicrosecondArray::from(vec![1_700_733_600_000_000]).with_timezone("UTC");
    let t = IcebergTransform::Hour;
    // Hours from epoch: 1700733600 / 3600 = 472426
    assert_eq!(t.apply(&ts_array, 0), json!(472426));
}
