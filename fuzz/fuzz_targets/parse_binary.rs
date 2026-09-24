// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Fuzz the binary/bit-vector literal parser (`B'10110101'`, `'\xB5'`), with and
//! without an expected bit count (the bit-count check is a common crash source).

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        let _ = benostreamdb::core::sql::literal::parse_binary_vector(s, None);
        let _ = benostreamdb::core::sql::literal::parse_binary_vector(s, Some(data.len()));
        let _ = benostreamdb::core::sql::literal::parse_binary_vector(s, Some(8));
    }
});
