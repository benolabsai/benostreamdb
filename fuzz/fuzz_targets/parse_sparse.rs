// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Fuzz the sparse vector literal parser (`{1:0.5, 10:0.3}`), which validates
//! index bounds and duplicate indices against a dimension.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        // Vary the dimension, including the degenerate 0 case, and pin a
        // realistic one so bounds/duplicate checks are exercised.
        let dim = data.first().copied().unwrap_or(0) as usize;
        let _ = benostreamdb::core::sql::literal::parse_sparse_vector(s, dim);
        let _ = benostreamdb::core::sql::literal::parse_sparse_vector(s, 1024);
    }
});
