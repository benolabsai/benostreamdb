// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Fuzz the pgvector-compatible dense vector literal parser
//! (`'[1.0, 2.0, 3.0]'::vector`). This is reached from SQL `ORDER BY v <-> '[...]'`
//! and from the `_search` request path, so malformed input must return `Err`
//! rather than panic.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        let _ = benostreamdb::core::sql::literal::parse_vector_literal(s);
    }
});
