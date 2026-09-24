// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Fuzz the SQL string rewriters applied to user SQL before planning:
//! `strip_partitioned_by` (regex-based) and `rewrite_sql_string` (pgvector
//! operator/cast rewriting). Both must be total for arbitrary input.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        let _ = benostreamdb::core::sql::partition_rewriter::strip_partitioned_by(s);
        let _ = benostreamdb::core::sql::pgvector_rewriter::rewrite_sql_string(s);
    }
});
