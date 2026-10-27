// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Fuzz the untrusted inputs the SQL connectors (Trino, Spark, Flight SQL) hand
//! to the engine:
//!
//! * the `filter` argument of `system.vector_search` / `system.hybrid_search`
//!   (`BenoStreamDBMergeSink`, `show_indexes`, …), which is parsed as a SQL
//!   WHERE clause, and
//! * the JSON query vector, which arrives as a string from the JVM.
//!
//! Both must be total: arbitrary bytes may produce an `Err`, never a panic,
//! abort, or unbounded allocation.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(input) = std::str::from_utf8(data) else {
        return;
    };

    // 1. Query vector JSON: `serde_json::from_str::<Vec<f32>>` on the JNI path.
    let _ = serde_json::from_str::<Vec<f32>>(input);

    // 2. Scalar predicate pushdown. The connectors pass the user's `filter`
    //    argument straight through, so this parser sees arbitrary text.
    let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    else {
        return;
    };

    let schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Int64, false),
        arrow::datatypes::Field::new("title", arrow::datatypes::DataType::Utf8, true),
        arrow::datatypes::Field::new("score", arrow::datatypes::DataType::Float64, true),
        arrow::datatypes::Field::new(
            "tags",
            arrow::datatypes::DataType::List(std::sync::Arc::new(arrow::datatypes::Field::new(
                "item",
                arrow::datatypes::DataType::Utf8,
                true,
            ))),
            true,
        ),
    ]));

    runtime.block_on(async {
        let _ = benostreamdb::core::planner::FilterExpr::parse_sql(input, schema).await;
    });
});
