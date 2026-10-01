// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Diagnostic: run a BM25 keyword search against a table's inverted index.

use benostreamdb::core::search::KeywordSearchParams;
use benostreamdb::Table;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let uri = std::env::args()
        .nth(1)
        .expect("usage: verify_bm25 <table-uri> [query] [column]");
    let query = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "Albert Einstein".to_string());
    let column = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "title".to_string());

    let table = Table::builder(uri).build_async().await?;
    let results = table
        .execute_keyword_search_as_scored(KeywordSearchParams {
            column: column.clone(),
            query: query.clone(),
            ..Default::default()
        })
        .await?;

    println!(
        "column={column:?} query={query:?} -> {} hit(s)",
        results.len()
    );
    for r in results.iter().take(5) {
        println!(
            "  seg={} row={} score={:.4}",
            r.segment_id, r.row_id, r.score
        );
    }
    Ok(())
}
