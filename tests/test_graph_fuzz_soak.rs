// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use arrow::array::{Float32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::SessionContext;
use rand::Rng;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[tokio::test]
#[ignore = "Long-running soak and fuzz test"]
async fn test_graph_udafs_fuzz_soak() {
    let soak_seconds = std::env::var("BSDB_SOAK_SECONDS")
        .unwrap_or_else(|_| "10".to_string())
        .parse::<u64>()
        .unwrap();
    let start_time = Instant::now();
    let duration = Duration::from_secs(soak_seconds);

    let ctx = SessionContext::new();
    for udaf in benostreamdb::core::sql::graph_udf::all_graph_aggregates() {
        ctx.register_udaf(udaf);
    }

    let mut iteration = 0;
    while start_time.elapsed() < duration {
        let mut rng = rand::thread_rng();
        let num_edges = rng.gen_range(10..1000);
        let max_node = rng.gen_range(5..100);

        let mut sources = Vec::with_capacity(num_edges);
        let mut targets = Vec::with_capacity(num_edges);
        let mut weights = Vec::with_capacity(num_edges);

        for _ in 0..num_edges {
            sources.push(rng.gen_range(0..max_node));
            targets.push(rng.gen_range(0..max_node));
            weights.push(rng.gen_range(0.1..10.0) as f32);
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("source", DataType::UInt64, false),
            Field::new("target", DataType::UInt64, false),
            Field::new("weight", DataType::Float32, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(sources)),
                Arc::new(UInt64Array::from(targets)),
                Arc::new(Float32Array::from(weights)),
            ],
        )
        .unwrap();

        let _ = ctx.deregister_table("edges");
        ctx.register_batch("edges", batch).unwrap();

        // queries to test
        let queries = vec![
            "SELECT graph_shortest_path(source, target, arrow_cast(0, 'UInt64'), arrow_cast(3, 'UInt64'), 'in_memory', 'in_memory') FROM edges",
            "SELECT graph_pagerank(source, target, 'in_memory', 'in_memory') FROM edges",
            "SELECT graph_connected_components(source, target, 'in_memory', 'in_memory') FROM edges",
            "SELECT graph_degree_centrality(source, target, 'in_memory', 'in_memory') FROM edges",
            "SELECT graph_strongly_connected_components(source, target, 'in_memory', 'in_memory') FROM edges",
            "SELECT graph_leiden_communities(source, target, weight, arrow_cast(1.0, 'Float64'), 'in_memory', 'in_memory') FROM edges",
            "SELECT drift_search(source, target, 'text', make_array(arrow_cast(1, 'UInt64')), arrow_cast(1, 'UInt32'), 'in_memory', 'in_memory') FROM edges",
            "SELECT regional_drift(source, target, 'text', make_array(arrow_cast(1, 'UInt64')), arrow_cast(1, 'UInt32'), arrow_cast(1, 'UInt32'), 'in_memory', 'in_memory') FROM edges",
        ];

        for query in &queries {
            let df = match ctx.sql(query).await {
                Ok(df) => df,
                Err(e) => {
                    if e.to_string().contains("Empty graph") {
                        continue;
                    }
                    panic!("Planning failed for {} on iter {}: {}", query, iteration, e);
                }
            };
            match df.collect().await {
                Ok(_) => {}
                Err(e) => {
                    if e.to_string().contains("Empty graph") {
                        continue;
                    }
                    panic!(
                        "Execution failed for {} on iter {}: {}",
                        query, iteration, e
                    );
                }
            }
        }
        iteration += 1;
    }

    println!(
        "Soak test finished after {} iterations over {:?}",
        iteration, duration
    );
}
