// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Integration test for the shared [`GraphMode`] resolver on [`Table`].
//!
//! The graph algorithms are written against the [`GraphView`] trait, so the same
//! algorithm can run on an in-memory adjacency map (`GraphMode::InMemory`) or the
//! mmap-backed CSR (`GraphMode::OutOfCore` / `GraphMode::Cached`). This test
//! builds one edge table with a CSR graph index and asserts that every mode
//! resolves to a graph view that reports the *same* neighbors and degrees — i.e.
//! the wrapper functions really do call the in-core and out-of-core paths and
//! they agree.

use std::sync::Arc;

use arrow::array::UInt64Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use benostreamdb::core::manifest::IndexAlgorithm;
use benostreamdb::core::sql::graph_udf::graph_view::GraphMode;
use benostreamdb::core::table::Table;
use tempfile::tempdir;

/// Build an edge table with a CSR graph index over `source`/`target`.
///
/// Graph: 0 -> 1, 2 ; 1 -> 2 ; 2 -> 0, 1, 3 ; 3 -> 0
async fn build_edge_table(uri: &str) -> anyhow::Result<Table> {
    let table = Table::new_async(uri.to_string()).await?;
    table.set_autocommit(false);
    table
        .add_index(
            "source".to_string(),
            IndexAlgorithm::CsrGraph {
                src_column: "source".to_string(),
                dst_column: "target".to_string(),
            },
        )
        .await?;

    let schema = Arc::new(Schema::new(vec![
        Field::new("source", DataType::UInt64, false),
        Field::new("target", DataType::UInt64, false),
    ]));
    let src: Vec<u64> = vec![0, 0, 1, 2, 2, 2, 3];
    let dst: Vec<u64> = vec![1, 2, 2, 0, 1, 3, 0];
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt64Array::from(src)),
            Arc::new(UInt64Array::from(dst)),
        ],
    )?;
    table.write_async(vec![batch]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;

    // The index build commits a follow-up manifest version in a background
    // task. Poll for it rather than sleeping a fixed interval, which is flaky
    // under parallel test load.
    for _ in 0..100 {
        if table.load_graph_index("source").await?.is_some() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    drop(table);

    // Re-open so index registration is inferred from the physical files.
    Table::new_async(uri.to_string()).await
}

fn sorted(mut v: Vec<u64>) -> Vec<u64> {
    v.sort_unstable();
    v
}

#[tokio::test]
async fn graph_view_in_memory_and_out_of_core_agree() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);

    let table = build_edge_table(&uri).await?;

    // Seeds [0] with 2 hops reach the whole graph (0,1,2,3).
    let seeds = vec![0u64];
    let hops = 2u32;

    let in_memory = table.graph_view(GraphMode::InMemory, &seeds, hops).await?;
    let out_of_core = table.graph_view(GraphMode::OutOfCore, &seeds, hops).await?;
    let cached = table.graph_view(GraphMode::Cached, &seeds, hops).await?;
    let auto = table.graph_view(GraphMode::Auto, &seeds, hops).await?;

    // Every mode must agree on neighbors and degree for every node in the graph.
    for node in 0u64..4 {
        let expected = sorted(in_memory.get_neighbors(node));
        assert_eq!(
            sorted(out_of_core.get_neighbors(node)),
            expected,
            "OutOfCore neighbors diverged from InMemory at node {node}"
        );
        assert_eq!(
            sorted(cached.get_neighbors(node)),
            expected,
            "Cached neighbors diverged from InMemory at node {node}"
        );
        assert_eq!(
            sorted(auto.get_neighbors(node)),
            expected,
            "Auto neighbors diverged from InMemory at node {node}"
        );

        assert_eq!(
            out_of_core.get_degree(node),
            in_memory.get_degree(node),
            "OutOfCore degree diverged from InMemory at node {node}"
        );
        assert_eq!(
            cached.get_degree(node),
            in_memory.get_degree(node),
            "Cached degree diverged from InMemory at node {node}"
        );
        assert_eq!(
            auto.get_degree(node),
            in_memory.get_degree(node),
            "Auto degree diverged from InMemory at node {node}"
        );
    }

    // Spot-check the concrete adjacency so a regression in both paths at once
    // cannot pass silently.
    assert_eq!(sorted(in_memory.get_neighbors(0)), vec![1, 2]);
    assert_eq!(sorted(in_memory.get_neighbors(2)), vec![0, 1, 3]);
    assert_eq!(in_memory.get_degree(2), 3);
    assert_eq!(out_of_core.get_degree(2), 3);

    Ok(())
}

#[tokio::test]
async fn graph_view_auto_picks_a_working_mode() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);

    let table = build_edge_table(&uri).await?;

    // Auto must resolve to *some* concrete mode and return a usable graph.
    let graph = table.graph_view(GraphMode::Auto, &[0], 2).await?;
    assert_eq!(graph.get_degree(0), 2);
    assert_eq!(sorted(graph.get_neighbors(0)), vec![1, 2]);

    Ok(())
}

/// The in-memory and out-of-core paths must run the *same* algorithm over the
/// *same* logical graph, so `regional_drift_with_mode` must return identical
/// results for every mode. This is the function-equality guarantee: the mode is
/// a memory strategy, not a different answer.
#[tokio::test]
async fn regional_drift_is_mode_invariant() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);

    let table = build_edge_table(&uri).await?;

    let seeds = vec![0u64];
    let run = |mode: GraphMode| {
        let table = &table;
        let seeds = seeds.clone();
        async move {
            table
                .regional_drift_with_mode("test query", &seeds, 5, 2, 2, 2, mode)
                .await
        }
    };

    let in_memory = run(GraphMode::InMemory).await?;
    let out_of_core = run(GraphMode::OutOfCore).await?;
    let cached = run(GraphMode::Cached).await?;
    let auto = run(GraphMode::Auto).await?;

    let mut expected = in_memory.all_discovered_nodes.clone();
    expected.sort_unstable();
    assert!(
        !expected.is_empty(),
        "DRIFT discovered no nodes — the equality check would be vacuous"
    );

    for (name, result) in [
        ("OutOfCore", out_of_core),
        ("Cached", cached),
        ("Auto", auto),
    ] {
        let mut got = result.all_discovered_nodes.clone();
        got.sort_unstable();
        assert_eq!(
            got, expected,
            "{name} DRIFT result diverged from InMemory — the modes must share \
             the same algorithm and return the same nodes"
        );
    }

    Ok(())
}

#[tokio::test]
async fn udafs_are_mode_invariant() -> anyhow::Result<()> {
    use benostreamdb::core::sql::session::BenoStreamSession;
    // Surface the graph-loading debug logs if a mode diverges.
    let _ = tracing_subscriber::fmt()
        .with_env_filter("benostreamdb=debug")
        .with_test_writer()
        .try_init();
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);

    let table = build_edge_table(&uri).await?;

    let session = BenoStreamSession::new(None);
    session.register_table("edges", Arc::new(table)).unwrap();

    let modes = vec!["in_memory", "out_of_core", "cached", "auto"];

    let queries = vec![
        "SELECT graph_triangle_count(source, target, '{URI}', '{MODE}') AS res FROM edges",
        "SELECT graph_closeness_centrality(source, target, '{URI}', '{MODE}') AS res FROM edges",
        "SELECT graph_betweenness_centrality(source, target, '{URI}', '{MODE}') AS res FROM edges",
        "SELECT graph_all_shortest_paths(source, target, arrow_cast(0, 'UInt64'), arrow_cast(3, 'UInt64'), '{URI}', '{MODE}') AS res FROM edges",
        "SELECT graph_shortest_path(source, target, arrow_cast(0, 'UInt64'), arrow_cast(3, 'UInt64'), '{URI}', '{MODE}') AS res FROM edges",
        "SELECT graph_leiden_communities(source, target, arrow_cast(1.0, 'Float32'), arrow_cast(1.0, 'Float64'), '{URI}', '{MODE}') AS res FROM edges",
        "SELECT graph_label_propagation(source, target, '{URI}', '{MODE}') AS res FROM edges",
        "SELECT graph_personalized_pagerank(source, target, make_array(arrow_cast(0, 'UInt64')), arrow_cast(0.85, 'Float64'), arrow_cast(30, 'UInt32'), true, make_array(arrow_cast(1.0, 'Float64')), '{URI}', '{MODE}') AS res FROM edges",
        "SELECT graph_degree_centrality(source, target, '{URI}', '{MODE}') AS res FROM edges",
        "SELECT graph_preferential_attachment(source, target, arrow_cast(0, 'UInt64'), arrow_cast(3, 'UInt64'), '{URI}', '{MODE}') AS res FROM edges",
        "SELECT graph_connected_components(source, target, '{URI}', '{MODE}') AS res FROM edges",
        "SELECT graph_neighbors(source, target, arrow_cast(0, 'UInt64'), arrow_cast(1, 'UInt32'), '{URI}', '{MODE}') AS res FROM edges",
        "SELECT graph_subgraph(source, target, make_array(arrow_cast(0, 'UInt64')), arrow_cast(1, 'UInt32'), true, '{URI}', '{MODE}') AS res FROM edges",
        "SELECT graph_jaccard_coefficient(source, target, arrow_cast(0, 'UInt64'), arrow_cast(3, 'UInt64'), '{URI}', '{MODE}') AS res FROM edges",
        "SELECT graph_strongly_connected_components(source, target, '{URI}', '{MODE}') AS res FROM edges",
        "SELECT graph_louvain_communities(source, target, arrow_cast(1.0, 'Float32'), arrow_cast(1.0, 'Float64'), '{URI}', '{MODE}') AS res FROM edges",
        "SELECT graph_pagerank(source, target, '{URI}', '{MODE}') AS res FROM edges",
        "SELECT drift_search(source, target, 'text', make_array(arrow_cast(1, 'UInt64')), arrow_cast(1, 'UInt32'), '{URI}', '{MODE}') AS res FROM edges",
        "SELECT regional_drift(source, target, 'text', make_array(arrow_cast(1, 'UInt64')), arrow_cast(1, 'UInt32'), arrow_cast(1, 'UInt32'), '{URI}', '{MODE}') AS res FROM edges",
    ];

    for query_tpl in queries {
        let mut expected = None;
        for mode in &modes {
            let query = query_tpl.replace("{URI}", &uri).replace("{MODE}", mode);
            let (batches, _) = session.sql(&query).await.unwrap();
            let formatted = arrow::util::pretty::pretty_format_batches(&batches)
                .unwrap()
                .to_string();

            if let Some(ref exp) = expected {
                assert_eq!(
                    &formatted, exp,
                    "UDAF results diverged for query: {} in mode: {}",
                    query_tpl, mode
                );
            } else {
                expected = Some(formatted);
            }
        }
    }

    Ok(())
}
