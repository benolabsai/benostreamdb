// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Tests for the vector companion aggregates (`centroid`, `vector_min`,
//! `vector_max`, `vector_stddev`, `vector_median`) and the text-scoring UDFs
//! (`bm25_score`, `tf_idf`).

use benostreamdb::core::sql::session::BenoStreamSession;

fn session() -> BenoStreamSession {
    BenoStreamSession::new(None)
}

#[tokio::test]
async fn vector_companion_aggregates() -> anyhow::Result<()> {
    let session = session();
    let (batches, _) = session
        .sql(
            "SELECT centroid(v) AS c, vector_min(v) AS mn, vector_max(v) AS mx, \
                    vector_stddev(v) AS sd, vector_median(v) AS md \
             FROM (VALUES (ARRAY[1.0, 2.0]), (ARRAY[3.0, 4.0]), (ARRAY[5.0, 6.0])) AS t(v);",
        )
        .await?;
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
    Ok(())
}

#[tokio::test]
async fn bm25_score_udf() -> anyhow::Result<()> {
    let session = session();
    let (batches, _) = session
        .sql("SELECT bm25_score('the quick brown fox', 'quick fox') AS s;")
        .await?;
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
    Ok(())
}

#[tokio::test]
async fn tf_idf_udf() -> anyhow::Result<()> {
    let session = session();
    let (batches, _) = session
        .sql("SELECT tf_idf('the quick brown fox') AS v;")
        .await?;
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
    Ok(())
}

#[test]
fn aggregates_are_registered() {
    use benostreamdb::core::sql::vector_udf;
    let names: Vec<String> = vector_udf::all_vector_aggregates()
        .iter()
        .map(|a| a.name().to_string())
        .collect();
    for expected in [
        "vector_sum",
        "vector_avg",
        "centroid",
        "vector_min",
        "vector_max",
        "vector_stddev",
        "vector_median",
    ] {
        assert!(
            names.contains(&expected.to_string()),
            "missing aggregate {expected}; got {names:?}"
        );
    }
}

#[test]
fn text_udfs_are_registered() {
    use benostreamdb::core::sql::vector_udf;
    let names: Vec<String> = vector_udf::all_vector_udfs()
        .iter()
        .map(|u| u.name().to_string())
        .collect();
    for expected in ["bm25_score", "tf_idf"] {
        assert!(
            names.contains(&expected.to_string()),
            "missing UDF {expected}; got {names:?}"
        );
    }
}
