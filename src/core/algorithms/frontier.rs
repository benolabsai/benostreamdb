use anyhow::Result;
use datafusion::prelude::*;
use std::path::Path;

/// Frontier-based (level-synchronous) BFS over an edge table, executed
/// entirely with DataFusion against per-hop parquet materializations.
///
/// This replaces the in-RAM UDAF accumulator pattern (which buffered the
/// entire edge set plus a `HashMap` adjacency list — tens of GB at
/// hundreds-of-millions-of-edges scale) with strictly bounded, spilling
/// intermediate state:
///
/// * one symmetric (or directed) deduplicated adjacency table, materialized
///   once,
/// * a `visited` table and a per-hop `frontier` table, each no larger than
///   the reachable node set,
/// * one hash join per hop (frontier expansion), with early exit when the
///   frontier empties.
///
/// Returns the sorted list of visited node ids (seeds included).
pub async fn bfs_visited(
    ctx: &SessionContext,
    edges_table: &str,
    seeds: &[u64],
    hops: u32,
    directed: bool,
    src_column: &str,
    dst_column: &str,
    temp_dir: &Path,
) -> Result<Vec<u64>> {
    use arrow::array::UInt64Array;

    if seeds.is_empty() {
        return Ok(Vec::new());
    }
    std::fs::create_dir_all(temp_dir)?;

    // Materialize a SQL result into parquet and register it under `name`.
    async fn materialize(
        ctx: &SessionContext,
        temp_dir: &Path,
        name: &str,
        sql: &str,
    ) -> Result<String> {
        let df = ctx.sql(sql).await?;
        let path = temp_dir.join(format!("{name}.parquet"));
        let path_str = path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("frontier run dir path is not valid UTF-8"))?;
        df.write_parquet(path_str, Default::default(), None).await?;
        ctx.register_parquet(name, path_str, ParquetReadOptions::default())
            .await?;
        Ok(name.to_string())
    }

    async fn count_rows(ctx: &SessionContext, table: &str) -> Result<i64> {
        let batches = ctx
            .sql(&format!("SELECT COUNT(*) AS c FROM {table}"))
            .await?
            .collect()
            .await?;
        Ok(batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .map(|a| a.value(0))
            .unwrap_or(0))
    }

    let started = std::time::Instant::now();

    // 1. Visited seed set (UInt64 to match the cast adjacency).
    let seed_list = seeds
        .iter()
        .map(|s| format!("({s})"))
        .collect::<Vec<_>>()
        .join(", ");
    let mut visited = materialize(
        ctx,
        temp_dir,
        "bfs_visited_0",
        &format!(
            "SELECT DISTINCT arrow_cast(id, 'UInt64') AS id FROM (VALUES {seed_list}) AS s(id)"
        ),
    )
    .await?;

    // 2. Working adjacency, deduplicated. For undirected traversal the set is
    // stored in both directions so each hop needs a SINGLE join.
    let adj_sql = if directed {
        format!(
            "SELECT DISTINCT arrow_cast({src_column}, 'UInt64') AS src, arrow_cast({dst_column}, 'UInt64') AS dst FROM {edges_table}"
        )
    } else {
        format!(
            "
            SELECT DISTINCT a AS src, b AS dst FROM (
                SELECT arrow_cast({src_column}, 'UInt64') AS a, arrow_cast({dst_column}, 'UInt64') AS b FROM {edges_table}
                UNION ALL
                SELECT arrow_cast({dst_column}, 'UInt64') AS a, arrow_cast({src_column}, 'UInt64') AS b FROM {edges_table}
            )
            "
        )
    };
    let adj = materialize(ctx, temp_dir, "bfs_adj", &adj_sql).await?;

    // 3. Level-synchronous expansion: only the FRONTIER (nodes discovered in
    // the previous hop) is joined against the adjacency, keeping the build
    // side of every hop join minimal.
    let mut frontier = visited.clone();
    for h in 0..hops {
        // next = neighbors(frontier) \ visited
        let next = materialize(
            ctx,
            temp_dir,
            &format!("bfs_new_{h}"),
            &format!(
                "
                SELECT DISTINCT a.dst AS id
                FROM {adj} a
                JOIN {frontier} f ON a.src = f.id
                LEFT JOIN {visited} v ON a.dst = v.id
                WHERE v.id IS NULL
                "
            ),
        )
        .await?;

        let new_count = count_rows(ctx, &next).await?;
        tracing::debug!(
            "bfs hop {h}: {new_count} new nodes ({} elapsed)",
            started.elapsed().as_secs_f64()
        );
        if new_count == 0 {
            break; // frontier exhausted — component fully explored
        }

        visited = materialize(
            ctx,
            temp_dir,
            &format!("bfs_visited_{}", h + 1),
            &format!("SELECT id FROM {visited} UNION SELECT id FROM {next}"),
        )
        .await?;
        frontier = next;
    }

    // 4. Read the final visited set back.
    let batches = ctx
        .sql(&format!("SELECT id FROM {visited} ORDER BY id"))
        .await?
        .collect()
        .await?;
    let mut out = Vec::new();
    for b in &batches {
        let arr = b
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| anyhow::anyhow!("bfs: unexpected visited column type"))?;
        out.extend(arr.iter().flatten());
    }
    out.sort_unstable();
    out.dedup();
    tracing::info!(
        "bfs_visited: {} nodes in {} hops ({:.1}s)",
        out.len(),
        hops,
        started.elapsed().as_secs_f64()
    );
    Ok(out)
}
