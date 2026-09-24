use anyhow::Result;
use arrow::array::{Float64Array, Int64Array};
use datafusion::prelude::*;
use std::fs;
use std::path::Path;
use std::time::Instant;

/// Weakly connected components via label propagation with
/// pointer jumping and edge contraction.
///
/// Compared to textbook per-hop propagation, this converges in O(log d)
/// rounds instead of O(d), and each round's expensive edge join runs over
/// a *contracted* edge set that shrinks to inter-component edges only.
/// The working edge set is stored symmetrically, so propagation needs a
/// single join per round rather than two.
///
/// Returns the name of a registered parquet table with columns
/// `(id, component_id)`.
pub async fn compute_connected_components(
    ctx: &SessionContext,
    edges_table: &str,
    temp_dir: &Path,
    max_iterations: usize,
    src_column: &str,
    dst_column: &str,
) -> Result<String> {
    let run_dir = temp_dir.join(format!("cc_{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&run_dir)?;

    // Materialize a SQL result into parquet and register it under `name`.
    async fn materialize(
        ctx: &SessionContext,
        run_dir: &Path,
        name: &str,
        sql: &str,
    ) -> Result<String> {
        let df = ctx.sql(sql).await?;
        let path = run_dir.join(format!("{name}.parquet"));
        let path_str = path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("algorithm run dir path is not valid UTF-8"))?;
        df.write_parquet(path_str, Default::default(), None).await?;
        ctx.register_parquet(name, path_str, ParquetReadOptions::default())
            .await?;
        Ok(name.to_string())
    }

    // Cheap single-pass convergence probe: (SUM(component_id), COUNT(*)).
    // Cast to DOUBLE explicitly so the output type is stable regardless of
    // the label column's integer width (a failed downcast would silently
    // read 0 and falsely report convergence).
    async fn checksum(ctx: &SessionContext, state_table: &str) -> Result<(f64, i64)> {
        let df = ctx
            .sql(&format!(
                "SELECT SUM(CAST(component_id AS DOUBLE)) AS s, COUNT(*) AS c FROM {state_table}"
            ))
            .await?;
        let batches = df.collect().await?;
        let sum = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .map(|a| a.value(0))
            .unwrap_or(f64::NAN);
        let cnt = batches[0]
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .map(|a| a.value(0))
            .unwrap_or(0);
        Ok((sum, cnt))
    }

    // 1. Initial state: label(v) = v for every endpoint.
    let init_sql = format!(
        "
        SELECT id, id AS component_id FROM (
            SELECT {src_column} AS id FROM {edges_table}
            UNION
            SELECT {dst_column} AS id FROM {edges_table}
        )
        "
    );
    let mut state_cur = materialize(ctx, &run_dir, "state_0", &init_sql).await?;
    let mut prev_checksum = checksum(ctx, &state_cur).await?;

    // 2. Working edge set: symmetric, deduplicated, no self-loops.
    let edges_init_sql = format!(
        "
        SELECT DISTINCT a AS src, b AS dst FROM (
            SELECT {src_column} AS a, {dst_column} AS b FROM {edges_table}
            UNION ALL
            SELECT {dst_column} AS a, {src_column} AS b FROM {edges_table}
        )
        WHERE a <> b
        "
    );
    let mut edges_cur = materialize(ctx, &run_dir, "edges_0", &edges_init_sql).await?;

    // 3. Propagation rounds with pointer jumping + contraction.
    for round in 0..max_iterations {
        let started = Instant::now();

        // 3a. Propagation: each node adopts the minimum label seen on any
        // incident edge (edges are stored in both directions -> one join).
        let prop_sql = format!(
            "
            SELECT id, MIN(lab) AS component_id FROM (
                SELECT id, component_id AS lab FROM {state_cur}
                UNION ALL
                SELECT e.dst AS id, s.component_id AS lab
                FROM {edges_cur} e JOIN {state_cur} s ON e.src = s.id
            )
            GROUP BY id
            "
        );
        let state_p = materialize(ctx, &run_dir, &format!("state_p{round}"), &prop_sql).await?;

        // 3b. Pointer jumping (twice): label(v) <- label(label(v)).
        // Joins are state x state (node-count sized), not edge-count sized.
        let state_j1 = materialize(
            ctx,
            &run_dir,
            &format!("state_j1_{round}"),
            &format!(
                "
                SELECT s.id AS id, COALESCE(j.component_id, s.component_id) AS component_id
                FROM {state_p} s LEFT JOIN {state_p} j ON s.component_id = j.id
                "
            ),
        )
        .await?;
        let state_new = materialize(
            ctx,
            &run_dir,
            &format!("state_j2_{round}"),
            &format!(
                "
                SELECT s.id AS id, COALESCE(j.component_id, s.component_id) AS component_id
                FROM {state_j1} s LEFT JOIN {state_j1} j ON s.component_id = j.id
                "
            ),
        )
        .await?;

        // 3c. Convergence: labels only ever decrease, so an unchanged
        // checksum implies a fixed point.
        let chk = checksum(ctx, &state_new).await?;
        let elapsed = started.elapsed();
        tracing::info!(
            "connected_components round {round}: converged={}",
            chk == prev_checksum
        );
        tracing::debug!("connected_components round {round} took {elapsed:?}");
        if chk == prev_checksum {
            state_cur = state_new;
            break;
        }
        prev_checksum = chk;

        // 3d. Edge contraction: re-key edges by current labels and keep only
        // inter-component edges. This collapses the edge set rapidly.
        let contract_sql = format!(
            "
            SELECT DISTINCT sl.component_id AS src, dl.component_id AS dst
            FROM {edges_cur} e
            JOIN {state_new} sl ON e.src = sl.id
            JOIN {state_new} dl ON e.dst = dl.id
            WHERE sl.component_id <> dl.component_id
            "
        );
        edges_cur = materialize(
            ctx,
            &run_dir,
            &format!("edges_{}", round + 1),
            &contract_sql,
        )
        .await?;
        state_cur = state_new;
    }

    Ok(state_cur)
}
