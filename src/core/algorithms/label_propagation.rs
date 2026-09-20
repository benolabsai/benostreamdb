use anyhow::Result;
use arrow::array::Int64Array;
use datafusion::prelude::*;
use std::fs;
use std::path::Path;

/// Weak label propagation community detection.
///
/// The edge set is pre-materialized symmetrically so each propagation round
/// performs a SINGLE join over edges instead of two full edge scans.
/// (Unlike connected components, majority voting cannot exploit edge
/// contraction, so this is the main scalable win available here.)
///
/// Returns the name of a registered parquet table with columns `(id, label)`.
pub async fn compute_label_propagation(
    ctx: &SessionContext,
    edges_table: &str,
    temp_dir: &Path,
    max_iterations: usize,
) -> Result<String> {
    let run_dir = temp_dir.join(format!("lp_{}", uuid::Uuid::new_v4()));
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
        df.write_parquet(path.to_str().unwrap(), Default::default(), None)
            .await?;
        ctx.register_parquet(name, path.to_str().unwrap(), ParquetReadOptions::default())
            .await?;
        Ok(name.to_string())
    }

    // 1. Symmetric working edge set (deduplicated, no self-loops).
    let edges_sym = materialize(
        ctx,
        &run_dir,
        "edges_sym",
        &format!(
            "
            SELECT DISTINCT a AS source, b AS target FROM (
                SELECT source AS a, target AS b FROM {edges_table}
                UNION ALL
                SELECT target AS a, source AS b FROM {edges_table}
            )
            WHERE a <> b
            "
        ),
    )
    .await?;

    // 2. Initial state: label(v) = v.
    let init_sql = format!(
        "
        SELECT id, id AS label FROM (
            SELECT source AS id FROM {edges_sym}
            UNION
            SELECT target AS id FROM {edges_sym}
        )
        "
    );
    let mut prev_table = materialize(ctx, &run_dir, "lp_0", &init_sql).await?;

    // 3. Iterative label propagation with a state-sized convergence check.
    for i in 0..max_iterations {
        let next_name = format!("lp_{}", i + 1);

        // Most frequent neighbor label wins; ties break on smallest label.
        // The symmetric edge set means one join covers both directions.
        let update_sql = format!(
            "
            SELECT id, label FROM (
                SELECT 
                    node_id AS id, 
                    label, 
                    ROW_NUMBER() OVER (PARTITION BY node_id ORDER BY cnt DESC, label ASC) as rn
                FROM (
                    SELECT node_id, label, COUNT(*) as cnt
                    FROM (
                        -- Current state (so isolated nodes retain their label)
                        SELECT id AS node_id, label FROM {prev_table}
                        UNION ALL
                        SELECT e.target AS node_id, c.label
                        FROM {edges_sym} e JOIN {prev_table} c ON e.source = c.id
                    )
                    GROUP BY node_id, label
                )
            )
            WHERE rn = 1
            "
        );

        let next_table = materialize(ctx, &run_dir, &next_name, &update_sql).await?;

        // Convergence check (state x state join, not edge-sized).
        let check_sql = format!(
            "
            SELECT COUNT(*) AS changes
            FROM {next_table} n
            JOIN {prev_table} p ON n.id = p.id
            WHERE n.label != p.label
            "
        );
        let batches = ctx.sql(&check_sql).await?.collect().await?;
        let changes = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .map(|a| a.value(0))
            .unwrap_or(0);

        tracing::debug!("label_propagation round {i}: changes={changes}");
        if changes == 0 {
            return Ok(next_table);
        }
        prev_table = next_table;
    }

    Ok(prev_table)
}
