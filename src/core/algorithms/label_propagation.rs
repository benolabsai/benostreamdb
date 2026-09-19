use anyhow::Result;
use arrow::array::Int64Array;
use datafusion::prelude::*;
use std::fs;
use std::path::Path;

pub async fn compute_label_propagation(
    ctx: &SessionContext,
    edges_table: &str,
    temp_dir: &Path,
    max_iterations: usize,
) -> Result<String> {
    fs::create_dir_all(temp_dir)?;

    // 1. Initialize State
    let init_sql = format!(
        "
        SELECT id, id AS label FROM (
            SELECT source AS id FROM {edges_table}
            UNION
            SELECT target AS id FROM {edges_table}
        )
    "
    );

    let init_df = ctx.sql(&init_sql).await?;
    let init_path = temp_dir.join("lp_0.parquet");
    init_df
        .write_parquet(init_path.to_str().unwrap(), Default::default(), None)
        .await?;

    ctx.register_parquet(
        "lp_0",
        init_path.to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await?;

    let mut current_table = "lp_0".to_string();

    // 2. Iterative Label Propagation
    for i in 0..max_iterations {
        let next_table = format!("lp_{}", i + 1);
        let next_path = temp_dir.join(format!("{}.parquet", next_table));

        // Find the most frequent label among neighbors for each node
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
                        -- Current state (so disconnected nodes retain their label)
                        SELECT id AS node_id, label FROM {current_table}
                        UNION ALL
                        SELECT e.target AS node_id, c.label
                        FROM {edges_table} e JOIN {current_table} c ON e.source = c.id
                        UNION ALL
                        SELECT e.source AS node_id, c.label
                        FROM {edges_table} e JOIN {current_table} c ON e.target = c.id
                    )
                    GROUP BY node_id, label
                )
            )
            WHERE rn = 1
        "
        );

        let df = ctx.sql(&update_sql).await?;
        df.write_parquet(next_path.to_str().unwrap(), Default::default(), None)
            .await?;

        ctx.register_parquet(
            &next_table,
            next_path.to_str().unwrap(),
            ParquetReadOptions::default(),
        )
        .await?;

        // 3. Convergence Check
        let check_sql = format!(
            "
            SELECT COUNT(*) AS changes
            FROM {next_table} n
            JOIN {current_table} c ON n.id = c.id
            WHERE n.label != c.label
        "
        );

        let check_df = ctx.sql(&check_sql).await?;
        let batches = check_df.collect().await?;
        let changes_array = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let changes = changes_array.value(0);

        if changes == 0 {
            return Ok(next_table); // Converged
        }

        current_table = next_table;
    }

    Ok(current_table)
}
