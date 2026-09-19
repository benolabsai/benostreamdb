use anyhow::Result;
use arrow::array::Int64Array;
use datafusion::prelude::*;
use std::fs;
use std::path::Path;

pub async fn compute_topological_sort(
    ctx: &SessionContext,
    edges_table: &str,
    temp_dir: &Path,
    max_iterations: usize,
) -> Result<String> {
    fs::create_dir_all(temp_dir)?;

    // 1. Initialize State
    // We will keep track of node, and its topological 'level'
    let init_sql = format!(
        "
        SELECT id, 0 AS level FROM (
            SELECT source AS id FROM {edges_table}
            UNION
            SELECT target AS id FROM {edges_table}
        )
    "
    );

    let init_df = ctx.sql(&init_sql).await?;
    let init_path = temp_dir.join("ts_0.parquet");
    init_df
        .write_parquet(init_path.to_str().unwrap(), Default::default(), None)
        .await?;

    ctx.register_parquet(
        "ts_0",
        init_path.to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await?;

    let mut current_table = "ts_0".to_string();

    // 2. Iterative Calculation of Longest Path from Roots
    // Level(v) = max(Level(u)) + 1 for all u where u -> v
    for i in 0..max_iterations {
        let next_table = format!("ts_{}", i + 1);
        let next_path = temp_dir.join(format!("{}.parquet", next_table));

        let update_sql = format!(
            "
            SELECT 
                c.id, 
                COALESCE(MAX(parent.level) + 1, c.level) as level
            FROM {current_table} c
            LEFT JOIN {edges_table} e ON c.id = e.target
            LEFT JOIN {current_table} parent ON e.source = parent.id
            GROUP BY c.id, c.level
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
            WHERE n.level != c.level
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
