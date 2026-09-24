use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::datasource::empty::EmptyTable;
use datafusion::prelude::SessionContext;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let ctx = SessionContext::new();
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
    ]));
    ctx.register_table("t", Arc::new(EmptyTable::new(schema)))
        .unwrap();

    let sql = "SELECT * FROM t WHERE (a, b) IN ((1, 2), (3, 4))";
    match ctx.sql(sql).await {
        Ok(df) => println!("Supported! Plan:\n{:?}", df.logical_plan()),
        Err(e) => println!("Error: {}", e),
    }
}
