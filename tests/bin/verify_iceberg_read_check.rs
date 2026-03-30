// Copyright (c) 2026 Richard Albright. All rights reserved.

use hyperstreamdb::Table;
use arrow::array::StringArray;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The path where the verification shell script creates the table.
    // verify_iceberg_rest_delete.sh:
    // TABLE_NAME="test_delete_table"
    // WAREHOUSE_DIR="/tmp/hdb_test_delete"
    // So the location is /tmp/hdb_test_delete/default/test_delete_table
    let table_uri = "file:///tmp/hdb_test_delete/default/test_delete_table";
    
    println!("Opening table at {}", table_uri);
    let table = Table::new_async(table_uri.to_string()).await?;
    
    println!("Scanning table...");
    let batches = table.read_async(None, None, None).await?;
    
    let mut total_rows = 0;
    let mut row_0_found = false;
    let mut row_5_found = false;
    let mut row_10_found = false;
    let mut found_categories = Vec::new();

    for batch in batches {
        total_rows += batch.num_rows();
        let col = batch.column_by_name("category").ok_or("category column missing")?;
        let str_col = col.as_any().downcast_ref::<StringArray>().ok_or("category not string")?;
        
        for i in 0..batch.num_rows() {
             let val = str_col.value(i);
             found_categories.push(val.to_string());
             if val == "row_0" { row_0_found = true; }
             if val == "row_5" { row_5_found = true; }
             if val == "row_10" { row_10_found = true; }
        }
    }
    
    println!("Total rows read: {}", total_rows);
    
    // 100 rows total. 
    // row_0 (pos delete 0), row_5 (pos delete 5), row_10 (equality delete)
    // Total should be 97.
    if total_rows != 97 {
        eprintln!("Expected 97 rows, got {}", total_rows);
        // Debug: print what was found short list
        if total_rows < 110 {
           println!("Found: {:?}", found_categories);
        }
        std::process::exit(1);
    }
    
    if row_0_found {
        eprintln!("FAILURE: row_0 was found but should be deleted!");
        std::process::exit(1);
    }
    if row_5_found {
        eprintln!("FAILURE: row_5 was found but should be deleted!");
        std::process::exit(1);
    }
    if row_10_found {
        eprintln!("FAILURE: row_10 was found but should be deleted!");
        std::process::exit(1);
    }
    
    println!("SUCCESS: Verification Passed! Deleted rows are gone.");
    Ok(())
}
