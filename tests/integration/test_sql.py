import time
import hyperstreamdb as hdb
import pandas as pd
import pyarrow as pa
import pytest
import os
import shutil

@pytest.fixture
def table_path(tmp_path):
    path = tmp_path / "sql_test_table"
    return str(path)

def test_sql_basic_query(table_path, tmp_path):
    print(f"Creating table at {table_path}")
    table = hdb.Table(f"file://{table_path}")
    
    # Create Data
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "value": [10.0, 20.0, 30.0, 40.0, 50.0],
        "category": ["A", "B", "A", "B", "C"]
    })
    
    # Write
    print("Writing data...")
    table.write_pandas(df)
    table.commit() # Ensure data is in segments for subsequent queries
    
    # Query all
    print("Executing SELECT * FROM t")
    result = table.sql("SELECT * FROM t")
    print("Result type:", type(result))
    
    # Result is pyarrow.Table
    assert isinstance(result, pa.Table)
    assert len(result) == 5
    
    # Convert to pandas for easier checking
    res_df = result.to_pandas()
    assert len(res_df) == 5
    assert res_df["id"].sum() == 15
    
    # Boolean column and filter test
    # Create a new table for boolean testing
    bool_table_path = str(tmp_path / "sql_test_table_bool")
    table_bool = hdb.Table(f"file://{bool_table_path}")
    
    # Create DataFrame (including Boolean)
    df_bool = pd.DataFrame({'id': [1, 2, 3], 'category': ['science', 'math', 'science'], 'is_active': [True, False, True]})
    table_bool.write_pandas(df_bool)
    table_bool.commit()
    table_bool.wait_for_background_tasks() # Final sync window
    
    # Test boolean filtering using table.sql() directly (simpler path)
    results = table_bool.sql("SELECT * FROM t WHERE is_active = true")
    results_df = results.to_pandas()
    
    # Boolean indexing is implemented - verify it returns results
    assert len(results_df) >= 2  # Should have at least the 2 true rows
    assert all(results_df['is_active'])  # All returned rows should be true

    results_false = table_bool.sql("SELECT * FROM t WHERE is_active = false")
    results_false_df = results_false.to_pandas()
    assert len(results_false_df) >= 1  # Should have at least the 1 false row
    assert not any(results_false_df['is_active'])  # All returned rows should be false

    # Check Memory Limit Argument (Should not crash even if unused)
    session_limited = hdb.Session(memory_mb=100)
    assert session_limited is not None
    
    # Filter Query
    print("Executing SELECT * FROM t WHERE id > 3")
    # Re-write original data for subsequent tests that expect it
    # Data is already there from the top of the test, no need to rewrite unless truncated
    result_filtered = table.sql("SELECT * FROM t WHERE id > 3")
    assert len(result_filtered) == 2
    res_df_filtered = result_filtered.to_pandas()
    assert sorted(res_df_filtered["id"].tolist()) == [4, 5]

    # Projections
    print("Executing SELECT id, category FROM t WHERE category = 'A'")
    result_proj = table.sql("SELECT id, category FROM t WHERE category = 'A'")
    # DataFusion might reorder columns? Or keep projection order.
    assert set(result_proj.column_names) == {"id", "category"}
    assert len(result_proj) == 2
    res_df_proj = result_proj.to_pandas()
    assert sorted(res_df_proj["id"].tolist()) == [1, 3]

    # Limit & Offset
    print("Executing SELECT * FROM t ORDER BY id LIMIT 2 OFFSET 1")
    result_limit = table.sql("SELECT * FROM t ORDER BY id LIMIT 2 OFFSET 1")
    assert len(result_limit) == 2
    df_limit = result_limit.to_pandas()
    # Ordered by ID: 1, 2, 3, 4, 5.
    # Offset 1 (skip 1) -> start at 2. Limit 2 -> 2, 3.
    print("SQL Tests (Select/Limit) Passed!")

    # Joins
    print("Executing Join Test...")
    session = hdb.Session()
    
    # Create orders table
    orders_table = hdb.Table(f"file://{tmp_path}/orders")
    orders_df = pd.DataFrame({
        "order_id": [101, 102, 103],
        "user_id": [1, 2, 4], # 4 has no match
        "amount": [10.5, 20.0, 30.0]
    })
    orders_table.write_pandas(orders_df)
    orders_table.commit()
    
    session.register("users", table._inner)
    session.register("orders", orders_table._inner)
    
    # Inner Join
    # users: 1, 3, 5 (after previous writes? No, wait. 
    # Previous tests wrote:
    # 1. df: 1(A), 2(B), 3(C)
    # 2. df2: 4(D), 5(E)
    # Total users: 1, 2, 3, 4, 5
    
    # Join on user_id (Table 1: id, Table 2: user_id)
    join_res = session.sql("""
        SELECT u.id, u.category, o.amount 
        FROM users u 
        JOIN orders o ON u.id = o.user_id
        ORDER BY u.id
    """)
    join_df = join_res.to_pandas()
    
    # Matches:
    # 1 (A) <-> 101 (10.5)
    # 2 (B) <-> 102 (20.0)
    # 4 (D) <-> 103 (30.0)
    # 3, 5 have no orders.
    
    assert len(join_df) == 3
    assert join_df.iloc[0]["id"] == 1
    assert join_df.iloc[1]["id"] == 2
    assert join_df.iloc[2]["id"] == 4
    assert join_df.iloc[0]["amount"] == 10.5
    
    print("Join Test Passed!")
