import hyperstreamdb as hdb
import pandas as pd
import pytest

def test_boolean_indexing(tmp_path):
    """Test that boolean columns can be indexed and filtered correctly."""
    table_path = str(tmp_path / "bool_test")
    table = hdb.Table(f"file://{table_path}")
    
    # Create DataFrame with boolean column
    df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'is_active': [True, False, True, False, True],
        'category': ['A', 'B', 'A', 'B', 'A']
    })
    
    # Write data (ensure it's committed for boolean search to find it)
    table.write_pandas(df)
    table.commit()
    
    # Test filtering for true values 
    result_true = table.sql("SELECT * FROM t WHERE is_active = true")
    df_true = result_true.to_pandas()
    
    print(f"True results: {len(df_true)} rows")
    print(df_true)
    
    # Should get rows 1, 3, 5 (all with is_active=True)
    assert len(df_true) == 3
    assert all(df_true['is_active'])
    assert sorted(df_true['id'].tolist()) == [1, 3, 5]
    
    # Test filtering for false values  
    result_false = table.sql("SELECT * FROM t WHERE is_active = false")
    df_false = result_false.to_pandas()
    
    print(f"False results: {len(df_false)} rows")
    print(df_false)
    
    # Should get rows 2, 4 (all with is_active=False)
    assert len(df_false) == 2
    assert not any(df_false['is_active'])
    assert sorted(df_false['id'].tolist()) == [2, 4]
    
    print("Boolean indexing test passed!")
