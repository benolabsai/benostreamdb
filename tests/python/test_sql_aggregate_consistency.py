import os
import pytest
import numpy as np
import pandas as pd
import pyarrow as pa
import hyperstreamdb as hdb

@pytest.fixture
def aggregate_data():
    """Generates a dataset with edge cases for testing aggregates."""
    np.random.seed(42)
    
    n_samples = 100
    groups = ["A", "B", "C"]
    
    # 1. Base data with normal values, negative values, and exact zero
    val1 = np.random.randn(n_samples)
    val1[10] = 0.0
    
    # 2. Data with NaNs
    val2 = np.random.randn(n_samples)
    val2[20:30] = np.nan
    
    # 3. Integer data
    int_val = np.random.randint(-100, 100, size=n_samples)
    
    # 4. Groupings
    group_col = np.random.choice(groups, size=n_samples)
    
    # 5. Embeddings for vector_avg
    embeddings = np.random.randn(n_samples, 4).astype(np.float32)
    
    df = pd.DataFrame({
        "id": range(n_samples),
        "group_col": group_col,
        "val1": val1,
        "val2": val2,
        "int_val": int_val,
        "embedding": embeddings.tolist()
    })
    
    return df

def test_standard_aggregates(aggregate_data, tmpdir):
    df = aggregate_data
    uri = str(tmpdir.join("test_aggs"))
    
    schema = pa.schema([
        ("id", pa.int64()),
        ("group_col", pa.string()),
        ("val1", pa.float64()),
        ("val2", pa.float64()),
        ("int_val", pa.int64()),
        ("embedding", pa.list_(pa.float32(), 4))
    ])
    
    table = hdb.Table.create(uri, schema)
    table.write(df)
    table.commit()
    
    # Execute SQL
    sql = """
    SELECT 
        COUNT(id) as count_id,
        SUM(val1) as sum_val1,
        AVG(val1) as avg_val1,
        MIN(val1) as min_val1,
        MAX(val1) as max_val1,
        SUM(val2) as sum_val2,
        AVG(val2) as avg_val2,
        MIN(val2) as min_val2,
        MAX(val2) as max_val2,
        SUM(int_val) as sum_int_val,
        AVG(int_val) as avg_int_val
    FROM t
    """
    res_df = table.execute_sql(sql).to_pandas()
    
    # pandas aggregates (using skipna=True which is SQL's default behavior)
    pd_sum_val1 = df["val1"].sum()
    pd_avg_val1 = df["val1"].mean()
    pd_min_val1 = df["val1"].min()
    pd_max_val1 = df["val1"].max()
    
    pd_sum_val2 = df["val2"].sum()
    pd_avg_val2 = df["val2"].mean()
    pd_min_val2 = df["val2"].min()
    pd_max_val2 = df["val2"].max()
    
    pd_sum_int = df["int_val"].sum()
    pd_avg_int = df["int_val"].mean()
    
    # Assertions
    assert res_df["count_id"][0] == 100
    assert np.isclose(res_df["sum_val1"][0], pd_sum_val1)
    assert np.isclose(res_df["avg_val1"][0], pd_avg_val1)
    assert np.isclose(res_df["min_val1"][0], pd_min_val1)
    assert np.isclose(res_df["max_val1"][0], pd_max_val1)
    
    assert np.isclose(res_df["sum_val2"][0], pd_sum_val2)
    assert np.isclose(res_df["avg_val2"][0], pd_avg_val2)
    assert np.isclose(res_df["min_val2"][0], pd_min_val2)
    assert np.isclose(res_df["max_val2"][0], pd_max_val2)
    
    assert res_df["sum_int_val"][0] == pd_sum_int
    assert np.isclose(res_df["avg_int_val"][0], pd_avg_int)

def test_group_by_aggregates(aggregate_data, tmpdir):
    df = aggregate_data
    uri = str(tmpdir.join("test_group_aggs"))
    
    schema = pa.schema([
        ("id", pa.int64()),
        ("group_col", pa.string()),
        ("val1", pa.float64()),
        ("val2", pa.float64()),
        ("int_val", pa.int64()),
        ("embedding", pa.list_(pa.float32(), 4))
    ])
    
    table = hdb.Table.create(uri, schema)
    table.write(df)
    table.commit()
    
    sql = """
    SELECT 
        group_col,
        COUNT(id) as count_id,
        SUM(val1) as sum_val1,
        AVG(val2) as avg_val2,
        MAX(int_val) as max_int_val
    FROM t
    GROUP BY group_col
    ORDER BY group_col
    """
    res_df = table.execute_sql(sql).to_pandas()
    
    pd_grouped = df.groupby("group_col").agg(
        count_id=("id", "count"),
        sum_val1=("val1", "sum"),
        avg_val2=("val2", "mean"),
        max_int_val=("int_val", "max")
    ).reset_index().sort_values("group_col").reset_index(drop=True)
    
    assert list(res_df["group_col"]) == list(pd_grouped["group_col"])
    assert list(res_df["count_id"]) == list(pd_grouped["count_id"])
    
    for i in range(len(res_df)):
        assert np.isclose(res_df["sum_val1"][i], pd_grouped["sum_val1"][i])
        assert np.isclose(res_df["avg_val2"][i], pd_grouped["avg_val2"][i])
        assert res_df["max_int_val"][i] == pd_grouped["max_int_val"][i]

def test_vector_avg(aggregate_data, tmpdir):
    df = aggregate_data
    uri = str(tmpdir.join("test_vector_avg"))
    
    schema = pa.schema([
        ("id", pa.int64()),
        ("group_col", pa.string()),
        ("val1", pa.float64()),
        ("val2", pa.float64()),
        ("int_val", pa.int64()),
        ("embedding", pa.list_(pa.float32(), 4))
    ])
    
    table = hdb.Table.create(uri, schema)
    table.write(df)
    table.commit()
    
    # In HyperStreamDB, we need to register the UDFs to the session.
    # Usually execute_sql does this automatically if Table method is used, but for specific vector_avg
    # it might need session. Wait, the existing test_sql_agg_order.py has logic for this.
    
    sql = """
    SELECT 
        group_col,
        vector_avg(embedding) as avg_emb
    FROM t
    GROUP BY group_col
    ORDER BY group_col
    """
    # Assuming Table.execute_sql handles UDFs
    res_df = table.execute_sql(sql).to_pandas()
    
    pd_grouped = df.groupby("group_col")
    
    groups = res_df["group_col"].tolist()
    avg_embs = res_df["avg_emb"].tolist()
    
    for group, avg_emb in zip(groups, avg_embs):
        group_df = df[df["group_col"] == group]
        expected_avg = np.mean(np.vstack(group_df["embedding"].values), axis=0)
        
        assert np.allclose(avg_emb, expected_avg, rtol=1e-5, atol=1e-5), f"Mismatch for group {group}: {avg_emb} vs {expected_avg}"
