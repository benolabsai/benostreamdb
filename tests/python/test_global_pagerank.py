import os
import shutil
import pytest
import pyarrow as pa
from hyperstreamdb import Table

@pytest.fixture
def table_fixture(tmpdir):
    cat_path = str(tmpdir.join("hyperstreamdb_test_pagerank"))
    schema = pa.schema([
        ("src", pa.uint64()),
        ("dst", pa.uint64())
    ])
    t = Table.create(cat_path, schema)
    yield t

def test_global_pagerank(table_fixture):
    table = table_fixture
    
    # Simple graph:
    # 1 -> 2
    # 1 -> 3
    # 2 -> 3
    # 3 -> 1
    # 4 -> 4 (disconnected)
    
    data = pa.table({
        "src": [1, 1, 2, 3, 4],
        "dst": [2, 3, 3, 1, 4]
    })
    
    table.insert(data)
    
    # 2. Run PageRank out-of-core
    result_table = table.graph.pagerank(source_col="src", target_col="dst", damping=0.85, iterations=5)
    
    # 3. Verify results
    result_df = result_table.to_pandas()
    assert len(result_df) == 4, "Should have 4 unique nodes"
    assert set(result_df["node"].tolist()) == {1, 2, 3, 4}
    
    print(result_df)
    
    # Basic sanity checks: node 3 should have highest pagerank because 1 and 2 point to it.
    pr_3 = result_df[result_df["node"] == 3]["pr"].values[0]
    pr_2 = result_df[result_df["node"] == 2]["pr"].values[0]
    
    assert pr_3 > pr_2, "Node 3 should have a higher PageRank than Node 2"
