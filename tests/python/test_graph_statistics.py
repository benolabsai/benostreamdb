import pytest
import pyarrow as pa
from hyperstreamdb import Table

@pytest.fixture
def graph_table(tmpdir):
    uri = str(tmpdir.join("graph_edges"))
    schema = pa.schema([
        ("source", pa.uint64()),
        ("target", pa.uint64()),
    ])
    table = Table.create(uri, schema)
    
    data = pa.Table.from_arrays([
        pa.array([1, 2, 3, 4], type=pa.uint64()),
        pa.array([2, 3, 1, 5], type=pa.uint64()),
    ], names=["source", "target"])
    
    table.insert(data)
    table.commit()
    return table

def test_pagerank(graph_table):
    df = graph_table.pagerank(iterations=5).to_pandas()
    assert "node" in df.columns
    assert "score" in df.columns
    assert len(df) == 5

def test_connected_components(graph_table):
    df = graph_table.connected_components().to_pandas()
    assert "component" in df.columns
    assert len(df) == 3

def test_strongly_connected_components(graph_table):
    df = graph_table.strongly_connected_components().to_pandas()
    assert "node_id" in df.columns
    assert "scc_id" in df.columns
    # 1,2,3 in one SCC, 4, 5 in separate SCCs
    assert len(df["scc_id"].unique()) == 3

def test_shortest_path(graph_table):
    df = graph_table.shortest_path(1, 3).to_pandas()
    assert "node" in df.columns
    nodes = df["node"].tolist()
    assert nodes == [1, 2, 3]

def test_graph_neighbors(graph_table):
    df = graph_table.graph_neighbors(1, hops=2).to_pandas()
    assert "neighbor" in df.columns
    neighbors = df["neighbor"].tolist()
    assert set(neighbors) == {2, 3}

def test_label_propagation(graph_table):
    df = graph_table.label_propagation_communities().to_pandas()
    assert "community" in df.columns
    assert len(df) == 2
