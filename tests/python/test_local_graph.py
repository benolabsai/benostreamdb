import pyarrow as pa
import pytest
from hyperstreamdb import Table
import os

@pytest.fixture
def graph_table(tmpdir):
    schema = pa.schema([
        ("source", pa.uint64()),
        ("target", pa.uint64()),
        ("weight", pa.float64())
    ])
    table_uri = str(tmpdir.join("test_local_graph"))
    t = Table.create(table_uri, schema)
    
    # 1 -> 2
    # 2 -> 3
    # 3 -> 1
    # 1 -> 4
    # 4 -> 5
    edges = pa.Table.from_arrays([
        pa.array([1, 2, 3, 1, 4], type=pa.uint64()),
        pa.array([2, 3, 1, 4, 5], type=pa.uint64()),
        pa.array([1.0, 2.0, 1.5, 0.5, 1.0], type=pa.float64())
    ], names=["source", "target", "weight"])
    
    t.add_index("target", {"type": "graph", "src_column": "source", "dst_column": "target"})
    t.insert(edges)
    t.commit()
    t.wait_for_background_tasks()
    return t

def test_shortest_path(graph_table):
    # Shortest path from 1 to 5
    path = graph_table.shortest_path(1, 5, graph_column="source")
    assert path == [1, 4, 5]
    
    # Shortest path from 3 to 2
    path = graph_table.shortest_path(3, 2, graph_column="source")
    assert path == [3, 1, 2]

def test_neighbors(graph_table):
    neighbors = graph_table.graph_neighbors(1, graph_column="source")
    # 1 has edges to 2 and 4
    assert set(neighbors) == {2, 4}

def test_subgraph(graph_table):
    edges = graph_table.subgraph([1, 2, 3], graph_column="source")
    # Expect (1, 2), (2, 3), (3, 1)
    assert set(edges) == {(1, 2), (2, 3), (3, 1)}

def test_connecting_paths(graph_table):
    edges = graph_table.connecting_paths([1, 5], graph_column="source")
    # path between 1 and 5 is 1->4->5, so edges should be (1, 4) and (4, 5)
    assert set(edges) == {(1, 4), (4, 5)}
