
import hyperstreamdb as hdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import os
import shutil

@pytest.fixture
def cleanup():
    if os.path.exists("test_splits_data"):
        shutil.rmtree("test_splits_data")
    yield
    if os.path.exists("test_splits_data"):
        shutil.rmtree("test_splits_data")

def test_read_split_with_projection(cleanup):
    os.makedirs("test_splits_data", exist_ok=True)
    table_uri = f"file://{os.getcwd()}/test_splits_data"
    
    # 1. Create a Parquet file with 3 Row Groups (approx)
    # 3 batches of 10 rows
    schema = pa.schema([
        ('id', pa.int32()),
        ('val', pa.string()),
        ('heavy', pa.string()) # Large column we want to skip
    ])
    
    file_path = f"{table_uri}/segment_abc.parquet"
    writer = pq.ParquetWriter(file_path.replace("file://", ""), schema, version='2.6')
    
    for i in range(3):
        ids = pa.array(range(i*10, (i+1)*10), type=pa.int32())
        vals = pa.array([f"val_{x}" for x in range(i*10, (i+1)*10)])
        heavy = pa.array(["X" * 1000 for _ in range(10)])
        batch = pa.Table.from_arrays([ids, vals, heavy], schema=schema)
        writer.write_table(batch, row_group_size=10)
        
    writer.close()
    
    # 2. Initialize Table
    # Note: Table::new expects existing table or creates it? 
    # Our simple Table::new just opens logic.
    # We need to manually construct splits or use list_data_files -> manual split construction if get_splits logic is simple.
    
    # Actually, let's use list_data_files to get the file, then create a split manually or use get_splits
    # We implemented `get_splits(max_split_size)`.
    # File size: 30 rows + heavy strings ~ 30KB.
    # Let's say max_split_size = 10KB.
    
    # Need to trick `get_splits`? 
    # Or just verify `read_split` directly by assuming we know row group IDs.
    
    # Let's interact via `read_split` manually first to test the API.
    
    catalog = hdb.create_catalog("nessie", {"url": "http://localhost:19120"}) # Type doesn't matter for local file read if we use direct path
    # Actually we don't have a specific `Table` python object exposed easily for direct `read_split` calls
    # except via the internal test wrapper or if we expose it on the Catalog?
    # Python binding `read_split` is on `PyTable`.
    
    # We need a `PyTable`.
    # `load_table` usually returns one. But we have a local file.
    # Let's use `hdb.Table` (if exposed?) No.
    # But `PyRestCatalog` etc return `PyTable`.
    # Does `PyTable.new(uri)` exist? It is not exposed to Python as `__init__`.
    
    # 3. Open Table directly
    # Note: open_table expects URI. For local file, typically table URI is parent dir.
    # But here we want to treat the file as a table/segment source.
    # Our simple Table implementation treats URI as "Table Root".
    # And read_split(split) uses split.file_path.
    
    # Let's open the "Table" at the directory level.
    table = hdb.open_table(table_uri)
    
    # 4. Construct a Split manually (or via Python wrapper if exposed)
    # PySplit is exposed.
    # We want to read Row Group 1 (rows 10-20).
    # file_path in split should be relative or absolute. Table assumes absolute or relative to store.
    # Let's use absolute path since we used absolute path in writer.
    
    # Row Group 1: 
    split = hdb.PySplit(
        file_path,
        0, 100, # offset/length ignored by current impl
        [1],    # row_group_ids
        None,   # index_file_path
        False   # can_use_indexes
    )
    
    # 5. Read Split with Projection (SKIP 'heavy')
    columns = ["id", "val"]
    arrow_table = table.read_split(split, columns)
    
    # 6. Verify
    assert arrow_table.num_columns == 2
    assert "id" in arrow_table.column_names
    assert "val" in arrow_table.column_names
    assert "heavy" not in arrow_table.column_names
    
    # Verify Rows (should be 10..20)
    ids = arrow_table.column("id").to_pylist()
    assert len(ids) == 10
    assert ids == list(range(10, 20))
    
    print("Test Passed: Read Split specific Row Group with Projection!")
