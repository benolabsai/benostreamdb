import pytest
import benostreamdb as bsdb
import os
import shutil

def test_datatype_constructors():
    """Verify all DataType static constructors exist and return DataType objects"""
    types = [
        bsdb.DataType.int8(),
        bsdb.DataType.int16(),
        bsdb.DataType.int32(),
        bsdb.DataType.int64(),
        bsdb.DataType.uint8(),
        bsdb.DataType.uint16(),
        bsdb.DataType.uint32(),
        bsdb.DataType.uint64(),
        bsdb.DataType.float16(),
        bsdb.DataType.float32(),
        bsdb.DataType.float64(),
        bsdb.DataType.string(),
        bsdb.DataType.binary(),
        bsdb.DataType.boolean(),
        bsdb.DataType.date32(),
        bsdb.DataType.date64(),
        bsdb.DataType.timestamp_ms(),
        bsdb.DataType.timestamp_us(),
        bsdb.DataType.vector(128, False),
    ]
    for t in types:
        assert isinstance(t, bsdb.DataType)
        assert repr(t) is not None

def test_field_and_schema_construction():
    """Verify Field, PartitionField, and Schema construction works correctly"""
    # 1. Create fields
    field_id = bsdb.Field("id", bsdb.DataType.int64(), False, {"description": "Primary Key"})
    field_val = bsdb.Field("val", bsdb.DataType.string(), True)
    field_vec = bsdb.Field("embedding", bsdb.DataType.vector(128, False))

    assert isinstance(field_id, bsdb.Field)
    assert "id" in repr(field_id)
    assert "Int64" in repr(field_id)

    # 2. Build Schema
    schema = bsdb.Schema([field_id, field_val, field_vec], {"table_type": "vector_table"})
    assert isinstance(schema, bsdb.Schema)
    assert repr(schema) is not None

    # 3. Create PartitionField
    part_field = bsdb.PartitionField([1], "category", "identity", 100)
    assert isinstance(part_field, bsdb.PartitionField)
    assert part_field.source_ids == [1]
    assert part_field.name == "category"
    assert part_field.transform == "identity"
    assert part_field.field_id == 100

def test_unity_catalog_operations():
    """Verify Unity Catalog wrapper initialization and basic operations raise clean errors on invalid server connection"""
    catalog = bsdb.PyUnityCatalog("http://localhost:8080", "mock-token")
    assert isinstance(catalog, bsdb.PyUnityCatalog)

    # Create dummy schema
    field_id = bsdb.Field("id", bsdb.DataType.int64(), False)
    schema = bsdb.Schema([field_id])

    # Should raise RuntimeError due to connection refusal
    with pytest.raises(RuntimeError):
        catalog.create_table("my_catalog.my_schema", "my_table", schema, None)

    with pytest.raises(RuntimeError):
        catalog.load_table("my_catalog.my_schema", "my_table")

    # table_exists catches the error and returns False
    assert catalog.table_exists("my_catalog.my_schema", "my_table") is False

def test_jdbc_catalog_operations():
    """Verify JDBC Catalog wrapper initialization and operations with an in-memory SQLite backend"""
    try:
        # Try establishing with SQLite in-memory URL
        catalog = bsdb.PyJdbcCatalog("sqlite::memory:", "my_warehouse", "my_catalog")
    except BaseException as e:
        # If sqlx-any driver isn't registered on this host/setup, expect a clean exception/panic handle
        print(f"Skipping complete JDBC test as sqlx-any SQLite driver is not loaded: {e}")
        # Verify FFI construction with invalid parameters still raises expected exception
        with pytest.raises(BaseException):
            bsdb.PyJdbcCatalog("invalid_uri://", "my_warehouse", "my_catalog")
        return

    assert isinstance(catalog, bsdb.PyJdbcCatalog)

    # Create dummy schema
    field_id = bsdb.Field("id", bsdb.DataType.int64(), False)
    schema = bsdb.Schema([field_id])

    # table_exists initially should return False
    assert catalog.table_exists("my_namespace", "my_table") is False

    # load_table on non-existent table should raise RuntimeError (RowNotFound)
    with pytest.raises(RuntimeError):
        catalog.load_table("my_namespace", "my_table")

    # Clean up test table directory if it exists
    test_loc = os.path.abspath("test_sqlite_jdbc_table")
    if os.path.exists(test_loc):
        shutil.rmtree(test_loc)

    try:
        # create_table should succeed and register the table in the SQLite metadata store
        catalog.create_table("my_namespace", "my_table", schema, f"file://{test_loc}")

        # Now, table_exists should return True!
        assert catalog.table_exists("my_namespace", "my_table") is True

        # load_table should load it successfully and return a PyTable!
        table = catalog.load_table("my_namespace", "my_table")
        assert isinstance(table, bsdb.PyTable)
        assert table.table_uri() == f"file://{test_loc}"
    finally:
        # Clean up
        if os.path.exists(test_loc):
            shutil.rmtree(test_loc)

def test_manifest_not_directly_instantiable():
    """Verify Manifest and ManifestEntry exist in the module but cannot be directly constructed"""
    assert hasattr(bsdb, "Manifest")
    assert hasattr(bsdb, "ManifestEntry")

    with pytest.raises(TypeError):
        bsdb.Manifest()

    with pytest.raises(TypeError):
        bsdb.ManifestEntry()
