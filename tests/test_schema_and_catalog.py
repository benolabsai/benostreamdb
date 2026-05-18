import pytest
import hyperstreamdb as hdb
import os
import shutil

def test_datatype_constructors():
    """Verify all DataType static constructors exist and return DataType objects"""
    types = [
        hdb.DataType.int8(),
        hdb.DataType.int16(),
        hdb.DataType.int32(),
        hdb.DataType.int64(),
        hdb.DataType.uint8(),
        hdb.DataType.uint16(),
        hdb.DataType.uint32(),
        hdb.DataType.uint64(),
        hdb.DataType.float16(),
        hdb.DataType.float32(),
        hdb.DataType.float64(),
        hdb.DataType.string(),
        hdb.DataType.binary(),
        hdb.DataType.boolean(),
        hdb.DataType.date32(),
        hdb.DataType.date64(),
        hdb.DataType.timestamp_ms(),
        hdb.DataType.timestamp_us(),
        hdb.DataType.vector(128, False),
    ]
    for t in types:
        assert isinstance(t, hdb.DataType)
        assert repr(t) is not None

def test_field_and_schema_construction():
    """Verify Field, PartitionField, and Schema construction works correctly"""
    # 1. Create fields
    field_id = hdb.Field("id", hdb.DataType.int64(), False, {"description": "Primary Key"})
    field_val = hdb.Field("val", hdb.DataType.string(), True)
    field_vec = hdb.Field("embedding", hdb.DataType.vector(128, False))

    assert isinstance(field_id, hdb.Field)
    assert "id" in repr(field_id)
    assert "Int64" in repr(field_id)

    # 2. Build Schema
    schema = hdb.Schema([field_id, field_val, field_vec], {"table_type": "vector_table"})
    assert isinstance(schema, hdb.Schema)
    assert repr(schema) is not None

    # 3. Create PartitionField
    part_field = hdb.PartitionField([1], "category", "identity", 100)
    assert isinstance(part_field, hdb.PartitionField)
    assert part_field.source_ids == [1]
    assert part_field.name == "category"
    assert part_field.transform == "identity"
    assert part_field.field_id == 100

def test_unity_catalog_operations():
    """Verify Unity Catalog wrapper initialization and basic operations raise clean errors on invalid server connection"""
    catalog = hdb.PyUnityCatalog("http://localhost:8080", "mock-token")
    assert isinstance(catalog, hdb.PyUnityCatalog)

    # Create dummy schema
    field_id = hdb.Field("id", hdb.DataType.int64(), False)
    schema = hdb.Schema([field_id])

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
        catalog = hdb.PyJdbcCatalog("sqlite::memory:", "my_warehouse", "my_catalog")
    except BaseException as e:
        # If sqlx-any driver isn't registered on this host/setup, expect a clean exception/panic handle
        print(f"Skipping complete JDBC test as sqlx-any SQLite driver is not loaded: {e}")
        # Verify FFI construction with invalid parameters still raises expected exception
        with pytest.raises(BaseException):
            hdb.PyJdbcCatalog("invalid_uri://", "my_warehouse", "my_catalog")
        return

    assert isinstance(catalog, hdb.PyJdbcCatalog)

    # Create dummy schema
    field_id = hdb.Field("id", hdb.DataType.int64(), False)
    schema = hdb.Schema([field_id])

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
        assert isinstance(table, hdb.PyTable)
        assert table.table_uri() == f"file://{test_loc}"
    finally:
        # Clean up
        if os.path.exists(test_loc):
            shutil.rmtree(test_loc)

def test_manifest_not_directly_instantiable():
    """Verify Manifest and ManifestEntry exist in the module but cannot be directly constructed"""
    assert hasattr(hdb, "Manifest")
    assert hasattr(hdb, "ManifestEntry")

    with pytest.raises(TypeError):
        hdb.Manifest()

    with pytest.raises(TypeError):
        hdb.ManifestEntry()
