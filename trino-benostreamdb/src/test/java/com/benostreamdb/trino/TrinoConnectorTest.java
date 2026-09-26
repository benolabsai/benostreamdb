package com.benostreamdb.trino;

import io.trino.spi.connector.ColumnHandle;
import io.trino.spi.connector.ConnectorSplit;
import io.trino.spi.connector.ConnectorTableHandle;
import io.trino.spi.connector.SchemaTableName;
import io.trino.spi.type.IntegerType;
import io.trino.spi.type.VarcharType;
import org.junit.Test;

import java.util.List;
import java.util.Map;

import static org.junit.Assert.*;

/**
 * Unit tests for the Trino BenoStream connector components.
 * Tests metadata handling, split management, and column handles without
 * requiring native libraries.
 */
public class TrinoConnectorTest {

    // ---- Plugin Tests ----

    @Test
    public void testPluginReturnsConnectorFactory() {
        BenoStreamDBPlugin plugin = new BenoStreamDBPlugin();
        var factories = plugin.getConnectorFactories();

        assertNotNull("Connector factories should not be null", factories);
        int count = 0;
        for (io.trino.spi.connector.ConnectorFactory factory : factories) {
            count++;
            assertTrue("Factory should be BenoStreamDBConnectorFactory",
                    factory instanceof BenoStreamDBConnectorFactory);
        }
        assertEquals("Should have exactly one factory", 1, count);
    }

    // ---- ConnectorFactory Tests ----

    @Test
    public void testConnectorFactoryName() {
        BenoStreamDBConnectorFactory factory = new BenoStreamDBConnectorFactory();
        assertEquals("Connector name should be benostreamdb", "benostreamdb", factory.getName());
    }

    @Test
    public void testConnectorFactoryCreate() {
        BenoStreamDBConnectorFactory factory = new BenoStreamDBConnectorFactory();
        var connector = factory.create("test_catalog", Map.of(), null);

        assertNotNull("Connector should not be null", connector);
    }

    @Test
    public void testConnectorGetMetadata() {
        BenoStreamDBConnectorFactory factory = new BenoStreamDBConnectorFactory();
        var connector = factory.create("test", Map.of(), null);

        var metadata = connector.getMetadata(null, null);
        assertNotNull("Metadata should not be null", metadata);
        assertTrue("Should be BenoStreamDBMetadata", metadata instanceof BenoStreamDBMetadata);
    }

    @Test
    public void testConnectorGetSplitManager() {
        BenoStreamDBConnectorFactory factory = new BenoStreamDBConnectorFactory();
        var connector = factory.create("test", Map.of(), null);

        var splitManager = connector.getSplitManager();
        assertNotNull("Split manager should not be null", splitManager);
        assertTrue("Should be BenoStreamDBSplitManager", splitManager instanceof BenoStreamDBSplitManager);
    }

    @Test
    public void testConnectorGetPageSourceProvider() {
        BenoStreamDBConnectorFactory factory = new BenoStreamDBConnectorFactory();
        var connector = factory.create("test", Map.of(), null);

        var provider = connector.getPageSourceProvider();
        assertNotNull("Page source provider should not be null", provider);
        assertTrue("Should be BenoStreamDBPageSourceProvider", provider instanceof BenoStreamDBPageSourceProvider);
    }

    @Test
    public void testConnectorBeginTransaction() {
        BenoStreamDBConnectorFactory factory = new BenoStreamDBConnectorFactory();
        var connector = factory.create("test", Map.of(), null);

        var tx = connector.beginTransaction(
                io.trino.spi.transaction.IsolationLevel.SERIALIZABLE, true, true);
        assertNotNull("Transaction handle should not be null", tx);
        assertTrue("Should be BenoStreamDBTransactionHandle",
                tx instanceof BenoStreamDBConnectorFactory.BenoStreamDBTransactionHandle);
    }

    // ---- Metadata Tests ----

    @Test
    public void testMetadataListSchemaNames() {
        BenoStreamDBMetadata metadata = new BenoStreamDBMetadata();
        var schemas = metadata.listSchemaNames(null);

        assertNotNull("Schema list should not be null", schemas);
        assertEquals("Should have default schema", 1, schemas.size());
        assertEquals("Schema should be default", "default", schemas.get(0));
    }

    @Test
    public void testMetadataGetTableHandle() {
        BenoStreamDBMetadata metadata = new BenoStreamDBMetadata();
        SchemaTableName tableName = new SchemaTableName("default", "my_table");

        ConnectorTableHandle handle = metadata.getTableHandle(null, tableName);
        assertNotNull("Table handle should not be null", handle);
        assertTrue("Should be BenoStreamDBTableHandle", handle instanceof BenoStreamDBTableHandle);

        BenoStreamDBTableHandle tableHandle = (BenoStreamDBTableHandle) handle;
        assertEquals("Schema name should match", "default", tableHandle.getSchemaName());
        assertEquals("Table name should match", "my_table", tableHandle.getTableName());
    }

    @Test
    public void testMetadataGetTableMetadata() {
        BenoStreamDBMetadata metadata = new BenoStreamDBMetadata();
        BenoStreamDBTableHandle tableHandle = new BenoStreamDBTableHandle("default", "test_table");

        var tableMetadata = metadata.getTableMetadata(null, tableHandle);
        assertNotNull("Table metadata should not be null", tableMetadata);
        assertEquals("Table name should match", "test_table", tableMetadata.getTable().getTableName());

        var columns = tableMetadata.getColumns();
        assertEquals("Should have 2 columns", 2, columns.size());
        assertEquals("First column should be id", "id", columns.get(0).getName());
        assertEquals("Second column should be name", "name", columns.get(1).getName());
    }

    @Test
    public void testMetadataListTables() {
        BenoStreamDBMetadata metadata = new BenoStreamDBMetadata();
        var tables = metadata.listTables(null, java.util.Optional.of("default"));

        assertNotNull("Table list should not be null", tables);
        assertEquals("Should have one test table", 1, tables.size());
        assertEquals("Table should be test_table", "test_table", tables.get(0).getTableName());
    }

    @Test
    public void testMetadataGetColumnHandles() {
        BenoStreamDBMetadata metadata = new BenoStreamDBMetadata();
        BenoStreamDBTableHandle tableHandle = new BenoStreamDBTableHandle("default", "test_table");

        Map<String, ColumnHandle> columnHandles = metadata.getColumnHandles(null, tableHandle);
        assertNotNull("Column handles should not be null", columnHandles);
        assertEquals("Should have 2 columns", 2, columnHandles.size());
        assertTrue("Should have id column", columnHandles.containsKey("id"));
        assertTrue("Should have name column", columnHandles.containsKey("name"));

        BenoStreamDBColumnHandle idHandle = (BenoStreamDBColumnHandle) columnHandles.get("id");
        assertEquals("id column type should be INTEGER", IntegerType.INTEGER, idHandle.getColumnType());

        BenoStreamDBColumnHandle nameHandle = (BenoStreamDBColumnHandle) columnHandles.get("name");
        assertEquals("name column type should be VARCHAR", VarcharType.VARCHAR, nameHandle.getColumnType());
    }

    @Test
    public void testMetadataGetColumnMetadata() {
        BenoStreamDBMetadata metadata = new BenoStreamDBMetadata();
        BenoStreamDBTableHandle tableHandle = new BenoStreamDBTableHandle("default", "test_table");
        BenoStreamDBColumnHandle columnHandle = new BenoStreamDBColumnHandle("id", IntegerType.INTEGER);

        var columnMetadata = metadata.getColumnMetadata(null, tableHandle, columnHandle);
        assertNotNull("Column metadata should not be null", columnMetadata);
        assertEquals("Column name should match", "id", columnMetadata.getName());
        assertEquals("Column type should match", IntegerType.INTEGER, columnMetadata.getType());
    }

    // ---- TableHandle Tests ----

    @Test
    public void testTableHandleGetters() {
        BenoStreamDBTableHandle handle = new BenoStreamDBTableHandle("public", "users");

        assertEquals("Schema name should match", "public", handle.getSchemaName());
        assertEquals("Table name should match", "users", handle.getTableName());
    }

    // ---- ColumnHandle Tests ----

    @Test
    public void testColumnHandleGetters() {
        BenoStreamDBColumnHandle handle = new BenoStreamDBColumnHandle("email", VarcharType.VARCHAR);

        assertEquals("Column name should match", "email", handle.getColumnName());
        assertEquals("Column type should match", VarcharType.VARCHAR, handle.getColumnType());
    }

    @Test
    public void testColumnHandleImmutability() {
        BenoStreamDBColumnHandle handle = new BenoStreamDBColumnHandle("id", IntegerType.INTEGER);

        // Verify that getters return the same values (immutability)
        assertEquals("id", handle.getColumnName());
        assertEquals(IntegerType.INTEGER, handle.getColumnType());
    }

    // ---- Split Tests ----

    @Test
    public void testSplitConstruction() {
        BenoStreamDBSplit split = new BenoStreamDBSplit("seg_001", "s3://bucket/seg_001.parquet", "0-100");

        assertEquals("Segment ID should match", "seg_001", split.getSegmentId());
        assertEquals("Path should match", "s3://bucket/seg_001.parquet", split.getPath());
        assertEquals("Row selection should match", "0-100", split.getRowSelection());
    }

    @Test
    public void testSplitIsRemotelyAccessible() {
        BenoStreamDBSplit split = new BenoStreamDBSplit("seg_001", "s3://bucket/seg_001.parquet", "all");
        assertTrue("Split should be remotely accessible", split.isRemotelyAccessible());
    }

    @Test
    public void testSplitGetAddresses() {
        BenoStreamDBSplit split = new BenoStreamDBSplit("seg_001", "s3://bucket/seg_001.parquet", "all");
        var addresses = split.getAddresses();
        assertNotNull("Addresses should not be null", addresses);
        assertTrue("Addresses should be empty (managed by connector)", addresses.isEmpty());
    }

    @Test
    public void testSplitGetInfo() {
        BenoStreamDBSplit split = new BenoStreamDBSplit("seg_001", "s3://bucket/seg_001.parquet", "all");
        Object info = split.getInfo();
        assertSame("Info should return the split itself", split, info);
    }

    // ---- SplitManager Tests ----

    @Test
    public void testSplitManagerGetSplitsReturnsSource() {
        BenoStreamDBSplitManager splitManager = new BenoStreamDBSplitManager(
                BenoStreamDBTableUri.DEFAULT_WAREHOUSE, "auto");
        BenoStreamDBTableHandle tableHandle = new BenoStreamDBTableHandle("default", "test_table");

        var splitSource = splitManager.getSplits(null, null, tableHandle, null, null);
        assertNotNull("Split source should not be null", splitSource);
    }

    // ---- PageSourceProvider Tests ----

    @Test
    public void testPageSourceProviderCreate() {
        BenoStreamDBPageSourceProvider provider = new BenoStreamDBPageSourceProvider("auto");
        BenoStreamDBSplit split = new BenoStreamDBSplit("seg_001", "/tmp/test.parquet", "all");
        BenoStreamDBTableHandle tableHandle = new BenoStreamDBTableHandle("default", "test_table");
        List<ColumnHandle> columns = List.of(
                new BenoStreamDBColumnHandle("id", IntegerType.INTEGER),
                new BenoStreamDBColumnHandle("name", VarcharType.VARCHAR));

        var pageSource = provider.createPageSource(null, null, split, tableHandle, columns, null);
        assertNotNull("Page source should not be null", pageSource);
        assertTrue("Should be BenoStreamDBPageSource", pageSource instanceof BenoStreamDBPageSource);
    }

    // ---- Integration Test: Full Connector Flow ----

    @Test
    public void testIntegrationConnectorFactoryToMetadata() {
        BenoStreamDBConnectorFactory factory = new BenoStreamDBConnectorFactory();
        var connector = factory.create("test_catalog", Map.of(), null);

        var metadata = connector.getMetadata(null, null);
        var schemas = metadata.listSchemaNames(null);
        var tables = metadata.listTables(null, java.util.Optional.of("default"));

        assertNotNull("Schemas should not be null", schemas);
        assertNotNull("Tables should not be null", tables);
        assertTrue("Should have schemas", !schemas.isEmpty());
        assertTrue("Should have tables", !tables.isEmpty());
    }

    @Test
    public void testIntegrationFullQueryFlow() {
        // Simulate: Factory -> Connector -> Metadata -> TableHandle -> ColumnHandles ->
        // SplitManager -> PageSource
        BenoStreamDBConnectorFactory factory = new BenoStreamDBConnectorFactory();
        var connector = factory.create("test", Map.of(), null);

        var metadata = connector.getMetadata(null, null);
        SchemaTableName tableName = new SchemaTableName("default", "test_table");
        ConnectorTableHandle tableHandle = metadata.getTableHandle(null, tableName);

        var columnHandles = metadata.getColumnHandles(null, tableHandle);
        var splitManager = connector.getSplitManager();
        var pageSourceProvider = connector.getPageSourceProvider();

        assertNotNull("Metadata should not be null", metadata);
        assertNotNull("Table handle should not be null", tableHandle);
        assertNotNull("Column handles should not be null", columnHandles);
        assertNotNull("Split manager should not be null", splitManager);
        assertNotNull("Page source provider should not be null", pageSourceProvider);
        assertEquals("Should have 2 columns", 2, columnHandles.size());
    }

    // ---- Write / MERGE path tests ----

    @Test
    public void testConnectorGetPageSinkProvider() {
        BenoStreamDBConnectorFactory factory = new BenoStreamDBConnectorFactory();
        var connector = factory.create("test", Map.of(), null);

        var provider = connector.getPageSinkProvider();
        assertNotNull("Page sink provider should not be null", provider);
        assertTrue("Should be BenoStreamDBPageSinkProvider",
                provider instanceof BenoStreamDBPageSinkProvider);
    }

    @Test
    public void testBeginInsertReturnsHandle() {
        BenoStreamDBMetadata metadata = new BenoStreamDBMetadata();
        BenoStreamDBTableHandle table = new BenoStreamDBTableHandle("default", "test_table");
        List<ColumnHandle> columns = List.of(
                new BenoStreamDBColumnHandle("id", IntegerType.INTEGER),
                new BenoStreamDBColumnHandle("name", VarcharType.VARCHAR));

        var handle = metadata.beginInsert(null, table, columns,
                io.trino.spi.connector.RetryMode.NO_RETRIES);
        assertTrue("Should be BenoStreamDBInsertTableHandle",
                handle instanceof BenoStreamDBInsertTableHandle);

        BenoStreamDBInsertTableHandle insert = (BenoStreamDBInsertTableHandle) handle;
        assertEquals("default", insert.getSchemaName());
        assertEquals("test_table", insert.getTableName());
        assertEquals("Should carry the insert columns", 2, insert.getColumns().size());
    }

    @Test
    public void testBeginMergeReturnsHandle() {
        BenoStreamDBMetadata metadata = new BenoStreamDBMetadata();
        BenoStreamDBTableHandle table = new BenoStreamDBTableHandle("default", "test_table");

        var handle = metadata.beginMerge(null, table,
                io.trino.spi.connector.RetryMode.NO_RETRIES);
        assertTrue("Should be BenoStreamDBMergeTableHandle",
                handle instanceof BenoStreamDBMergeTableHandle);

        BenoStreamDBMergeTableHandle merge = (BenoStreamDBMergeTableHandle) handle;
        assertNotNull("Merge handle should wrap the table handle", merge.getTableHandle());
        assertNotNull("Merge handle should carry an insert handle", merge.getInsertHandle());
    }

    @Test
    public void testGetMergeRowIdColumnHandle() {
        BenoStreamDBMetadata metadata = new BenoStreamDBMetadata();
        BenoStreamDBTableHandle table = new BenoStreamDBTableHandle("default", "test_table");

        ColumnHandle rowId = metadata.getMergeRowIdColumnHandle(null, table);
        assertTrue("Should be BenoStreamDBColumnHandle", rowId instanceof BenoStreamDBColumnHandle);
        // The merge row id is exposed as the target's key (first) column.
        assertEquals("id", ((BenoStreamDBColumnHandle) rowId).getColumnName());
    }

    @Test
    public void testArrowConverterSchemaMapping() {
        List<BenoStreamDBColumnHandle> columns = List.of(
                new BenoStreamDBColumnHandle("id", IntegerType.INTEGER),
                new BenoStreamDBColumnHandle("name", VarcharType.VARCHAR));

        var schema = BenoStreamDBArrowConverter.toArrowSchema(columns);
        assertEquals("Should have 2 fields", 2, schema.getFields().size());
        assertEquals("id", schema.getFields().get(0).getName());
        assertEquals("name", schema.getFields().get(1).getName());
    }

    @Test
    public void testTableUriMapping() {
        assertEquals("s3://default/default/test_table",
                BenoStreamDBTableUri.of(BenoStreamDBTableUri.DEFAULT_WAREHOUSE, "default", "test_table"));
    }

    @Test
    public void testTableUriMappingCustomWarehouse() {
        assertEquals("s3://my-bucket/warehouse/default/test_table",
                BenoStreamDBTableUri.of("s3://my-bucket/warehouse", "default", "test_table"));
        // A trailing slash is normalized.
        assertEquals("s3://my-bucket/warehouse/default/test_table",
                BenoStreamDBTableUri.of("s3://my-bucket/warehouse/", "default", "test_table"));
    }

    @Test
    public void testConnectorFactoryWarehouseConfig() {
        BenoStreamDBConnectorFactory factory = new BenoStreamDBConnectorFactory();
        var connector = factory.create("test",
                Map.of("benostream.warehouse", "s3://my-bucket/wh"), null);
        assertNotNull("Connector should not be null", connector);

        var metadata = connector.getMetadata(null, null);
        assertTrue("Should be BenoStreamDBMetadata", metadata instanceof BenoStreamDBMetadata);
    }

    @Test
    public void testBeginCreateTableReturnsHandle() {
        BenoStreamDBMetadata metadata = new BenoStreamDBMetadata();
        var tableMetadata = new io.trino.spi.connector.ConnectorTableMetadata(
                new SchemaTableName("default", "new_table"),
                List.of(
                        new io.trino.spi.connector.ColumnMetadata("id", IntegerType.INTEGER),
                        new io.trino.spi.connector.ColumnMetadata("name", VarcharType.VARCHAR)));

        var handle = metadata.beginCreateTable(null, tableMetadata, java.util.Optional.empty(),
                io.trino.spi.connector.RetryMode.NO_RETRIES);
        assertTrue("Should be BenoStreamDBOutputTableHandle",
                handle instanceof BenoStreamDBOutputTableHandle);

        BenoStreamDBOutputTableHandle out = (BenoStreamDBOutputTableHandle) handle;
        assertEquals("default", out.getSchemaName());
        assertEquals("new_table", out.getTableName());
        assertEquals("Should carry the output columns", 2, out.getColumns().size());
    }

    @Test
    public void testSchemaJsonMapping() {
        List<BenoStreamDBColumnHandle> columns = List.of(
                new BenoStreamDBColumnHandle("id", IntegerType.INTEGER),
                new BenoStreamDBColumnHandle("name", VarcharType.VARCHAR));

        String json = BenoStreamDBArrowConverter.toSchemaJson(columns);
        assertTrue("Should contain the id field", json.contains("\"name\":\"id\""));
        assertTrue("Should map INTEGER to Int32", json.contains("\"type\":\"Int32\""));
        assertTrue("Should contain the name field", json.contains("\"name\":\"name\""));
        assertTrue("Should map VARCHAR to Utf8", json.contains("\"type\":\"Utf8\""));
    }
}
