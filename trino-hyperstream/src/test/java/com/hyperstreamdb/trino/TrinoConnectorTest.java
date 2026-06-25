package com.hyperstreamdb.trino;

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
 * Unit tests for the Trino HyperStream connector components.
 * Tests metadata handling, split management, and column handles without
 * requiring native libraries.
 */
public class TrinoConnectorTest {

    // ---- Plugin Tests ----

    @Test
    public void testPluginReturnsConnectorFactory() {
        HyperStreamDBPlugin plugin = new HyperStreamDBPlugin();
        var factories = plugin.getConnectorFactories();

        assertNotNull("Connector factories should not be null", factories);
        int count = 0;
        for (io.trino.spi.connector.ConnectorFactory factory : factories) {
            count++;
            assertTrue("Factory should be HyperStreamDBConnectorFactory",
                    factory instanceof HyperStreamDBConnectorFactory);
        }
        assertEquals("Should have exactly one factory", 1, count);
    }

    // ---- ConnectorFactory Tests ----

    @Test
    public void testConnectorFactoryName() {
        HyperStreamDBConnectorFactory factory = new HyperStreamDBConnectorFactory();
        assertEquals("Connector name should be hyperstreamdb", "hyperstreamdb", factory.getName());
    }

    @Test
    public void testConnectorFactoryCreate() {
        HyperStreamDBConnectorFactory factory = new HyperStreamDBConnectorFactory();
        var connector = factory.create("test_catalog", Map.of(), null);

        assertNotNull("Connector should not be null", connector);
    }

    @Test
    public void testConnectorGetMetadata() {
        HyperStreamDBConnectorFactory factory = new HyperStreamDBConnectorFactory();
        var connector = factory.create("test", Map.of(), null);

        var metadata = connector.getMetadata(null, null);
        assertNotNull("Metadata should not be null", metadata);
        assertTrue("Should be HyperStreamDBMetadata", metadata instanceof HyperStreamDBMetadata);
    }

    @Test
    public void testConnectorGetSplitManager() {
        HyperStreamDBConnectorFactory factory = new HyperStreamDBConnectorFactory();
        var connector = factory.create("test", Map.of(), null);

        var splitManager = connector.getSplitManager();
        assertNotNull("Split manager should not be null", splitManager);
        assertTrue("Should be HyperStreamDBSplitManager", splitManager instanceof HyperStreamDBSplitManager);
    }

    @Test
    public void testConnectorGetPageSourceProvider() {
        HyperStreamDBConnectorFactory factory = new HyperStreamDBConnectorFactory();
        var connector = factory.create("test", Map.of(), null);

        var provider = connector.getPageSourceProvider();
        assertNotNull("Page source provider should not be null", provider);
        assertTrue("Should be HyperStreamDBPageSourceProvider", provider instanceof HyperStreamDBPageSourceProvider);
    }

    @Test
    public void testConnectorBeginTransaction() {
        HyperStreamDBConnectorFactory factory = new HyperStreamDBConnectorFactory();
        var connector = factory.create("test", Map.of(), null);

        var tx = connector.beginTransaction(
                io.trino.spi.transaction.IsolationLevel.SERIALIZABLE, true, true);
        assertNotNull("Transaction handle should not be null", tx);
        assertTrue("Should be HyperStreamDBTransactionHandle",
                tx instanceof HyperStreamDBConnectorFactory.HyperStreamDBTransactionHandle);
    }

    // ---- Metadata Tests ----

    @Test
    public void testMetadataListSchemaNames() {
        HyperStreamDBMetadata metadata = new HyperStreamDBMetadata();
        var schemas = metadata.listSchemaNames(null);

        assertNotNull("Schema list should not be null", schemas);
        assertEquals("Should have default schema", 1, schemas.size());
        assertEquals("Schema should be default", "default", schemas.get(0));
    }

    @Test
    public void testMetadataGetTableHandle() {
        HyperStreamDBMetadata metadata = new HyperStreamDBMetadata();
        SchemaTableName tableName = new SchemaTableName("default", "my_table");

        ConnectorTableHandle handle = metadata.getTableHandle(null, tableName);
        assertNotNull("Table handle should not be null", handle);
        assertTrue("Should be HyperStreamDBTableHandle", handle instanceof HyperStreamDBTableHandle);

        HyperStreamDBTableHandle tableHandle = (HyperStreamDBTableHandle) handle;
        assertEquals("Schema name should match", "default", tableHandle.getSchemaName());
        assertEquals("Table name should match", "my_table", tableHandle.getTableName());
    }

    @Test
    public void testMetadataGetTableMetadata() {
        HyperStreamDBMetadata metadata = new HyperStreamDBMetadata();
        HyperStreamDBTableHandle tableHandle = new HyperStreamDBTableHandle("default", "test_table");

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
        HyperStreamDBMetadata metadata = new HyperStreamDBMetadata();
        var tables = metadata.listTables(null, java.util.Optional.of("default"));

        assertNotNull("Table list should not be null", tables);
        assertEquals("Should have one test table", 1, tables.size());
        assertEquals("Table should be test_table", "test_table", tables.get(0).getTableName());
    }

    @Test
    public void testMetadataGetColumnHandles() {
        HyperStreamDBMetadata metadata = new HyperStreamDBMetadata();
        HyperStreamDBTableHandle tableHandle = new HyperStreamDBTableHandle("default", "test_table");

        Map<String, ColumnHandle> columnHandles = metadata.getColumnHandles(null, tableHandle);
        assertNotNull("Column handles should not be null", columnHandles);
        assertEquals("Should have 2 columns", 2, columnHandles.size());
        assertTrue("Should have id column", columnHandles.containsKey("id"));
        assertTrue("Should have name column", columnHandles.containsKey("name"));

        HyperStreamDBColumnHandle idHandle = (HyperStreamDBColumnHandle) columnHandles.get("id");
        assertEquals("id column type should be INTEGER", IntegerType.INTEGER, idHandle.getColumnType());

        HyperStreamDBColumnHandle nameHandle = (HyperStreamDBColumnHandle) columnHandles.get("name");
        assertEquals("name column type should be VARCHAR", VarcharType.VARCHAR, nameHandle.getColumnType());
    }

    @Test
    public void testMetadataGetColumnMetadata() {
        HyperStreamDBMetadata metadata = new HyperStreamDBMetadata();
        HyperStreamDBTableHandle tableHandle = new HyperStreamDBTableHandle("default", "test_table");
        HyperStreamDBColumnHandle columnHandle = new HyperStreamDBColumnHandle("id", IntegerType.INTEGER);

        var columnMetadata = metadata.getColumnMetadata(null, tableHandle, columnHandle);
        assertNotNull("Column metadata should not be null", columnMetadata);
        assertEquals("Column name should match", "id", columnMetadata.getName());
        assertEquals("Column type should match", IntegerType.INTEGER, columnMetadata.getType());
    }

    // ---- TableHandle Tests ----

    @Test
    public void testTableHandleGetters() {
        HyperStreamDBTableHandle handle = new HyperStreamDBTableHandle("public", "users");

        assertEquals("Schema name should match", "public", handle.getSchemaName());
        assertEquals("Table name should match", "users", handle.getTableName());
    }

    // ---- ColumnHandle Tests ----

    @Test
    public void testColumnHandleGetters() {
        HyperStreamDBColumnHandle handle = new HyperStreamDBColumnHandle("email", VarcharType.VARCHAR);

        assertEquals("Column name should match", "email", handle.getColumnName());
        assertEquals("Column type should match", VarcharType.VARCHAR, handle.getColumnType());
    }

    @Test
    public void testColumnHandleImmutability() {
        HyperStreamDBColumnHandle handle = new HyperStreamDBColumnHandle("id", IntegerType.INTEGER);

        // Verify that getters return the same values (immutability)
        assertEquals("id", handle.getColumnName());
        assertEquals(IntegerType.INTEGER, handle.getColumnType());
    }

    // ---- Split Tests ----

    @Test
    public void testSplitConstruction() {
        HyperStreamDBSplit split = new HyperStreamDBSplit("seg_001", "s3://bucket/seg_001.parquet", "0-100");

        assertEquals("Segment ID should match", "seg_001", split.getSegmentId());
        assertEquals("Path should match", "s3://bucket/seg_001.parquet", split.getPath());
        assertEquals("Row selection should match", "0-100", split.getRowSelection());
    }

    @Test
    public void testSplitIsRemotelyAccessible() {
        HyperStreamDBSplit split = new HyperStreamDBSplit("seg_001", "s3://bucket/seg_001.parquet", "all");
        assertTrue("Split should be remotely accessible", split.isRemotelyAccessible());
    }

    @Test
    public void testSplitGetAddresses() {
        HyperStreamDBSplit split = new HyperStreamDBSplit("seg_001", "s3://bucket/seg_001.parquet", "all");
        var addresses = split.getAddresses();
        assertNotNull("Addresses should not be null", addresses);
        assertTrue("Addresses should be empty (managed by connector)", addresses.isEmpty());
    }

    @Test
    public void testSplitGetInfo() {
        HyperStreamDBSplit split = new HyperStreamDBSplit("seg_001", "s3://bucket/seg_001.parquet", "all");
        Object info = split.getInfo();
        assertSame("Info should return the split itself", split, info);
    }

    // ---- SplitManager Tests ----

    @Test
    public void testSplitManagerGetSplitsReturnsSource() {
        HyperStreamDBSplitManager splitManager = new HyperStreamDBSplitManager("auto");
        HyperStreamDBTableHandle tableHandle = new HyperStreamDBTableHandle("default", "test_table");

        var splitSource = splitManager.getSplits(null, null, tableHandle, null, null);
        assertNotNull("Split source should not be null", splitSource);
    }

    // ---- PageSourceProvider Tests ----

    @Test
    public void testPageSourceProviderCreate() {
        HyperStreamDBPageSourceProvider provider = new HyperStreamDBPageSourceProvider("auto");
        HyperStreamDBSplit split = new HyperStreamDBSplit("seg_001", "/tmp/test.parquet", "all");
        HyperStreamDBTableHandle tableHandle = new HyperStreamDBTableHandle("default", "test_table");
        List<ColumnHandle> columns = List.of(
                new HyperStreamDBColumnHandle("id", IntegerType.INTEGER),
                new HyperStreamDBColumnHandle("name", VarcharType.VARCHAR));

        var pageSource = provider.createPageSource(null, null, split, tableHandle, columns, null);
        assertNotNull("Page source should not be null", pageSource);
        assertTrue("Should be HyperStreamDBPageSource", pageSource instanceof HyperStreamDBPageSource);
    }

    // ---- Integration Test: Full Connector Flow ----

    @Test
    public void testIntegrationConnectorFactoryToMetadata() {
        HyperStreamDBConnectorFactory factory = new HyperStreamDBConnectorFactory();
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
        HyperStreamDBConnectorFactory factory = new HyperStreamDBConnectorFactory();
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
}
