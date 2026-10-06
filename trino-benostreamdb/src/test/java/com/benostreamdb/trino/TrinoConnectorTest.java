package com.benostreamdb.trino;

import io.trino.spi.connector.ColumnHandle;
import io.trino.spi.connector.ConnectorSplit;
import io.trino.spi.connector.ConnectorTableHandle;
import io.trino.spi.connector.SchemaTableName;
import io.trino.spi.type.IntegerType;
import io.trino.spi.type.VarcharType;
import org.junit.Assume;
import org.junit.Test;

import java.util.List;
import java.util.Map;

import static org.junit.Assert.*;

/**
 * Unit tests for the Trino BenoStream connector components.
 *
 * Tests that do not need the native engine run unconditionally. Tests that
 * exercise metadata/query paths are gated on {@link BenoStreamDBJNIBridge#isLoaded()}
 * so the suite still runs on a machine without {@code libbenostreamdb}.
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

    // ---- Handle equality (required by Trino's planner) ----

    @Test
    public void testColumnHandleValueEquality() {
        // Trino's MERGE planner compares ColumnHandles across separate
        // getColumnHandles() calls via List.indexOf (QueryPlanner.planMerge), so
        // value equality is mandatory. Two independently constructed handles for
        // the same column must be equal and share a hashCode.
        BenoStreamDBColumnHandle a = new BenoStreamDBColumnHandle("id", IntegerType.INTEGER);
        BenoStreamDBColumnHandle b = new BenoStreamDBColumnHandle("id", IntegerType.INTEGER);
        assertEquals(a, b);
        assertEquals(a.hashCode(), b.hashCode());
        assertNotEquals(a, new BenoStreamDBColumnHandle("value", IntegerType.INTEGER));
        assertNotEquals(a, new BenoStreamDBColumnHandle("id", VarcharType.VARCHAR));

        // The hidden row-id handle must also compare by value.
        assertEquals(BenoStreamDBColumnHandle.rowId("id"), BenoStreamDBColumnHandle.rowId("id"));
        assertNotEquals(BenoStreamDBColumnHandle.rowId("id"), BenoStreamDBColumnHandle.rowId("other"));
    }

    @Test
    public void testTableHandleValueEquality() {
        assertEquals(new BenoStreamDBTableHandle("s", "t"), new BenoStreamDBTableHandle("s", "t"));
        assertNotEquals(new BenoStreamDBTableHandle("s", "t"), new BenoStreamDBTableHandle("s", "u"));
    }

    @Test
    public void testSplitValueEquality() {
        assertEquals(new BenoStreamDBSplit("uri", "SELECT 1"), new BenoStreamDBSplit("uri", "SELECT 1"));
        assertNotEquals(new BenoStreamDBSplit("uri", "SELECT 1"), new BenoStreamDBSplit("uri", "SELECT 2"));
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

    // ---- Metadata: no mock fallback ----

    @Test
    public void testMetadataFailsLoudlyWithoutNative() {
        // The connector must never silently return mock metadata. When the
        // native library is absent, metadata calls throw.
        Assume.assumeFalse("native library is loaded; skip the no-native assertion",
                BenoStreamDBJNIBridge.isLoaded());

        BenoStreamDBMetadata metadata = new BenoStreamDBMetadata();
        try {
            metadata.listSchemaNames(null);
            fail("listSchemaNames should throw when the native library is unavailable");
        } catch (IllegalStateException expected) {
            // expected
        }
    }

    @Test
    public void testMetadataListSchemaNames() {
        Assume.assumeTrue(BenoStreamDBJNIBridge.isLoaded());
        BenoStreamDBMetadata metadata = new BenoStreamDBMetadata();
        var schemas = metadata.listSchemaNames(null);
        assertNotNull("Schema list should not be null", schemas);
    }

    @Test
    public void testMetadataListTables() {
        Assume.assumeTrue(BenoStreamDBJNIBridge.isLoaded());
        BenoStreamDBMetadata metadata = new BenoStreamDBMetadata();
        var tables = metadata.listTables(null, java.util.Optional.empty());
        assertNotNull("Table list should not be null", tables);
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

        assertEquals("id", handle.getColumnName());
        assertEquals(IntegerType.INTEGER, handle.getColumnType());
    }

    // ---- Split Tests ----

    @Test
    public void testSplitConstruction() {
        BenoStreamDBSplit split = new BenoStreamDBSplit(
                "s3://bucket/table", "SELECT * FROM t WHERE id = 1");

        assertEquals("Table URI should match", "s3://bucket/table", split.getTableUri());
        assertEquals("SQL should match", "SELECT * FROM t WHERE id = 1", split.getSql());
    }

    @Test
    public void testSplitIsRemotelyAccessible() {
        BenoStreamDBSplit split = new BenoStreamDBSplit("s3://bucket/table", "SELECT * FROM t");
        assertTrue("Split should be remotely accessible", split.isRemotelyAccessible());
    }

    @Test
    public void testSplitGetAddresses() {
        BenoStreamDBSplit split = new BenoStreamDBSplit("s3://bucket/table", "SELECT * FROM t");
        var addresses = split.getAddresses();
        assertNotNull("Addresses should not be null", addresses);
        assertTrue("Addresses should be empty (managed by connector)", addresses.isEmpty());
    }

    @Test
    public void testSplitGetSplitInfo() {
        BenoStreamDBSplit split = new BenoStreamDBSplit("s3://bucket/table", "SELECT * FROM t");
        // Trino 468 replaced ConnectorSplit.getInfo() with getSplitInfo().
        var info = split.getSplitInfo();
        assertNotNull("Split info should not be null", info);
    }

    // ---- SplitManager Tests ----

    @Test
    public void testSplitManagerEmitsSingleSplitWithPushedDownSql() throws Exception {
        BenoStreamDBSplitManager splitManager = new BenoStreamDBSplitManager(
                "s3://my-bucket/wh", "auto");
        BenoStreamDBTableHandle tableHandle = new BenoStreamDBTableHandle(
                "default", "events", java.util.Optional.of("severity = 'ERROR'"));

        var splitSource = splitManager.getSplits(null, null, tableHandle, null, null);
        assertNotNull("Split source should not be null", splitSource);

        List<ConnectorSplit> splits = splitSource.getNextBatch(1000).get().getSplits();
        assertEquals("Should emit exactly one split", 1, splits.size());

        BenoStreamDBSplit split = (BenoStreamDBSplit) splits.get(0);
        assertEquals("s3://my-bucket/wh/default/events", split.getTableUri());
        assertEquals("SELECT * FROM t WHERE severity = 'ERROR'", split.getSql());
    }

    @Test
    public void testSplitManagerWithoutFilter() throws Exception {
        BenoStreamDBSplitManager splitManager = new BenoStreamDBSplitManager(
                "s3://my-bucket/wh", "auto");
        BenoStreamDBTableHandle tableHandle = new BenoStreamDBTableHandle("default", "events");

        var splitSource = splitManager.getSplits(null, null, tableHandle, null, null);
        List<ConnectorSplit> splits = splitSource.getNextBatch(1000).get().getSplits();
        assertEquals("SELECT * FROM t", ((BenoStreamDBSplit) splits.get(0)).getSql());
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
                io.trino.spi.connector.RetryMode.NO_RETRIES, false);
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
