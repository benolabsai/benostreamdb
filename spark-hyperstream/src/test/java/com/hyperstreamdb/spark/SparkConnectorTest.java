package com.hyperstreamdb.spark;

import org.apache.spark.sql.connector.catalog.TableCapability;
import org.apache.spark.sql.connector.read.ScanBuilder;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;
import org.junit.Before;
import org.junit.Test;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.*;

/**
 * Unit tests for the Spark HyperStream connector components.
 * Tests schema handling, table configuration, and partition planning without
 * requiring native libraries.
 */
public class SparkConnectorTest {

    private StructType testSchema;
    private CaseInsensitiveStringMap options;

    @Before
    public void setUp() {
        testSchema = new StructType()
                .add("id", DataTypes.IntegerType)
                .add("name", DataTypes.StringType)
                .add("embedding", DataTypes.createArrayType(DataTypes.DoubleType, true));

        Map<String, String> opts = new HashMap<>();
        opts.put("path", "file:///tmp/test_hyperstream_table");
        options = new CaseInsensitiveStringMap(opts);
    }

    @Test
    public void testDefaultSourceInferSchema() {
        DefaultSource source = new DefaultSource();
        CaseInsensitiveStringMap emptyOptions = new CaseInsensitiveStringMap(Collections.emptyMap());

        StructType schema = source.inferSchema(emptyOptions);
        assertNotNull("Inferred schema should not be null", schema);
        assertEquals("Schema should have 2 fields", 2, schema.fields().length);
        assertEquals("First field should be id", "id", schema.fields()[0].name());
        assertEquals("Second field should be name", "name", schema.fields()[1].name());
    }

    @Test
    public void testDefaultSourceGetTable() {
        DefaultSource source = new DefaultSource();
        Map<String, String> properties = new HashMap<>();
        properties.put("path", "s3://bucket/table");

        org.apache.spark.sql.connector.catalog.Table table = source.getTable(
                testSchema,
                new org.apache.spark.sql.connector.expressions.Transform[0],
                properties);
        assertNotNull("Table should not be null", table);
        assertEquals("Table name should be hyperstream_table", "hyperstream_table", table.name());
        assertSame("Table schema should match input", testSchema, table.schema());
    }

    @Test
    public void testDefaultSourceSupportsExternalMetadata() {
        DefaultSource source = new DefaultSource();
        assertTrue("Should support external metadata", source.supportsExternalMetadata());
    }

    @Test
    public void testHyperStreamTableCapabilities() {
        Map<String, String> properties = new HashMap<>();
        properties.put("path", "file:///tmp/test");
        HyperStreamTable table = new HyperStreamTable(testSchema, properties);

        Set<TableCapability> caps = table.capabilities();
        assertNotNull("Capabilities should not be null", caps);
        assertTrue("Should support BATCH_READ", caps.contains(TableCapability.BATCH_READ));
    }

    @Test
    public void testHyperStreamTableSchema() {
        Map<String, String> properties = new HashMap<>();
        HyperStreamTable table = new HyperStreamTable(testSchema, properties);

        assertSame("Schema should be preserved", testSchema, table.schema());
        assertEquals("Schema field count should match", 3, table.schema().fields().length);
    }

    @Test
    public void testHyperStreamTableNewScanBuilder() {
        Map<String, String> properties = new HashMap<>();
        HyperStreamTable table = new HyperStreamTable(testSchema, properties);

        ScanBuilder scanBuilder = table.newScanBuilder(options);
        assertNotNull("Scan builder should not be null", scanBuilder);
        assertTrue("Should be HyperStreamScanBuilder", scanBuilder instanceof HyperStreamScanBuilder);
    }

    @Test
    public void testHyperStreamScanBuilderBuild() {
        HyperStreamScanBuilder scanBuilder = new HyperStreamScanBuilder(testSchema, options);
        org.apache.spark.sql.connector.read.Scan scan = scanBuilder.build();

        assertNotNull("Scan should not be null", scan);
        assertSame("Scan should return itself", scanBuilder, scan);
    }

    @Test
    public void testHyperStreamScanBuilderReadSchema() {
        HyperStreamScanBuilder scanBuilder = new HyperStreamScanBuilder(testSchema, options);

        assertSame("Read schema should match constructor schema", testSchema, scanBuilder.readSchema());
    }

    @Test
    public void testHyperStreamScanBuilderToBatch() {
        HyperStreamScanBuilder scanBuilder = new HyperStreamScanBuilder(testSchema, options);
        org.apache.spark.sql.connector.read.Batch batch = scanBuilder.toBatch();

        assertNotNull("Batch should not be null", batch);
        assertSame("Batch should return itself", scanBuilder, batch);
    }

    @Test
    public void testHyperStreamScanBuilderPlanInputPartitionsWithMock() {
        // When native lib is not available, should fallback to mock partition
        HyperStreamScanBuilder scanBuilder = new HyperStreamScanBuilder(testSchema, options);
        org.apache.spark.sql.connector.read.InputPartition[] partitions = scanBuilder.planInputPartitions();

        assertNotNull("Partitions should not be null", partitions);
        assertTrue("Should have at least one partition (mock)", partitions.length >= 1);
    }

    @Test
    public void testHyperStreamScanBuilderCreateReaderFactory() {
        HyperStreamScanBuilder scanBuilder = new HyperStreamScanBuilder(testSchema, options);
        org.apache.spark.sql.connector.read.PartitionReaderFactory factory = scanBuilder.createReaderFactory();

        assertNotNull("Reader factory should not be null", factory);
        assertTrue("Should be HyperStreamPartitionReaderFactory", factory instanceof HyperStreamPartitionReaderFactory);
    }

    @Test
    public void testHyperStreamPartitionConstruction() {
        HyperStreamPartition partition = new HyperStreamPartition("seg_001", "file:///data/seg_001.parquet");

        assertEquals("Segment ID should match", "seg_001", partition.getSegmentId());
        assertEquals("Path should match", "file:///data/seg_001.parquet", partition.getPath());
    }

    @Test
    public void testHyperStreamPartitionReaderFactorySupportsColumnar() {
        HyperStreamPartitionReaderFactory factory = new HyperStreamPartitionReaderFactory(testSchema);
        HyperStreamPartition partition = new HyperStreamPartition("seg_001", "file:///data/seg_001.parquet");

        assertTrue("Should support columnar reads", factory.supportColumnarReads(partition));
    }

    @Test
    public void testHyperStreamPartitionReaderFactoryCreateColumnarReader() {
        HyperStreamPartitionReaderFactory factory = new HyperStreamPartitionReaderFactory(testSchema);
        HyperStreamPartition partition = new HyperStreamPartition("seg_001", "file:///data/seg_001.parquet");

        org.apache.spark.sql.connector.read.PartitionReader reader = factory.createColumnarReader(partition);
        assertNotNull("Reader should not be null", reader);
        assertTrue("Should be HyperStreamPartitionReader", reader instanceof HyperStreamPartitionReader);
    }

    @Test(expected = UnsupportedOperationException.class)
    public void testHyperStreamPartitionReaderFactoryRowReaderUnsupported() {
        HyperStreamPartitionReaderFactory factory = new HyperStreamPartitionReaderFactory(testSchema);
        HyperStreamPartition partition = new HyperStreamPartition("seg_001", "file:///data/seg_001.parquet");

        // Row-based reading is not optimized
        factory.createReader(partition);
    }

    @Test
    public void testScanBuilderWithNullPathOption() {
        CaseInsensitiveStringMap emptyOptions = new CaseInsensitiveStringMap(Collections.emptyMap());
        HyperStreamScanBuilder scanBuilder = new HyperStreamScanBuilder(testSchema, emptyOptions);

        // Should not throw, should fallback to default S3 path
        org.apache.spark.sql.connector.read.InputPartition[] partitions = scanBuilder.planInputPartitions();
        assertNotNull("Partitions should not be null even with missing path", partitions);
    }

    @Test
    public void testIntegrationScanBuilderEndToEnd() {
        // Simulate the full scan builder flow: build -> toBatch -> planPartitions ->
        // createReaderFactory
        HyperStreamScanBuilder scanBuilder = new HyperStreamScanBuilder(testSchema, options);

        org.apache.spark.sql.connector.read.Scan scan = scanBuilder.build();
        org.apache.spark.sql.connector.read.Batch batch = scan.toBatch();
        org.apache.spark.sql.connector.read.InputPartition[] partitions = batch.planInputPartitions();
        org.apache.spark.sql.connector.read.PartitionReaderFactory factory = batch.createReaderFactory();

        assertNotNull("Scan should not be null", scan);
        assertNotNull("Batch should not be null", batch);
        assertNotNull("Partitions should not be null", partitions);
        assertNotNull("Factory should not be null", factory);
        assertTrue("Should have partitions", partitions.length > 0);
    }
}
