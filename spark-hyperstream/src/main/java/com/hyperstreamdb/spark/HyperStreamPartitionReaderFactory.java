package com.hyperstreamdb.spark;

import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.connector.read.InputPartition;
import org.apache.spark.sql.connector.read.PartitionReader;
import org.apache.spark.sql.connector.read.PartitionReaderFactory;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.vectorized.ColumnarBatch;

public class HyperStreamPartitionReaderFactory implements PartitionReaderFactory {

    private final StructType schema;

    public HyperStreamPartitionReaderFactory(StructType schema) {
        this.schema = schema;
    }

    @Override
    public PartitionReader<InternalRow> createReader(InputPartition partition) {
        // We prefer Columnar Reader for perf, but fallback to Row Reader if needed.
        // For Vectorized support, we must implement supportColumnarReads(partition) in
        // Scan.
        throw new UnsupportedOperationException("Row-based reading not optimized. Use Columnar.");
    }

    @Override
    public PartitionReader<ColumnarBatch> createColumnarReader(InputPartition partition) {
        return new HyperStreamPartitionReader((HyperStreamPartition) partition, schema);
    }

    @Override
    public boolean supportColumnarReads(InputPartition partition) {
        return true;
    }
}
