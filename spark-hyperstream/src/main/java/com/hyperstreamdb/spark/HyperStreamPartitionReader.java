package com.hyperstreamdb.spark;

import org.apache.spark.sql.connector.read.PartitionReader;
import org.apache.spark.sql.vectorized.ColumnarBatch;
import org.apache.spark.sql.vectorized.ColumnVector; // Spark's Vector
// import org.apache.spark.sql.vectorized.ArrowColumnVector; // Available in Spark 3.x+

import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.RootAllocator;

import java.io.IOException;

public class HyperStreamPartitionReader implements PartitionReader<ColumnarBatch> {

    private final HyperStreamPartition partition;
    private long nativeHandle = 0;
    private final RootAllocator allocator;
    private boolean finished = false;
    private ColumnarBatch currentBatch = null;

    static {
        try {
            System.loadLibrary("hyperstreamdb");
        } catch (UnsatisfiedLinkError e) {
            // Ignored for unit testing compilation
        }
    }

    // JNI
    private native long openSession(String path);

    private native long readBatch(long handle, long outArrayPtr, long outSchemaPtr);

    public HyperStreamPartitionReader(HyperStreamPartition partition, org.apache.spark.sql.types.StructType schema) {
        this.partition = partition;
        this.allocator = new RootAllocator();

        System.out.println("HyperStreamPartitionReader: Opening session for " + partition.getPath());
        try {
            this.nativeHandle = openSession(partition.getPath());
        } catch (UnsatisfiedLinkError e) {
            System.err.println("JNI not found, using mock");
            this.nativeHandle = 12345;
        }
    }

    @Override
    public boolean next() throws IOException {
        if (finished)
            return false;

        // 1. Allocate C Data Structs
        try (ArrowArray arrowArray = ArrowArray.allocateNew(allocator);
                ArrowSchema arrowSchema = ArrowSchema.allocateNew(allocator)) {

            // 2. Call Native
            long result = 0;
            try {
                result = readBatch(nativeHandle, arrowArray.memoryAddress(), arrowSchema.memoryAddress());
            } catch (UnsatisfiedLinkError e) {
            }

            if (result == 0) {
                finished = true;
                return false;
            }

            // 3. Import to Arrow VectorSchemaRoot
            try (VectorSchemaRoot root = Data.importVectorSchemaRoot(allocator, arrowArray, arrowSchema, null)) {

                // 4. Convert to Spark ColumnarBatch
                // In a real Spark env, we wrap the Arrow vectors in Spark's ArrowColumnVector
                // ColumnVector[] sparkVectors = new
                // ColumnVector[root.getFieldVectors().size()];
                // for (int i=0; i<sparkVectors.length; i++) {
                // sparkVectors[i] = new
                // org.apache.spark.sql.vectorized.ArrowColumnVector(root.getVector(i));
                // }
                // currentBatch = new ColumnarBatch(sparkVectors, root.getRowCount());

                // MOCK Implementation for compilation without full Spark runtime deps present
                currentBatch = new ColumnarBatch(new ColumnVector[0], root.getRowCount());
                return true;
            } catch (Exception e) {
                throw new IOException("Failed to import Arrow batch", e);
            }
        }
    }

    @Override
    public ColumnarBatch get() {
        return currentBatch;
    }

    @Override
    public void close() throws IOException {
        allocator.close();
    }
}
