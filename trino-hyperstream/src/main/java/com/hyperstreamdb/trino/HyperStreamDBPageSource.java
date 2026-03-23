package com.hyperstreamdb.trino;

import io.trino.spi.Page;
import io.trino.spi.connector.ColumnHandle;
import io.trino.spi.connector.ConnectorPageSource;
import io.trino.spi.block.BlockBuilder;
import io.trino.spi.type.IntegerType;
import io.trino.spi.type.VarcharType;
import java.util.List;
import java.io.IOException;

import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.VectorSchemaRoot;

public class HyperStreamDBPageSource implements ConnectorPageSource {

    private final HyperStreamDBSplit split;
    private final List<ColumnHandle> columns;
    private long nativeHandle = 0;
    private boolean finished = false;
    private final BufferAllocator allocator;

    // Load Native Lib
    static {
        try {
            System.loadLibrary("hyperstreamdb");
        } catch (UnsatisfiedLinkError e) {
            /* handled by SplitManager or ignored in tests */ }
    }

    // JNI Declarations
    private native long openSession(String path);

    // Updated signature: returns 1 for success/has_more, 0 for done/empty
    private native long readBatch(long handle, long outArrayPtr, long outSchemaPtr);

    public HyperStreamDBPageSource(HyperStreamDBSplit split, List<ColumnHandle> columns) {
        this.split = split;
        this.columns = columns;
        this.allocator = new RootAllocator();

        System.out.println("HyperStreamDBPageSource: Opening session for " + split.getPath());
        try {
            this.nativeHandle = openSession(split.getPath());
        } catch (UnsatisfiedLinkError e) {
            System.err.println("JNI openSession not found. using mock handle.");
            this.nativeHandle = 12345;
        }
    }

    @Override
    public long getCompletedBytes() {
        return 0;
    }

    @Override
    public long getReadTimeNanos() {
        return 0;
    }

    @Override
    public boolean isFinished() {
        return finished;
    }

    @Override
    public long getMemoryUsage() {
        return 0;
    }

    @Override
    public Page getNextPage() {
        if (finished)
            return null;

        // 1. Allocate C Data Interface Structures
        try (ArrowArray arrowArray = ArrowArray.allocateNew(allocator);
                ArrowSchema arrowSchema = ArrowSchema.allocateNew(allocator)) {

            // 2. Call Native (Passing memory addresses)
            long result = 0;
            try {
                result = readBatch(nativeHandle, arrowArray.memoryAddress(), arrowSchema.memoryAddress());
            } catch (UnsatisfiedLinkError e) {
                System.err.println("JNI readBatch not found.");
            }

            if (result == 0) {
                finished = true;
                return null;
            }

            // 3. Import data into Java Arrow Vector
            try (VectorSchemaRoot root = Data.importVectorSchemaRoot(allocator, arrowArray, arrowSchema, null)) {
                System.out.println("Java: Received Arrow Batch with " + root.getRowCount() + " rows");

                // 4. Convert Arrow VectorSchemaRoot to Trino Page
                return convertArrowToTrinoPage(root);
            } catch (Exception e) {
                throw new RuntimeException("Failed to import Arrow batch", e);
            }
        }
    }

    private Page convertArrowToTrinoPage(VectorSchemaRoot root) {
        // Real Implementation:
        // Iterate root.getFieldVectors(), match with 'columns', append to BlockBuilder

        // Ensure PageBuilder matches requested columns
        io.trino.spi.PageBuilder pageBuilder = new io.trino.spi.PageBuilder(
                columns.stream().map(c -> ((HyperStreamDBColumnHandle) c).getColumnType())
                        .collect(java.util.stream.Collectors.toList()));

        int rowCount = root.getRowCount();
        pageBuilder.declarePositions(rowCount);

        for (int i = 0; i < columns.size(); i++) {
            HyperStreamDBColumnHandle col = (HyperStreamDBColumnHandle) columns.get(i);
            BlockBuilder blockBuilder = pageBuilder.getBlockBuilder(i);

            // TODO: Match col.getColumnName() with root.getVector(name)
            // For now, defaulting to Mock value as verification of flow
            for (int r = 0; r < rowCount; r++) {
                if (col.getColumnName().equals("id")) {
                    IntegerType.INTEGER.writeLong(blockBuilder, 42); // Placeholder
                } else {
                    VarcharType.VARCHAR.writeString(blockBuilder, "Real Arrow Flow");
                }
            }
        }

        return pageBuilder.build();
    }

    @Override
    public void close() throws IOException {
        System.out.println("Closing HyperStreamDBPageSource handle: " + nativeHandle);
        allocator.close();
    }
}
