package com.benostreamdb.trino;

import io.airlift.slice.Slice;
import io.trino.spi.Page;
import io.trino.spi.block.Block;
import io.trino.spi.connector.ConnectorMergeSink;
import io.trino.spi.type.BigintType;
import io.trino.spi.type.IntegerType;
import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * Applies MERGE row-level operations produced by Trino.
 *
 * <p>Trino sends each merged row as a page laid out as
 * {@code [rowId, operation, ...dataColumns]}. Insert/update-insert rows are
 * appended; delete/update-delete rows are removed by the merge row id, which the
 * connector exposes as the target's key column (see
 * {@link BenoStreamDBMetadata#getMergeRowIdColumnHandle}).</p>
 */
public class BenoStreamDBMergeSink implements ConnectorMergeSink {
    private final BenoStreamDBMergeTableHandle handle;
    private final String warehouse;
    private final String gpuDevice;
    private final BufferAllocator allocator = new RootAllocator();
    private final List<Page> insertPages = new ArrayList<>();
    private final List<Long> deleteRowIds = new ArrayList<>();

    public BenoStreamDBMergeSink(BenoStreamDBMergeTableHandle handle, String warehouse, String gpuDevice) {
        this.handle = handle;
        this.warehouse = warehouse;
        this.gpuDevice = gpuDevice;
    }

    @Override
    public void storeMergedRows(Page page) {
        int positions = page.getPositionCount();
        Block rowIdBlock = page.getBlock(0);
        Block opBlock = page.getBlock(1);
        for (int p = 0; p < positions; p++) {
            long rowId = BigintType.BIGINT.getLong(rowIdBlock, p);
            int operation = (int) IntegerType.INTEGER.getLong(opBlock, p);
            switch (operation) {
                case INSERT_OPERATION_NUMBER:
                case UPDATE_INSERT_OPERATION_NUMBER:
                    insertPages.add(extractDataPage(page, p));
                    break;
                case DELETE_OPERATION_NUMBER:
                case UPDATE_DELETE_OPERATION_NUMBER:
                    deleteRowIds.add(rowId);
                    break;
                default:
                    throw new IllegalArgumentException("Unknown MERGE operation number: " + operation);
            }
        }
    }

    /** A single-row page containing only the data columns (rowId/operation dropped). */
    private Page extractDataPage(Page page, int position) {
        int dataColumns = page.getChannelCount() - 2;
        Block[] blocks = new Block[dataColumns];
        for (int i = 0; i < dataColumns; i++) {
            blocks[i] = page.getBlock(i + 2).getRegion(position, 1);
        }
        return new Page(blocks);
    }

    @Override
    public CompletableFuture<Collection<Slice>> finish() {
        BenoStreamDBInsertTableHandle insert = handle.getInsertHandle();
        String tableUri = BenoStreamDBTableUri.of(warehouse, insert.getSchemaName(), insert.getTableName());

        if (!insertPages.isEmpty()) {
            try (VectorSchemaRoot root =
                    BenoStreamDBArrowConverter.toArrow(insert.getColumns(), insertPages, allocator);
                    ArrowArray array = ArrowArray.allocateNew(allocator);
                    ArrowSchema schema = ArrowSchema.allocateNew(allocator)) {
                Data.exportVectorSchemaRoot(allocator, root, null, array, schema);
                // The native side takes ownership of the exported buffers; `close()`
                // frees only the Java-side struct buffers (not the data).
                if (!BenoStreamDBJNIBridge.appendBatch(
                        tableUri, array.memoryAddress(), schema.memoryAddress())) {
                    throw new RuntimeException("BenoStreamDB appendBatch failed for " + tableUri);
                }
            }
        }

        if (!deleteRowIds.isEmpty()) {
            // The merge row id is the target's key column (see getMergeRowIdColumnHandle).
            String keyColumn = insert.getColumns().get(0).getColumnName();
            for (long rowId : deleteRowIds) {
                if (!BenoStreamDBJNIBridge.deleteRows(tableUri, keyColumn + " = " + rowId)) {
                    throw new RuntimeException("BenoStreamDB deleteRows failed for " + tableUri);
                }
            }
        }

        insertPages.clear();
        deleteRowIds.clear();
        return CompletableFuture.completedFuture(List.of());
    }

    @Override
    public void abort() {
        insertPages.clear();
        deleteRowIds.clear();
    }
}
