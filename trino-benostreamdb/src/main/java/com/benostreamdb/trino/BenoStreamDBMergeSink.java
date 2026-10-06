package com.benostreamdb.trino;

import io.airlift.slice.Slice;
import io.trino.spi.Page;
import io.trino.spi.block.Block;
import io.trino.spi.connector.ConnectorMergeSink;
import io.trino.spi.connector.MergePage;
import io.trino.spi.type.BigintType;
import io.trino.spi.type.BooleanType;
import io.trino.spi.type.DoubleType;
import io.trino.spi.type.IntegerType;
import io.trino.spi.type.RealType;
import io.trino.spi.type.Type;
import io.trino.spi.type.VarcharType;
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
 * {@code [dataColumns..., operation, case, rowId]}. The SPI {@link MergePage}
 * helper splits it into an insertions page (data columns) and a deletions page
 * (data columns + row id). Insert/update-insert rows are appended; delete and
 * update-delete rows are removed by the merge row id, which the connector
 * exposes as the target's primary key (see
 * {@link BenoStreamDBMetadata#getMergeRowIdColumnHandle}).</p>
 */
public class BenoStreamDBMergeSink implements ConnectorMergeSink {
    private final BenoStreamDBMergeTableHandle handle;
    private final String warehouse;
    private final String gpuDevice;
    private final BufferAllocator allocator = new RootAllocator();
    private final List<Page> insertPages = new ArrayList<>();
    /** Fully-formed delete predicates ({@code key = literal}) for the row ids. */
    private final List<String> deletePredicates = new ArrayList<>();

    public BenoStreamDBMergeSink(BenoStreamDBMergeTableHandle handle, String warehouse, String gpuDevice) {
        this.handle = handle;
        this.warehouse = warehouse;
        this.gpuDevice = gpuDevice;
    }

    @Override
    public void storeMergedRows(Page page) {
        // Trino 468 lays the merge page out as
        // [dataColumns..., operation (TINYINT), case (INTEGER), rowId]. The SPI
        // helper MergePage parses it into separate insertions/deletions pages, so
        // the connector does not hard-code column positions (this is exactly what
        // the reference Iceberg connector does).
        int dataColumnCount = handle.getInsertHandle().getColumns().size();
        MergePage mergePage = MergePage.createDeleteAndInsertPages(page, dataColumnCount);

        // The insertions page carries just the data columns, ready to append.
        mergePage.getInsertionsPage().ifPresent(insertPages::add);

        mergePage.getDeletionsPage().ifPresent(deletions -> {
            // The deletions page is [dataColumns..., rowId]; the merge row id
            // (the target's key column) is the last channel. Its type follows the
            // primary key, so non-numeric keys are quoted correctly.
            BenoStreamDBColumnHandle rowIdColumn = handle.getRowIdColumn();
            Type rowIdType = rowIdColumn.getColumnType();
            String keyColumn = rowIdColumn.getRowIdSourceColumn();
            Block rowIdBlock = deletions.getBlock(deletions.getChannelCount() - 1);
            for (int p = 0; p < deletions.getPositionCount(); p++) {
                if (!rowIdBlock.isNull(p)) {
                    deletePredicates.add(keyColumn + " = " + sqlLiteral(rowIdType, rowIdBlock, p));
                }
            }
        });
    }

    @Override
    public CompletableFuture<Collection<Slice>> finish() {
        BenoStreamDBInsertTableHandle insert = handle.getInsertHandle();
        String tableUri = BenoStreamDBTableUri.of(warehouse, insert.getSchemaName(), insert.getTableName());

        // DELETE before INSERT: the merge row id is the target's key column, so
        // an UPDATE (delete-old + insert-new) reuses the same key. Inserting
        // first would let the delete reap the freshly inserted row.
        for (String predicate : deletePredicates) {
            if (!BenoStreamDBJNIBridge.deleteRows(tableUri, predicate)) {
                throw new RuntimeException("BenoStreamDB deleteRows failed for " + tableUri + " (" + predicate + ")");
            }
        }

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

        insertPages.clear();
        deletePredicates.clear();
        return CompletableFuture.completedFuture(List.of());
    }

    @Override
    public void abort() {
        insertPages.clear();
        deletePredicates.clear();
    }

    /** Render a block value as a SQL literal for the engine's filter parser. */
    private static String sqlLiteral(Type type, Block block, int position) {
        if (type instanceof VarcharType) {
            String value = ((VarcharType) type).getSlice(block, position).toStringUtf8();
            return "'" + value.replace("'", "''") + "'";
        }
        if (type instanceof BigintType || type instanceof IntegerType) {
            return Long.toString(type.getLong(block, position));
        }
        if (type instanceof DoubleType) {
            return Double.toString(type.getDouble(block, position));
        }
        if (type instanceof RealType) {
            return Float.toString(Float.intBitsToFloat((int) type.getLong(block, position)));
        }
        if (type instanceof BooleanType) {
            return Boolean.toString(type.getBoolean(block, position));
        }
        throw new UnsupportedOperationException(
                "MERGE row id of type " + type + " is not supported; use a numeric, string, or boolean primary key");
    }
}
