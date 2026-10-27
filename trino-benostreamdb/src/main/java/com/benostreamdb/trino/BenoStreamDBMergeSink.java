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
    /**
     * Keys per {@code key IN (...)} predicate. A MERGE that touches 5,000 rows
     * used to issue 5,000 sequential JNI calls, each with its own engine
     * transaction; the same 5,000 rows now cost 5 calls.
     */
    private static final int MAX_KEYS_PER_IN_LIST = 1000;

    /**
     * Upper bound on the rendered predicate length. Identifier-heavy keys can
     * still grow the string, so the flush loop also respects this budget.
     */
    private static final int MAX_PREDICATE_CHARS = 64 * 1024;

    private final BenoStreamDBMergeTableHandle handle;
    private final String warehouse;
    private final String gpuDevice;
    private final BufferAllocator allocator = new RootAllocator();
    private final List<Page> insertPages = new ArrayList<>();
    /** Rendered SQL literals of the merge row ids pending deletion. */
    private final List<String> deleteKeyLiterals = new ArrayList<>();

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
            Block rowIdBlock = deletions.getBlock(deletions.getChannelCount() - 1);
            for (int p = 0; p < deletions.getPositionCount(); p++) {
                if (!rowIdBlock.isNull(p)) {
                    deleteKeyLiterals.add(sqlLiteral(rowIdType, rowIdBlock, p));
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
        deleteRows(tableUri);

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
        deleteKeyLiterals.clear();
        return CompletableFuture.completedFuture(List.of());
    }

    @Override
    public void abort() {
        insertPages.clear();
        deleteKeyLiterals.clear();
    }

    /**
     * Apply the buffered row-id deletions as a bounded number of
     * {@code key IN (...)} predicates instead of one JNI call per row.
     */
    private void deleteRows(String tableUri) {
        if (deleteKeyLiterals.isEmpty()) {
            return;
        }
        String keyColumn = handle.getRowIdColumn().getRowIdSourceColumn();

        int start = 0;
        while (start < deleteKeyLiterals.size()) {
            int end = Math.min(start + MAX_KEYS_PER_IN_LIST, deleteKeyLiterals.size());
            // Respect the predicate length budget as well as the key-count cap.
            while (end > start + 1 && renderedLength(keyColumn, deleteKeyLiterals, start, end) > MAX_PREDICATE_CHARS) {
                end = start + Math.max(1, (end - start) / 2);
            }
            String predicate = keyColumn + " IN (" + String.join(", ", deleteKeyLiterals.subList(start, end)) + ")";
            if (!BenoStreamDBJNIBridge.deleteRows(tableUri, predicate)) {
                throw new RuntimeException("BenoStreamDB deleteRows failed for " + tableUri + " (" + predicate + ")");
            }
            start = end;
        }
        deleteKeyLiterals.clear();
    }

    /** Approximate rendered length of {@code column IN (literals[start:end])}. */
    private static int renderedLength(String keyColumn, List<String> literals, int start, int end) {
        int length = keyColumn.length() + " IN ()".length();
        for (int i = start; i < end; i++) {
            length += literals.get(i).length() + 2; // literal + ", "
        }
        return length;
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
