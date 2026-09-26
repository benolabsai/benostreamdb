package com.benostreamdb.trino;

import io.airlift.slice.Slice;
import io.trino.spi.Page;
import io.trino.spi.connector.ConnectorPageSink;
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
 * Writes Trino pages into a BenoStreamDB table. Pages are buffered and flushed
 * as a single Arrow batch on {@link #finish()}.
 */
public class BenoStreamDBPageSink implements ConnectorPageSink {
    private final BenoStreamDBInsertTableHandle handle;
    private final String warehouse;
    private final String gpuDevice;
    private final boolean createTable;
    private final BufferAllocator allocator = new RootAllocator();
    private final List<Page> pages = new ArrayList<>();

    public BenoStreamDBPageSink(BenoStreamDBInsertTableHandle handle, String warehouse, String gpuDevice) {
        this(handle, warehouse, gpuDevice, false);
    }

    public BenoStreamDBPageSink(BenoStreamDBInsertTableHandle handle, String warehouse, String gpuDevice,
            boolean createTable) {
        this.handle = handle;
        this.warehouse = warehouse;
        this.gpuDevice = gpuDevice;
        this.createTable = createTable;
    }

    @Override
    public CompletableFuture<?> appendPage(Page page) {
        pages.add(page);
        return NOT_BLOCKED;
    }

    @Override
    public CompletableFuture<Collection<Slice>> finish() {
        if (!pages.isEmpty() || createTable) {
            String tableUri = BenoStreamDBTableUri.of(warehouse, handle.getSchemaName(), handle.getTableName());
            if (createTable) {
                String schemaJson = BenoStreamDBArrowConverter.toSchemaJson(handle.getColumns());
                if (!BenoStreamDBJNIBridge.createTable(tableUri, schemaJson)) {
                    throw new RuntimeException("BenoStreamDB createTable failed for " + tableUri);
                }
            }
            if (!pages.isEmpty()) {
                try (VectorSchemaRoot root =
                        BenoStreamDBArrowConverter.toArrow(handle.getColumns(), pages, allocator);
                        ArrowArray array = ArrowArray.allocateNew(allocator);
                        ArrowSchema schema = ArrowSchema.allocateNew(allocator)) {
                    Data.exportVectorSchemaRoot(allocator, root, null, array, schema);
                    // The native side takes ownership of the exported buffers
                    // (Arrow C Data Interface). `close()` frees only the Java-side
                    // struct buffers — it does NOT invoke the C release callback —
                    // so the data is released exactly once, by the native side.
                    boolean ok = BenoStreamDBJNIBridge.appendBatch(
                            tableUri, array.memoryAddress(), schema.memoryAddress());
                    if (!ok) {
                        throw new RuntimeException("BenoStreamDB appendBatch failed for " + tableUri);
                    }
                }
            }
        }
        pages.clear();
        return CompletableFuture.completedFuture(List.of());
    }

    @Override
    public void abort() {
        pages.clear();
    }
}
