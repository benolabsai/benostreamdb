package com.benostreamdb.trino;

import io.trino.spi.Page;
import io.trino.spi.connector.ColumnHandle;
import io.trino.spi.connector.ConnectorPageSource;
import io.trino.spi.block.BlockBuilder;
import java.util.List;
import java.io.IOException;

import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorSchemaRoot;

/**
 * Streams a BenoStreamDB SQL query result into Trino pages.
 *
 * The query is opened once through the JNI bridge (which runs it via the
 * engine's DataFusion session, applying index/vector pushdown) and batches are
 * pulled until exhausted. There is no mock fallback: if the native library is
 * unavailable the page source fails loudly.
 */
public class BenoStreamDBPageSource implements ConnectorPageSource {

    private final List<ColumnHandle> columns;
    private final BufferAllocator allocator;
    private long queryHandle = 0;
    private boolean finished = false;

    public BenoStreamDBPageSource(BenoStreamDBSplit split, List<ColumnHandle> columns, String gpuDevice) {
        this.columns = columns;
        this.allocator = new RootAllocator();

        if (!BenoStreamDBJNIBridge.isLoaded()) {
            throw new IllegalStateException(
                    "BenoStreamDB native library (libbenostreamdb) is not loaded; "
                            + "add it to java.library.path");
        }
        BenoStreamDBJNIBridge.setGpuContext(gpuDevice);
        this.queryHandle = BenoStreamDBJNIBridge.openQuery(split.getTableUri(), split.getSql());
        if (this.queryHandle == 0) {
            throw new RuntimeException("BenoStreamDB openQuery failed for " + split.getTableUri());
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
        if (finished) {
            return null;
        }

        try (ArrowArray arrowArray = ArrowArray.allocateNew(allocator);
                ArrowSchema arrowSchema = ArrowSchema.allocateNew(allocator)) {

            long result = BenoStreamDBJNIBridge.readQueryBatch(
                    queryHandle, arrowArray.memoryAddress(), arrowSchema.memoryAddress());
            if (result == 0) {
                finished = true;
                return null;
            }

            try (VectorSchemaRoot root = Data.importVectorSchemaRoot(allocator, arrowArray, arrowSchema, null)) {
                return convertArrowToTrinoPage(root);
            } catch (Exception e) {
                throw new RuntimeException("Failed to import Arrow batch", e);
            }
        }
    }

    private Page convertArrowToTrinoPage(VectorSchemaRoot root) {
        io.trino.spi.PageBuilder pageBuilder = new io.trino.spi.PageBuilder(
                columns.stream().map(c -> ((BenoStreamDBColumnHandle) c).getColumnType())
                        .collect(java.util.stream.Collectors.toList()));

        int rowCount = root.getRowCount();
        pageBuilder.declarePositions(rowCount);

        for (int i = 0; i < columns.size(); i++) {
            BenoStreamDBColumnHandle col = (BenoStreamDBColumnHandle) columns.get(i);
            BlockBuilder blockBuilder = pageBuilder.getBlockBuilder(i);

            FieldVector vector = root.getVector(col.getColumnName());

            for (int r = 0; r < rowCount; r++) {
                if (vector == null || vector.isNull(r)) {
                    blockBuilder.appendNull();
                    continue;
                }

                io.trino.spi.type.Type trinoType = col.getColumnType();
                Object obj = vector.getObject(r);

                writeTrinoObject(trinoType, blockBuilder, obj);
            }
        }

        return pageBuilder.build();
    }

    private void writeTrinoObject(io.trino.spi.type.Type trinoType, BlockBuilder blockBuilder, Object obj) {
        if (obj == null) {
            blockBuilder.appendNull();
            return;
        }

        if (trinoType instanceof io.trino.spi.type.ArrayType) {
            io.trino.spi.type.ArrayType arrayType = (io.trino.spi.type.ArrayType) trinoType;
            io.trino.spi.type.Type elementType = arrayType.getElementType();
            java.util.List<?> list = (java.util.List<?>) obj;
            // Trino 435: build nested values via `buildEntry` (the old
            // `beginBlockEntry`/`closeEntry` pair was removed from BlockBuilder).
            ((io.trino.spi.block.ArrayBlockBuilder) blockBuilder).buildEntry(elementBuilder -> {
                for (Object element : list) {
                    writeTrinoObject(elementType, elementBuilder, element);
                }
            });
        } else if (trinoType instanceof io.trino.spi.type.RowType) {
            io.trino.spi.type.RowType rowType = (io.trino.spi.type.RowType) trinoType;
            java.util.Map<?, ?> map = (java.util.Map<?, ?>) obj;
            ((io.trino.spi.block.RowBlockBuilder) blockBuilder).buildEntry(fieldBuilders -> {
                java.util.List<io.trino.spi.type.RowType.Field> fields = rowType.getFields();
                for (int i = 0; i < fields.size(); i++) {
                    Object element = map.get(fields.get(i).getName().orElse(""));
                    writeTrinoObject(fields.get(i).getType(), fieldBuilders.get(i), element);
                }
            });
        } else if (trinoType instanceof io.trino.spi.type.IntegerType
                || trinoType instanceof io.trino.spi.type.BigintType) {
            trinoType.writeLong(blockBuilder, ((Number) obj).longValue());
        } else if (trinoType instanceof io.trino.spi.type.DoubleType) {
            trinoType.writeDouble(blockBuilder, ((Number) obj).doubleValue());
        } else if (trinoType instanceof io.trino.spi.type.RealType) {
            trinoType.writeLong(blockBuilder, Float.floatToRawIntBits(((Number) obj).floatValue()));
        } else if (trinoType instanceof io.trino.spi.type.BooleanType) {
            trinoType.writeBoolean(blockBuilder, (Boolean) obj);
        } else {
            io.trino.spi.type.VarcharType.VARCHAR.writeString(blockBuilder, obj.toString());
        }
    }

    @Override
    public void close() throws IOException {
        if (queryHandle != 0) {
            BenoStreamDBJNIBridge.closeQuery(queryHandle);
            queryHandle = 0;
        }
        allocator.close();
    }
}
