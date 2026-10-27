package com.benostreamdb.trino;

import io.trino.spi.Page;
import io.trino.spi.block.BlockBuilder;
import io.trino.spi.connector.ConnectorPageSource;
import io.trino.spi.connector.SourcePage;
import io.trino.spi.type.BigintType;
import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.VectorSchemaRoot;

/**
 * Runs the engine's DRIFT search once and exposes the discovered node ids.
 *
 * <p>Mirrors the Spark bridge's {@code runRegionalDriftSearch}: the engine
 * exports a single Arrow batch through the C data interface with a
 * {@code node_id} column, and only that column is read. The difference is that
 * this path returns the rows instead of discarding them.</p>
 */
public class RegionalDriftPageSource implements ConnectorPageSource {

    static final String NODE_ID = "node_id";

    private final Page page;
    private boolean finished;

    public RegionalDriftPageSource(RegionalDriftFunction.Split split) {
        this.page = runSearch(split);
    }

    private static Page runSearch(RegionalDriftFunction.Split split) {
        BlockBuilder nodeIds = BigintType.BIGINT.createBlockBuilder(null, 16);

        try (BufferAllocator allocator = new RootAllocator()) {
            ArrowArray arrowArray = ArrowArray.allocateNew(allocator);
            ArrowSchema arrowSchema = ArrowSchema.allocateNew(allocator);
            try {
                int status = BenoStreamDBJNIBridge.regionalDriftSearch(
                        split.getTableUri(),
                        split.getQuery(),
                        seedsJson(split.getSeeds()),
                        split.getTopK(),
                        split.getHops(),
                        split.getNDepth(),
                        split.getKFollowups(),
                        split.getMode(),
                        arrowArray.memoryAddress(),
                        arrowSchema.memoryAddress());

                // A negative status means the engine refused or failed; the same
                // signal the Spark bridge treats as "no results".
                if (status >= 0) {
                    try (VectorSchemaRoot root = Data.importVectorSchemaRoot(
                            allocator, arrowArray, arrowSchema, null)) {
                        BigIntVector vector = (BigIntVector) root.getVector(NODE_ID);
                        if (vector != null) {
                            for (int i = 0; i < root.getRowCount(); i++) {
                                if (vector.isNull(i)) {
                                    nodeIds.appendNull();
                                } else {
                                    BigintType.BIGINT.writeLong(nodeIds, vector.get(i));
                                }
                            }
                        }
                    }
                }
            } finally {
                arrowArray.close();
                arrowSchema.close();
            }
        }

        return new Page(nodeIds.build());
    }

    /** The engine expects a JSON array of numeric strings, as the Spark bridge sends. */
    private static String seedsJson(String seeds) {
        StringBuilder json = new StringBuilder("[");
        boolean first = true;
        for (String seed : seeds.split(",")) {
            String trimmed = seed.trim();
            if (trimmed.isEmpty()) {
                continue;
            }
            if (!first) {
                json.append(',');
            }
            json.append('"').append(trimmed).append('"');
            first = false;
        }
        return json.append(']').toString();
    }

    @Override
    public SourcePage getNextSourcePage() {
        if (finished) {
            return null;
        }
        finished = true;
        return SourcePage.create(page);
    }

    @Override
    public boolean isFinished() {
        return finished;
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
    public long getMemoryUsage() {
        return 0;
    }

    @Override
    public void close() {
    }
}
