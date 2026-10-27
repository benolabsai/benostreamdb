package com.benostreamdb.trino;

import io.trino.spi.StandardErrorCode;
import io.trino.spi.TrinoException;
import io.trino.spi.connector.ConnectorSession;

/**
 * Stored procedures exposed under {@code system} by the Trino connector.
 *
 * <p>Argument names are uppercase because Trino's {@code Procedure.Argument}
 * constructor enforces uppercase names for CALL compatibility.</p>
 *
 * <p>The parameter types here are {@code String}, not {@link io.airlift.slice.Slice},
 * and that asymmetry is deliberate. Two independent checks apply to a VARCHAR
 * argument:</p>
 * <ul>
 *   <li>{@code Procedure.Argument} validates a declared <em>default value</em>
 *       against {@code type.getJavaType()} (Slice for varchar), so the defaults
 *       in the connector factory must be {@code Slices.utf8Slice(...)}.</li>
 *   <li>the CALL execution path supplies {@code String} values and rejects a
 *       handle whose parameter type is Slice with "Argument 'TABLE' has invalid
 *       type io.airlift.slice.Slice (expected java.lang.String)".</li>
 * </ul>
 */
public final class BenoStreamProcedures {

    private final String warehouse;

    public BenoStreamProcedures(String warehouse) {
        this.warehouse = warehouse;
    }

    public void addIndex(ConnectorSession session, String table, String column, String algorithm) {
        String uri = BenoStreamDBTableUri.resolve(warehouse, table);
        String resolvedAlgorithm = algorithm == null || algorithm.isBlank()
                ? "vector"
                : algorithm.trim();
        if (!BenoStreamDBJNIBridge.addIndex(uri, column, resolvedAlgorithm)) {
            throw new TrinoException(
                    StandardErrorCode.GENERIC_INTERNAL_ERROR,
                    "BenoStreamDB add_index failed for " + table + "." + column
                            + " using " + resolvedAlgorithm);
        }
    }

    public void buildIndex(ConnectorSession session, String table, String segmentId) {
        String uri = BenoStreamDBTableUri.resolve(warehouse, table);
        String target = segmentId == null || segmentId.isBlank() ? "all" : segmentId.trim();
        if (!BenoStreamDBJNIBridge.buildIndex(uri, target)) {
            throw new TrinoException(
                    StandardErrorCode.GENERIC_INTERNAL_ERROR,
                    "BenoStreamDB build_index failed for " + table + " (" + target + ")");
        }
    }

    public void dropIndex(ConnectorSession session, String table, String column) {
        String uri = BenoStreamDBTableUri.resolve(warehouse, table);
        if (!BenoStreamDBJNIBridge.dropIndex(uri, column, "")) {
            throw new TrinoException(
                    StandardErrorCode.GENERIC_INTERNAL_ERROR,
                    "BenoStreamDB drop_index failed for " + table + "." + column);
        }
    }

    /**
     * Spark's {@code rebuild_index(table, column)} rebuilds the segment named by
     * the column, which is what the engine expects as a segment id.
     */
    public void rebuildIndex(ConnectorSession session, String table, String column) {
        String uri = BenoStreamDBTableUri.resolve(warehouse, table);
        String target = column == null || column.isBlank() ? "all" : column.trim();
        if (!BenoStreamDBJNIBridge.buildIndex(uri, target)) {
            throw new TrinoException(
                    StandardErrorCode.GENERIC_INTERNAL_ERROR,
                    "BenoStreamDB rebuild_index failed for " + table + " (" + target + ")");
        }
    }

    public void compact(ConnectorSession session, String table) {
        String uri = BenoStreamDBTableUri.resolve(warehouse, table);
        if (!BenoStreamDBJNIBridge.compactTable(uri)) {
            throw new TrinoException(
                    StandardErrorCode.GENERIC_INTERNAL_ERROR,
                    "BenoStreamDB compact failed for " + table);
        }
    }

    public void setPrimaryKey(ConnectorSession session, String table, String columns) {
        String uri = BenoStreamDBTableUri.resolve(warehouse, table);
        if (!BenoStreamDBJNIBridge.setPrimaryKey(uri, columns)) {
            throw new TrinoException(
                    StandardErrorCode.GENERIC_INTERNAL_ERROR,
                    "BenoStreamDB set_primary_key failed for " + table + " (" + columns + ")");
        }
    }

    public void dropPrimaryKey(ConnectorSession session, String table, String columns) {
        String uri = BenoStreamDBTableUri.resolve(warehouse, table);
        if (!BenoStreamDBJNIBridge.dropPrimaryKey(uri, columns)) {
            throw new TrinoException(
                    StandardErrorCode.GENERIC_INTERNAL_ERROR,
                    "BenoStreamDB drop_primary_key failed for " + table + " (" + columns + ")");
        }
    }
}
