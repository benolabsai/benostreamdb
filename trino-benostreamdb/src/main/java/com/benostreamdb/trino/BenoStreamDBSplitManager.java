package com.benostreamdb.trino;

import io.trino.spi.connector.*;

import java.util.List;
import java.util.Set;

/**
 * Emits a single split per table.
 *
 * The engine's session does the parallelism internally, so there is no
 * file-range split fan-out. The predicate pushed down by
 * {@link AbstractBenoStreamDBMetadata#applyFilter} travels as a SQL {@code WHERE}
 * clause in the split, so the engine applies scalar and inverted-index pruning
 * before decoding Parquet.
 *
 * <p>Table functions ({@code vector_search}, {@code hybrid_search},
 * {@code graph_*}) hand over a complete statement instead: when
 * {@code pushdownSql} is present it replaces the generated
 * {@code SELECT * FROM t [WHERE ...]}, which is what lets the native engine run
 * KNN, RRF, or traversal itself rather than post-filtering in Trino.</p>
 */
public class BenoStreamDBSplitManager implements ConnectorSplitManager {
    private final String warehouse;
    private final String gpuDevice;

    public BenoStreamDBSplitManager(String warehouse, String gpuDevice) {
        this.warehouse = warehouse;
        this.gpuDevice = gpuDevice;
    }

    @Override
    public ConnectorSplitSource getSplits(
            ConnectorTransactionHandle transaction,
            ConnectorSession session,
            ConnectorTableHandle table,
            Set<ColumnHandle> columns,
            Constraint constraint) {

        if (table instanceof RegionalDriftFunction.Handle driftHandle) {
            // The whole search is one unit of work; the split carries the
            // resolved request so the page source performs it exactly once.
            String uri = RegionalDriftFunction.resolveUri(warehouse, driftHandle);
            return new FixedSplitSource(List.of(driftHandle.toSplit(uri)));
        }

        if (table instanceof ListIndexesTableHandle listHandle) {
            // Index metadata is produced by the engine's manifest, not by a scan,
            // so it is fetched here and carried in the split as JSON.
            String uri = ListIndexesFunction.resolveUri(warehouse, listHandle);
            String json = BenoStreamDBJNIBridge.listIndexes(uri);
            return new FixedSplitSource(List.of(new ListIndexesSplit(json)));
        }

        BenoStreamDBTableHandle tableHandle = (BenoStreamDBTableHandle) table;
        String uri = BenoStreamDBTableUri.of(warehouse, tableHandle.getSchemaName(), tableHandle.getTableName());
        String filter = tableHandle.getFilterString().orElse("").trim();
        String sql = tableHandle.getPushdownSql().orElseGet(
                () -> filter.isEmpty() ? "SELECT * FROM t" : "SELECT * FROM t WHERE " + filter);
        return new FixedSplitSource(List.of(new BenoStreamDBSplit(
                uri, sql, tableHandle.getHybridSearch(), tableHandle.getTablesJson())));
    }
}
