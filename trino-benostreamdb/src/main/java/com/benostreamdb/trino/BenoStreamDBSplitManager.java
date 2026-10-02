package com.benostreamdb.trino;

import io.trino.spi.connector.*;

import java.util.List;

/**
 * Emits a single split per table.
 *
 * The engine's session does the parallelism internally, so there is no
 * file-range split fan-out. The predicate pushed down by
 * {@link BenoStreamDBMetadata#applyFilter} travels as a SQL {@code WHERE}
 * clause in the split, so the engine's planner applies its full pushdown
 * (scalar/inverted indexes and vector search) before decoding Parquet.
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
            DynamicFilter dynamicFilter,
            Constraint constraint) {

        BenoStreamDBTableHandle tableHandle = (BenoStreamDBTableHandle) table;
        String uri = BenoStreamDBTableUri.of(warehouse, tableHandle.getSchemaName(), tableHandle.getTableName());
        String filter = tableHandle.getFilterString().orElse("").trim();
        String sql = filter.isEmpty() ? "SELECT * FROM t" : "SELECT * FROM t WHERE " + filter;
        return new FixedSplitSource(List.of(new BenoStreamDBSplit(uri, sql)));
    }
}
